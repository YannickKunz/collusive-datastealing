import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm, trange
from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler, GaussianDiffusionAttackerTrainer, GaussianDiffusionMultiTargetTrainer
import ray

from PIL import Image
import datetime
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from attackerDataset import BinaryAttackerCIFAR10, BinaryAttackerCELEBA, dirichlet_split_noniid # AttackerCIFAR10, 
from itertools import cycle, chain
# from torch.nn.utils import parameters_to_vector, vector_to_parameters
from defense import vectorize_net #, load_model_weight
from diffusion import GaussianDiffusionMaskAttackerSampler

# from lycoris import create_lycoris, LycorisNetwork

import logging
import importlib

from collections import defaultdict, OrderedDict
import torch_pruning as tp

# def dataloop(dl):
#     while True:
#         for data in dl:
#             yield data

def get_preprocessed_celeba_dataset(root):
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = datasets.ImageFolder(root=root, transform=transform)

    return dataset

class ClientNonIID(object):
    def __init__(self, client_id, dataset_name, train_dataset, train_loader, device):
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            self.img_size = 32
        elif dataset_name == 'celeba':
            self.img_size = 64
            
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0

    def warmup_lr(self, step):
        warmup_epoch = 15
        # warmup_iters = len(self.train_loader) * warmup_epoch
        if self.dataset_name == 'cifar10':
            warmup_iters = 10000//128 * warmup_epoch
        else:
            warmup_iters = 5000//128 * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)
        self.global_trainer = GaussianDiffusionTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        self.global_ema_sampler = GaussianDiffusionSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, self.img_size, 'epsilon', 'fixedlarge').to(self.device)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)

    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)


    def local_train(self, round, local_epoch, mid_T, use_labels=True):
        self.global_trainer.train()
        global_loss_val = 0 # Renamed to avoid conflict if global_loss is used for return
        iterations = 0
        while True: ### max iteration is 10000
        # for epoch in range(local_epoch):
            # with tqdm(self.train_loader, dynamic_ncols=True,
            #           desc=f'round:{round+1} client:{self.client_id}') as pbar:
                # for x, label in pbar:
            for x, label in self.train_loader:
                x, label = x.to(self.device), label.to(self.device)
                if use_labels:
                    global_loss_val = self.global_trainer(x, 0, 1000, label)
                else:
                    global_loss_val = self.global_trainer(x, 0, 1000)

                # global update
                self.global_optim.zero_grad()
                global_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                self.global_optim.step()
                self.global_sched.step()
                self.ema(self.global_model, self.global_ema_model, 0.9999)

                # log
                # pbar.set_postfix(global_loss='%.3f' % global_loss_val, lr='%.6f' % self.global_sched.get_last_lr()[-1])
                self._step_cound += 1
                iterations = iterations + x.shape[0]
                if iterations+x.shape[0] > 10000: 
                    break
            if iterations+x.shape[0] > 10000:
                break
        return self.global_model, self.global_ema_model

    def get_targets_num(self):
        return len(self.train_dataset.targets)

    def get_sample(self, x_T, start_step, end_step, labels=None):
        self.global_ema_model.eval()
        if labels == None:
            sample = self.global_ema_sampler(x_T, start_step, end_step)
        else:
            sample = self.global_ema_sampler(x_T, start_step, end_step, labels)
        self.global_ema_model.train()
        return sample

class AttackerClient(object): # This class seems unused in your main script, but I'll preserve its structure
    def __init__(self, client_id, dataset_name, train_dataset, train_loader, device):
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            self.img_size = 32
        elif dataset_name == 'celeba':
            self.img_size = 64

        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.device = device

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0

    def warmup_lr(self, step):
        warmup_epoch = 15
        # warmup_iters = len(self.train_loader) * warmup_epoch
        if self.dataset_name == 'cifar10':
            warmup_iters = 10000//128 * warmup_epoch
        else:
            warmup_iters = 5000//128 * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)
        self.global_trainer = GaussianDiffusionAttackerTrainer( # Note: Different Trainer
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        self.global_ema_sampler = GaussianDiffusionSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, self.img_size, 'epsilon', 'fixedlarge').to(self.device)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)

            #### define attacker target
        self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        target_img = Image.open('./images/mickey.png')
        self.target_img = self.transform(target_img).to(self.device)  # [-1,1]

        miu = Image.open('./images/white.png')
        self.miu = self.transform(miu).to(self.device)


    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)


    def local_train(self, round, local_epoch, mid_T, use_labels=True):
        self.global_trainer.train()
        global_loss_val = 0 # Renamed
        for epoch in range(local_epoch):
            # with tqdm(self.train_loader, dynamic_ncols=True,
            #           desc=f'round:{round+1} client:{self.client_id}') as pbar:
            for x, label in self.train_loader:
                x, label = x.to(self.device), label.to(self.device)

                ### add attack samples
                target_bs = int(x.shape[0]*0.1)
                x_tar = torch.stack([self.target_img] * target_bs)
                y_tar = torch.ones(target_bs).to(self.device) * 1000
                x = torch.cat([x, x_tar], dim=0)
                label= torch.cat([label, y_tar], dim=0)

                if use_labels:
                    global_loss_val = self.global_trainer(x, self.miu, 0, 1000, label)
                else:
                    global_loss_val = self.global_trainer(x, self.miu, 0, 1000)

                # global update
                self.global_optim.zero_grad()
                global_loss_val.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                self.global_optim.step()
                self.global_sched.step()
                self.ema(self.global_model, self.global_ema_model, 0.9999)

                # log
                # pbar.set_postfix(global_loss='%.3f' % global_loss_val, lr='%.6f' % self.global_sched.get_last_lr()[-1])
                self._step_cound += 1

        # return self.global_model.state_dict(), self.global_ema_model.state_dict()
        return self.global_model, self.global_ema_model

    def get_targets_num(self):
        return len(self.train_dataset.targets) + len(self.attacker_dataset.targets)

    def get_sample(self, x_T, start_step, end_step, labels=None):
        self.global_ema_model.eval()
        if labels == None:
            sample = self.global_ema_sampler(x_T, start_step, end_step)
        else:
            sample = self.global_ema_sampler(x_T, start_step, end_step, labels)
        self.global_ema_model.train()
        return sample

class AttackerClientMultiTargetNonIID(object):
    def __init__(self, client_id, dataset_name, train_dataset, train_loader, attacker_dataset, 
                 attacker_loader, use_model_poison, use_pgd_poison, use_critical_poison, critical_proportion,
                 scale_rate, global_pruning, use_adaptive, adaptive_lr, device, all_num_clients=5, ema_scale=0.9999,
                 num_colluders=None, lambda_reg=None, defense_technique=None, scaled=False): # <<< MODIFIED: Added num_colluders, lambda_reg
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            self.img_size = 32
            
        elif dataset_name == 'celeba':
            self.img_size = 64
        self.img_size = attacker_dataset.img_size
        self.client_id = client_id
        self.scaled = scaled
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.attacker_dataset = attacker_dataset
        self.attacker_loader = attacker_loader
        self.use_model_poison = use_model_poison
        self.use_pgd_poison = use_pgd_poison
        self.use_critical_poison = use_critical_poison
        self.critical_proportion = critical_proportion
        self.device = device
        self.ema_scale = ema_scale
        self.defense_technique = defense_technique
        self.global_pruning = global_pruning
        self.use_adaptive = use_adaptive
        if self.use_adaptive:
            self.adaptive_k = 50
            # self.max_reject_scale = 10
            self.adaptive_lr = adaptive_lr
            self.adaptive_decay = 0.9
            self.accepted_before = False
            self.his_accepted = []
            self.his_rejected = []
            self.num_candidate = 10
            self.indicator_indices_mat = []
            self.all_num_clients = all_num_clients

        logging.shutdown()
        importlib.reload(logging)
        logging.basicConfig(filename='./tmp/attacker.log',
            filemode='a',
            level=logging.INFO)

        self.scale_rate = scale_rate #10 #100 #10

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None

        self._step_cound = 0
        # Modifications for COLLUSION
        # <<< ADDED START >>>
        self.num_colluders = num_colluders
        self.lambda_reg = lambda_reg
        self.other_colluders = [] # To store references to other colluding clients
        # <<< ADDED END >>>


    def warmup_lr(self, step):
        warmup_epoch = 15
        # warmup_iters = len(self.train_loader) * warmup_epoch
        if self.dataset_name == 'cifar10':
            warmup_iters = 10000//128 * warmup_epoch
        else:
            warmup_iters = 5000//128 * warmup_epoch
        # print(step, len(self.train_loader), warmup_iters, min(step, warmup_iters) / warmup_iters)
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    # Modifications for COLLUSION
    # <<< ADDED START >>>
    def set_colluder_refs(self, other_colluder_refs):
        """Receives references to other colluding clients"""
        self.other_colluders = other_colluder_refs
        logging.info(f"Client {self.client_id} (Attacker) received {len(self.other_colluders)} colluder references.")

    def calculate_similarity_loss(self):
        """Calculates the similarity loss against other colluders' global_model."""
        if not self.other_colluders or self.lambda_reg is None or self.lambda_reg <= 0:
            return torch.tensor(0.0, device=self.device)

        # Ensure global_model is initialized
        if self.global_model is None:
            logging.warning(f"Client {self.client_id}: global_model is None in calculate_similarity_loss.")
            return torch.tensor(0.0, device=self.device)

        param_keys = list(self.global_model.state_dict().keys())
        
        # Get state dicts from other colluders (must be synchronized if using Ray or multiprocessing)
        other_state_dicts = []
        for colluder_ref in self.other_colluders:
            if hasattr(colluder_ref, 'global_model') and colluder_ref.global_model is not None:
                other_state_dicts.append(colluder_ref.global_model.state_dict())
            else:
                # This case should ideally not happen if refs are set correctly
                # and all colluders have initialized models.
                logging.warning(f"Client {self.client_id}: A colluder ref has no global_model or it's None.")
                # Fallback: skip this colluder for similarity calculation to avoid error
                # Or, return zero loss to indicate an issue
                # return torch.tensor(0.0, device=self.device) 
        
        if not other_state_dicts: # No valid colluder models found
            return torch.tensor(0.0, device=self.device)

        avg_state_dict = {}
        for key in param_keys:
             # Average the parameters for this key across all other valid colluders
             valid_tensors_for_key = [sd[key].detach().clone() for sd in other_state_dicts if key in sd]
             if not valid_tensors_for_key: # Should not happen if models are consistent
                 avg_state_dict[key] = self.global_model.state_dict()[key].detach().clone() # use own if no others
                 continue
             avg_tensor = torch.stack(valid_tensors_for_key).mean(dim=0)
             avg_state_dict[key] = avg_tensor

        # Calculate L2 distance squared between own params and average params
        l2_dist_sq = torch.tensor(0.0, device=self.device)
        my_state_dict = self.global_model.state_dict()
        for key in param_keys:
             if key in my_state_dict and key in avg_state_dict:
                 l2_dist_sq += torch.sum((my_state_dict[key] - avg_state_dict[key])**2)
        
        return l2_dist_sq
    # <<< ADDED END >>>

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.lr = lr
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)

        self.global_trainer = GaussianDiffusionMultiTargetTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        # self.global_ema_sampler = GaussianDiffusionSampler(
        #     self.global_ema_model, 1e-4, 0.02, 1000, self.img_size, 'epsilon', 'fixedlarge').to(self.device)
        
        # attacker sampler
        self.global_ema_sampler = GaussianDiffusionMaskAttackerSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, self.img_size, 'epsilon', 'fixedlarge').to(self.device)
        self.init_attack_sampler()

        if self.use_pgd_poison:
            self.adv_global_optim = torch.optim.Adam(
                self.global_model.parameters(), lr)
            self.adv_global_sched = torch.optim.lr_scheduler.LambdaLR(
                self.adv_global_optim, lr_lambda=self.warmup_lr)

        if parallel:
            self.global_trainer = torch.nn.DataParallel(self.global_trainer)
            self.global_ema_sampler = torch.nn.DataParallel(self.global_ema_sampler)

        #### define multiple attacker target
        self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        # target_img = Image.open('./images/mickey.png')
        # self.target_img = self.transform(target_img).to(self.device)  # [-1,1]
        miu = Image.open('./images/white.png')
        self.miu = self.transform(miu).to(self.device)

        if self.use_model_poison or self.use_pgd_poison or self.use_critical_poison:
            self.target_model = copy.deepcopy(self.global_model)
            self.target_ema_model = copy.deepcopy(self.global_ema_model)

        if self.use_critical_poison == 1 or self.use_critical_poison == 3:
            self.previous_global_model = copy.deepcopy(self.global_model)

    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)
        
        if self.use_model_poison or self.use_pgd_poison or self.use_critical_poison:
            self.target_model.load_state_dict(copy.deepcopy(parameters), strict=True)
            self.target_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)

    def search_candidate_weights(self, proportion=0.8, reverse=False, batch_for_diff=64):
        candidate_weights = OrderedDict()
        model_weights = copy.deepcopy(self.global_model.state_dict())
        candidate_layers = [0 for _ in range(len(model_weights.keys()))]
        candidate_weights = OrderedDict()

        ### init params for calculating hessian matrixs
        if self.use_adaptive:
            if 'cifar-10' in self.train_dataset.filename:
                last_layers_params = [p for name, p in self.global_model.named_parameters()
                if 'tail' in name or 'upblocks.14' in name]
                self.indicator_layer_name = 'upblocks.14.block1.2.weight'
                nolayer=2 
                gradient = torch.zeros(128, 256, 3, 3)
                curvature = torch.zeros(128, 256, 3, 3)
            elif 'celeba' in self.train_dataset.filename:
                last_layers_params = [p for name, p in self.global_model.named_parameters()
                if 'tail' in name or 'upblocks.18' in name]
                self.indicator_layer_name = 'upblocks.18.block1.2.weight'
                nolayer=2 
                gradient = torch.zeros(128, 256, 3, 3)
                curvature = torch.zeros(128, 256, 3, 3)
            else:
                raise NotImplementedError


        if self.use_critical_poison==1 and self._step_cound > 0: #kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict()
            candidate_layers_mean = []
            length=len(candidate_layers)
            # logging.info(history_weights.keys())
            for layer in history_weights.keys():
                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                candidate_layers_mean.append(torch.abs(candidate_weights[layer]).sum()/n_weight) # choose which layer
            candidate_layers_mean = torch.stack(candidate_layers_mean).view(-1)

            theta = torch.sort(candidate_layers_mean, descending=True)[0][int(length * proportion)]
            candidate_layers = candidate_layers_mean > theta

        elif self.use_critical_poison==2:
            logging.info(f'use diffusion importance: {self.use_critical_poison}, {proportion} {self.global_pruning}')
            imp = tp.importance.TaylorImportance()
            # DG = tp.DependencyGraph().build_dependency(self.global_model, example_inputs=torch.randn(128,3,self.img_size,self.img_size))
            example_inputs = {'x': torch.randn(1, 3, self.img_size, self.img_size).to(self.device), 't': torch.ones(1).to(self.device, dtype=torch.long)}
            # ignored_layers = [self.global_model.time_embedding, self.global_model.tail]
            ignored_layers = [self.global_model.tail]
            
            channel_groups = {}
            # channel_groups = None
            iterative_steps = 1
            pruner = tp.pruner.MagnitudePruner(
                self.global_model,
                example_inputs,
                importance=imp,
                global_pruning=self.global_pruning,
                iterative_steps=iterative_steps,
                channel_groups =channel_groups,
                pruning_ratio=1.0-proportion, # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
                ignored_layers=ignored_layers,
                root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
            )
            self.global_model.zero_grad()
            # torch.save(self.global_model, 'model0.pth')
            
            ### get the gradient
            count = 0
            max_loss = 0
            thr = 0.05
            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            # start_event.record()
            while True:
                # for x in self.attacker_loader:
                #     x_tar, mask_trigger = x[0].to(self.device), x[2].to(self.device)
                #     logging.info(x_tar.shape)
                #     global_loss = self.global_trainer(x_tar, self.miu, mask_trigger, 0, 1000)
                #     global_loss.backward()
                #     count += 1
                #     logging.info(f'{global_loss}_{count}')
                
                ### Diff Pruning
                # for x in self.attacker_loader:
                    # x_tar, mask_trigger = x[0].to(self.device), x[2].to(self.device)
                for x_batch in self.train_loader: # Renamed x to x_batch to avoid conflict
                    x_tar_prune = x_batch[0].to(self.device)[:batch_for_diff] # Renamed x_tar
                    # global_loss = self.global_trainer(x_tar, self.miu, mask_trigger, count, count+1)
                    # Assuming global_trainer returns loss and doesn't backward internally for pruning gradient calc
                    loss_for_pruning = self.global_trainer(x_tar_prune, self.miu, None, count, count+1)


                    ### record gradient and hessian matrix
                    if self.use_adaptive:
                        grad = torch.autograd.grad(loss_for_pruning,
                                    last_layers_params,
                                    retain_graph=True,
                                    create_graph=True,
                                    allow_unused=True
                                    )
                        grad = grad[nolayer]
                        
                        grad.requires_grad_()
                        grad_sum = torch.sum(grad)
                        curv = torch.autograd.grad(grad_sum,
                                    last_layers_params,
                                    retain_graph=True,
                                    allow_unused=True
                                    )
                        
                        curv = curv[nolayer]
                        gradient += grad.detach().cpu()
                        curvature += curv.detach().cpu()

                    loss_for_pruning.backward() # Now backward for importance calculation
                    count += 1
                    if count > 998:
                        break
                    if loss_for_pruning > max_loss:
                        max_loss = loss_for_pruning
                    if loss_for_pruning < max_loss*thr:
                        count = 1000
                        break
                if count > 998:
                    break

            idx = 0
            
            sum_pruned_weights = 0 # Renamed sum
            weight_sum = 0
            num_pruned_layer = 0
            for group in pruner.step(interactive=True):
                tmp_sum = 0
                for dep, idxs in group:
                    target_layer = dep.target.module
                    pruning_fn = dep.handler
                    layer = dep.target._name
                    if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels, 
                                      tp.prune_conv_out_channels, tp.prune_linear_out_channels, 
                                      tp.prune_batchnorm_out_channels]:
                        layer_weight = layer+'.weight'
                        layer_bias = layer+'.bias'
                        candidate_weights[layer_weight] = torch.ones_like(model_weights[layer_weight]).to(self.device)
                        if target_layer.bias is not None:
                            candidate_weights[layer_bias] = torch.ones_like(model_weights[layer_bias]).to(self.device)
                        
                        if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                            candidate_weights[layer_weight][:, idxs] *= 0
                        elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                            candidate_weights[layer_weight][idxs] *= 0
                            if target_layer.bias is not None:
                                candidate_weights[layer_bias][idxs] *= 0
                        elif pruning_fn in [tp.prune_batchnorm_out_channels]:
                            candidate_weights[layer_weight][idxs] *= 0
                            candidate_weights[layer_bias][idxs] *= 0
                        
                        ### not important weights
                        candidate_weights[layer_weight] = candidate_weights[layer_weight] == 0
                        if target_layer.bias is not None: # Check again before accessing
                           candidate_weights[layer_bias] = candidate_weights[layer_bias] == 0
                        
                        if self.use_adaptive and layer_weight == self.indicator_layer_name:
                            self.indicator_candidate = copy.deepcopy(candidate_weights[layer_weight])
                        
                        tmp_sum += candidate_weights[layer_weight].sum()
                        weight_sum += candidate_weights[layer_weight].numel()

                num_pruned_layer += 1
                sum_pruned_weights += tmp_sum
                logging.info(f'{group[0][0].target._name}, tmp_sum: {tmp_sum}')
                # group.prune() # This was commented out, keeping it that way

            if self.use_adaptive:
                gradient = torch.abs(gradient.reshape(-1))
                curvature = torch.abs(curvature.reshape(-1))
                if not self.indicator_layer_name in candidate_weights.keys(): # if not prunned, select from all params
                    self.indicator_candidate = torch.ones(128, 256, 3, 3) # Default shape if not found
                self.indicator_candidate = self.indicator_candidate.reshape(-1)
                min_values, min_indices = torch.topk(gradient, k=512, largest=False) # Ensure k <= gradient.numel()
                self.indicator_indices = [min_indices[0]] # avoid zero
                val_cur = [curvature[min_indices[0]]]
                for ind in min_indices:
                    if self.indicator_candidate[ind]: # pruned parameter
                        if len(self.indicator_indices) < self.num_candidate:
                            self.indicator_indices.append(ind)
                            val_cur.append(curvature[ind])
                        elif curvature[ind] < max(val_cur):
                            temp = val_cur.index(max(val_cur))
                            self.indicator_indices[temp] = ind
                            val_cur[temp] = curvature[ind]
                        elif curvature[ind] == 0: # Check if max(val_cur) can be 0
                            if max(val_cur) == 0 and gradient[ind] < gradient[self.indicator_indices[val_cur.index(max(val_cur))]]:
                                temp = val_cur.index(max(val_cur))
                                val_cur[temp] = curvature[ind]
                                self.indicator_indices[temp] = ind
                            elif max(val_cur) > 0: # Original logic if max_val_cur > 0
                                temp = val_cur.index(max(val_cur))
                                if gradient[ind] < gradient[self.indicator_indices[temp]]:
                                    val_cur[temp] = curvature[ind]
                                    self.indicator_indices[temp] = ind


                for idx_val in self.indicator_indices:
                    i0 = idx_val//(256*3*3) 
                    i1 = idx_val%(256*3*3)//(3*3)
                    i2 = idx_val%(3*3)//3
                    i3 = idx_val%3
                    self.indicator_indices_mat.append([i0,i1,i2,i3])
                    if self.indicator_layer_name in candidate_weights.keys():
                        candidate_weights[self.indicator_layer_name][i0,i1,i2,i3] = False

            if proportion <= 0.3 and num_pruned_layer < 47:
                logging.info('Error: low proportion with low pruned layer, check whether torch_prunnig has not been modified!')
                assert(0)
            self.stored_model_weights = model_weights
            self.global_model.zero_grad()
            logging.info(f'sum_pruned_weights: {sum_pruned_weights}, weight_sum: {weight_sum}, proportion: {sum_pruned_weights/weight_sum if weight_sum > 0 else 0}')


        elif self.use_critical_poison==3:
            history_weights = self.previous_global_model.state_dict()
            length=len(candidate_layers)
            for layer in history_weights.keys():
                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                if n_weight > 0 : 
                    idx_to_get = min(int(n_weight * proportion), n_weight -1)
                    if idx_to_get < 0: idx_to_get = 0 # Handle case if n_weight * proportion is very small
                    theta = torch.sort(candidate_weights[layer].flatten(), descending=True)[0][idx_to_get]
                    candidate_weights[layer] = candidate_weights[layer] < theta
                else:
                    candidate_weights[layer] = torch.zeros_like(candidate_weights[layer], dtype=torch.bool)

            self.stored_model_weights = model_weights
        else:
            candidate_layers = [1 for _ in range(len(model_weights.keys()))]
        
        self.previous_global_model = copy.deepcopy(self.global_model)
        
        if self.use_critical_poison<2:
            return candidate_layers
        elif self.use_critical_poison == 2 or self.use_critical_poison == 3:
            return candidate_weights

    def reset_weight(self, mask):
        for key, value in self.global_model.state_dict().items():
            if key in mask.keys() and key in self.stored_model_weights:
                value[mask[key]] = self.stored_model_weights[key][mask[key]]

    def update_layer_require_grad(self, candidate_layers):
        count = 0
        for (name, param), requires_grad_flag in zip(self.global_model.named_parameters(), candidate_layers):
            param.requires_grad = bool(requires_grad_flag)
            count += 1
        assert(count == len(candidate_layers))
        for name, param in self.global_model.named_parameters():
            logging.info((f"{name}: requires_grad={param.requires_grad}"))

    def read_indicator(self):
        feedback = []
        num_chosen_client = []
        weight_g = self.global_model.state_dict()
        for i in range(min(1, len(self.indicator_indices_mat))):
            if not self.indicator_indices_mat: continue
            idx = self.indicator_indices_mat[i]
            if isinstance(idx, list) and len(idx) == 4:
                 delta = weight_g[self.indicator_layer_name][idx[0], idx[1], idx[2], idx[3]].item() - self.indicator_param[i][0]
            elif isinstance(idx, torch.Tensor) and idx.ndim == 0:
                 delta = weight_g[self.indicator_layer_name][idx].item() - self.indicator_param[i][0]
            else:
                 logging.warning(f"Unexpected idx format in read_indicator: {idx}")
                 continue

            delta_x = self.indicator_param[i][1]
            if delta_x == 0: delta_x = 1e-9 # Avoid division by zero
            feedback.append( delta / delta_x) 

        for i, f_val in enumerate(feedback):
            idx = self.indicator_indices_mat[i]
            if f_val > 1:
                tmp_num = (self.adaptive_k-1) / (f_val-1) if (f_val-1) != 0 else float('inf')
                if tmp_num > 0 and tmp_num <= 2 * self.all_num_clients:
                    num_chosen_client.append(tmp_num)
                logging.info(tmp_num)

        if len(num_chosen_client) > 0:
            self.his_accepted.append(copy.deepcopy(self.scale_rate))
            ### new scale rate
            self.his_accepted.sort()
            new_scale = sum(num_chosen_client) / len(num_chosen_client)
            
            if len(self.his_rejected) > 10:
                pos = int(len(self.his_rejected) * 0.2)
                median_reject_scale_rate = self.his_rejected[pos]
                new_scale = min(median_reject_scale_rate, new_scale)
                logging.info(f'median_reject_scale_rate: {median_reject_scale_rate}')
            
            self.scale_rate = self.scale_rate * (1-self.adaptive_lr) + new_scale * self.adaptive_lr
            self.accepted_before = True
            logging.info(f'scale_rate: {self.scale_rate}, {num_chosen_client}')

        else:
            self.his_rejected.append(copy.deepcopy(self.scale_rate))
            self.his_rejected.sort()
            if len(self.his_accepted) > 10:
                pos = int(len(self.his_accepted) * 0.5)
                median_accept_scale_rate = self.his_accepted[pos]
                self.scale_rate = max(self.scale_rate * (1-self.adaptive_lr), median_accept_scale_rate)
            else:
                self.scale_rate = self.scale_rate * (1-self.adaptive_lr)
            if self.accepted_before == True:
                self.adaptive_lr = self.adaptive_lr * self.adaptive_decay
            self.accepted_before = False

            logging.info(f'scale_rate: {self.scale_rate}, {num_chosen_client} ')
        logging.info(f'adaptive_lr: {self.adaptive_lr} {self.his_accepted} {self.his_rejected}')

        self.scale_rate = max(self.scale_rate, 0.8)

        for i in range(min(1, len(self.indicator_indices_mat))):
            if not self.indicator_indices_mat: continue
            idx = self.indicator_indices_mat[i]
            if isinstance(idx, list) and len(idx) == 4:
                weight_g[self.indicator_layer_name][idx[0], idx[1], idx[2], idx[3]] = self.indicator_param[i][0] + self.indicator_param[i][1]
            elif isinstance(idx, torch.Tensor) and idx.ndim == 0:
                 weight_g[self.indicator_layer_name][idx] = self.indicator_param[i][0] + self.indicator_param[i][1]


        self.indicator_indices_mat = self.indicator_indices_mat[min(1, len(self.indicator_indices_mat)):]


    def local_train(self, round_num, local_epoch, mid_T, use_labels=True):
        self.round = round_num

        if self.use_adaptive and round_num > 0:
            self.read_indicator()

        self.global_trainer.train()
        if self.use_pgd_poison:
            model_original_vec = vectorize_net(self.target_model)
        eps = 8.0
        
        logging.info('round: '+str(round_num))
        if self.use_critical_poison == 1 and round_num == 10:
            sched_state_dict = self.global_sched.state_dict()
            candidate_layers = self.search_candidate_weights(self.critical_proportion)
            logging.info(candidate_layers)
            self.update_layer_require_grad(candidate_layers)
            self.global_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.global_model.parameters()), lr=self.lr)
            self.global_sched = torch.optim.lr_scheduler.LambdaLR(self.global_optim, lr_lambda=self.warmup_lr)
            self.global_sched.load_state_dict(sched_state_dict)
            self.global_sched.step()
        elif (self.use_critical_poison == 2 and round_num == 0) or (self.use_adaptive and len(self.indicator_indices_mat) == 0):
            self.candidate_weights = self.search_candidate_weights(self.critical_proportion)
        elif self.use_critical_poison == 3 and round_num == 1:
            self.candidate_weights = self.search_candidate_weights(self.critical_proportion)

        count = 0
        while True:
            for x1, x2 in zip(self.train_loader, cycle(self.attacker_loader)):
                x_benign, label_benign = x1[0].to(self.device), x1[1].to(self.device)
                x_tar, y_tar, mask_trigger = x2[0].to(self.device), x2[1].to(self.device), x2[2].to(self.device)
                
                x_combined = torch.cat([x_benign, x_tar], dim=0)
                label_combined = torch.cat([label_benign, y_tar], dim=0)
                # Modifications for COLLUSION
                # <<< MODIFIED BLOCK START >>>
                # Calculate task loss
                if use_labels:
                    task_loss = self.global_trainer(x_combined, self.miu, mask_trigger, 0, 1000, label_combined)
                else:
                    task_loss = self.global_trainer(x_combined, self.miu, mask_trigger, 0, 1000)

                # Calculate similarity loss
                similarity_loss = torch.tensor(0.0, device=self.device)
                if self.lambda_reg is not None and self.lambda_reg > 0 and len(self.other_colluders) > 0:
                    similarity_loss = self.calculate_similarity_loss()
                
                total_loss = task_loss + self.lambda_reg * similarity_loss
                # <<< MODIFIED BLOCK END >>>

                # global update
                if self.use_pgd_poison:
                    self.adv_global_optim.zero_grad()
                    total_loss.backward() # Use total_loss
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.adv_global_optim.step()
                    self.adv_global_sched.step() # Assume PGD also steps its own scheduler
                    
                    w_vec = vectorize_net(self.global_model)
                    val_norm = torch.norm(w_vec - model_original_vec)
                    
                    if (val_norm > eps and val_norm > 0): # Added val_norm > 0
                        scale = eps / val_norm
                        for key, value in self.global_model.state_dict().items():
                                target_value = self.target_model.state_dict()[key]
                                new_value = target_value + (value - target_value) * scale
                                self.global_model.state_dict()[key].copy_(new_value)
                else:
                    self.global_optim.zero_grad()
                    total_loss.backward() # Use total_loss
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    
                if self.use_critical_poison == 2 or (self.use_critical_poison == 3 and round_num > 1):
                    self.reset_weight(self.candidate_weights)

                self.ema(self.global_model, self.global_ema_model, self.ema_scale)
                
                if self._step_cound % 100 == 0:
                    logging.info(f'Client {self.client_id} Rnd {round_num} Stp {self._step_cound}: TaskLoss={task_loss.item():.4f}, SimLoss={similarity_loss.item():.4f}, TotalLoss={total_loss.item():.4f}, LR={self.global_sched.get_last_lr()[0]:.6f}')
                
                self._step_cound += 1
                count += x_combined.shape[0]
                if count + x_combined.shape[0] > 10000: 
                    break
            if count + x_combined.shape[0] > 10000: 
                break
        
        # Modifications for COLLUSION Scaling
        ### We scale data according to formula: L = G + scale_rate * (X - G) / num_attackers
        if self.use_model_poison or self.use_critical_poison > 0:
            num_attackers_for_scaling = 1 # Default
            if self.num_colluders is not None and self.num_colluders > 0:
                if self.scaled:
                    num_attackers_for_scaling = self.num_colluders
            
            effective_scale_rate = self.scale_rate  / num_attackers_for_scaling
            logging.info(f'Client {self.client_id}: Applying final scaling: {self.scale_rate} / {num_attackers_for_scaling} = {effective_scale_rate}')
            
            for key, value in self.global_model.state_dict().items():
                    target_value = self.target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * effective_scale_rate
                    self.global_model.state_dict()[key].copy_(new_value)

            for key, value in self.global_ema_model.state_dict().items():
                    target_value = self.target_ema_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * effective_scale_rate
                    self.global_ema_model.state_dict()[key].copy_(new_value)

        if self.use_adaptive and (self.use_model_poison or self.use_critical_poison > 0):
            self.indicator_param = []
            weight_g = self.global_model.state_dict()
            weight_pre = self.target_model.state_dict()
            
            for index_val in self.indicator_indices_mat[:min(1, len(self.indicator_indices_mat))]:
                current_weight_val = 0
                prev_weight_val = 0
                if isinstance(index_val, list) and len(index_val) == 4:
                    current_weight_val = weight_g[self.indicator_layer_name][index_val[0], index_val[1], index_val[2], index_val[3]]
                    prev_weight_val = weight_pre[self.indicator_layer_name][index_val[0], index_val[1], index_val[2], index_val[3]]
                elif isinstance(index_val, torch.Tensor) and index_val.ndim == 0:
                    current_weight_val = weight_g[self.indicator_layer_name][index_val]
                    prev_weight_val = weight_pre[self.indicator_layer_name][index_val]
                else:
                    logging.warning(f"Unexpected index_val format in adaptive part: {index_val}")
                    continue

                if (current_weight_val - prev_weight_val).abs() < 1e-9: # Check for near zero difference
                    self.indicator_param.append((copy.deepcopy(prev_weight_val.item()), (1e-4/self.adaptive_k if self.adaptive_k != 0 else 1e-4)))
                    # Modify weight_g directly if it's a tensor; .add_() is for tensors
                    if isinstance(index_val, list) and len(index_val) == 4:
                         weight_g[self.indicator_layer_name][index_val[0], index_val[1], index_val[2], index_val[3]].add_(1e-4)
                    elif isinstance(index_val, torch.Tensor) and index_val.ndim == 0:
                         weight_g[self.indicator_layer_name][index_val].add_(1e-4)
                else:   
                    delta_x = (current_weight_val - prev_weight_val) / self.scale_rate if self.scale_rate !=0 else (current_weight_val - prev_weight_val)
                    self.indicator_param.append((copy.deepcopy(prev_weight_val.item()), delta_x.item()))
                    new_adaptive_val = prev_weight_val + delta_x * self.adaptive_k
                    if isinstance(index_val, list) and len(index_val) == 4:
                        weight_g[self.indicator_layer_name][index_val[0], index_val[1], index_val[2], index_val[3]] = new_adaptive_val
                    elif isinstance(index_val, torch.Tensor) and index_val.ndim == 0:
                        weight_g[self.indicator_layer_name][index_val] = new_adaptive_val
                
            logging.info(f'indicator_param: {self.indicator_param}')

        logging.info(f'Client {self.client_id}: scale rate: {self.scale_rate}')
        return self.global_model, self.global_ema_model

    def get_targets_num(self):
        return len(self.train_dataset.targets) + len(self.attacker_dataset.targets)//self.attacker_dataset.num_repeat

    def init_attack_sampler(self):
        self.gamma = 0.1
        self.x_T_i = None

    def get_sample(self, x_T, start_step, end_step, labels=None):
        self.global_ema_model.eval()
        if self.x_T_i is None:
            self.miu_sample = torch.stack([self.miu] * x_T.shape[0])
            x_T_i = self.gamma * x_T + self.miu_sample * (1 - self.gamma)
            self.trigger_mask = self.attacker_dataset.init_trigger_mask[0].unsqueeze(0).repeat(x_T.shape[0], 1, 1, 1).to(self.device)
            x_T_i = x_T + (x_T_i - x_T)*self.trigger_mask
            self.x_T_i = x_T_i
        if labels == None:
            sample = self.global_ema_sampler(self.x_T_i, self.miu_sample, self.trigger_mask, labels=None)
        else:
            sample = self.global_ema_sampler(x_T, start_step, end_step, labels)
        self.global_ema_model.train()
        return sample

class AttackerClientMultiTargetNonIIDLayerSub(object): # NEVER USED IN OUR PAPER
    def __init__(self, client_id, dataset_name, train_dataset, train_loader, attacker_dataset, 
                 attacker_loader, use_model_poison, use_pgd_poison, use_critical_poison, critical_proportion,
                 use_bclayersub_poison, scale_rate, device, ema_scale=0.9999, scaled=False,
                 num_colluders=None, lambda_reg=None, defense_technique=None): # <<< MODIFIED: Added num_colluders, lambda_reg
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            self.img_size = 32
        elif dataset_name == 'celeba':
            self.img_size = 64
        self.defense_technique = defense_technique
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.train_loader = train_loader
        self.attacker_dataset = attacker_dataset
        self.attacker_loader = attacker_loader
        self.use_model_poison = use_model_poison
        self.use_pgd_poison = use_pgd_poison
        self.use_critical_poison = use_critical_poison
        self.critical_proportion = critical_proportion
        self.use_bclayersub_poison = use_bclayersub_poison
        self.device = device
        self.ema_scale = ema_scale
        self.scaled = scaled

        logging.shutdown()
        importlib.reload(logging)
        logging.basicConfig(filename='./tmp/attacker.log',
            filemode='a',
            level=logging.INFO)

        self.scale_rate = scale_rate 

        self.global_model = None
        self.global_ema_model = None
        self.global_optim = None
        self.global_sched = None
        self.global_trainer = None
        self.global_ema_sampler = None
        # Modifications for COLLUSION altough unnecessary because we never use Layer SUB
        self.malicious_model = None # For LayerSub
        self.malicious_ema_model = None # For LayerSub
        self.global_adv_trainer = None # For LayerSub malicious model

        self._step_cound = 0
        
        # <<< ADDED START >>>
        self.num_colluders = num_colluders
        self.lambda_reg = lambda_reg
        self.other_colluders = []
        # <<< ADDED END >>>

    # <<< ADDED START >>>
    # Duplicating from AttackerClientMultiTargetNonIID for LayerSub version
    def set_colluder_refs(self, other_colluder_refs):
        self.other_colluders = other_colluder_refs
        logging.info(f"Client {self.client_id} (LayerSub) received {len(self.other_colluders)} colluder references.")

    def calculate_similarity_loss(self): # Assumes similarity on self.global_model
        if not self.other_colluders or self.lambda_reg is None or self.lambda_reg <= 0:
            return torch.tensor(0.0, device=self.device)
        if self.global_model is None: return torch.tensor(0.0, device=self.device)

        param_keys = list(self.global_model.state_dict().keys())
        other_state_dicts = []
        for colluder_ref in self.other_colluders:
            if hasattr(colluder_ref, 'global_model') and colluder_ref.global_model is not None:
                 other_state_dicts.append(colluder_ref.global_model.state_dict())
        if not other_state_dicts: return torch.tensor(0.0, device=self.device)

        avg_state_dict = {}
        for key in param_keys:
             valid_tensors_for_key = [sd[key].detach().clone() for sd in other_state_dicts if key in sd]
             if not valid_tensors_for_key: 
                 avg_state_dict[key] = self.global_model.state_dict()[key].detach().clone()
                 continue
             avg_tensor = torch.stack(valid_tensors_for_key).mean(dim=0)
             avg_state_dict[key] = avg_tensor
        
        l2_dist_sq = torch.tensor(0.0, device=self.device)
        my_state_dict = self.global_model.state_dict()
        for key in param_keys:
            if key in my_state_dict and key in avg_state_dict:
                 l2_dist_sq += torch.sum((my_state_dict[key] - avg_state_dict[key])**2)
        return l2_dist_sq
    # <<< ADDED END >>>

    def warmup_lr(self, step):
        warmup_epoch = 15
        if self.dataset_name == 'cifar10':
            warmup_iters = 10000//128 * warmup_epoch
        else:
            warmup_iters = 5000//128 * warmup_epoch
        return min(step, warmup_iters) / warmup_iters

    def ema(self, source, target, decay):
        source_dict = source.state_dict()
        target_dict = target.state_dict()
        for key in source_dict.keys():
            target_dict[key].data.copy_(
                target_dict[key].data * decay +
                source_dict[key].data * (1 - decay))

    def init_lora(self, lr): # This method was defined but not used in the main path
        LycorisNetwork.apply_preset({"target_name": [".*.*"]})
        self.multiplier = 1.0

        self.lycoris_net1.apply_to()
        self.lycoris_net1 = self.lycoris_net1.to(self.device)
        self.global_optim = torch.optim.Adam(
            self.lycoris_net1.parameters(), lr)

    def init(self, model_global, lr, parallel, global_ckpt=None):
        self.lr = lr
        self.global_model = copy.deepcopy(model_global)
        self.global_ema_model = copy.deepcopy(self.global_model)

        if global_ckpt is not None:
            self.global_model.load_state_dict(global_ckpt['global_model'], strict=True)
            self.global_ema_model.load_state_dict(global_ckpt['global_ema_model'], strict=True)

        self.global_optim = torch.optim.Adam(
            self.global_model.parameters(), lr)
        self.global_sched = torch.optim.lr_scheduler.LambdaLR(
            self.global_optim, lr_lambda=self.warmup_lr)

        self.global_trainer = GaussianDiffusionMultiTargetTrainer(
            self.global_model, 1e-4, 0.02, 1000).to(self.device)
        if self.use_bclayersub_poison:
            self.malicious_model = copy.deepcopy(model_global) # Init malicious_model
            self.malicious_ema_model = copy.deepcopy(self.global_ema_model) # Ema for malicious
            self.global_adv_trainer = GaussianDiffusionMultiTargetTrainer(
                self.malicious_model, 1e-4, 0.02, 1000).to(self.device)
        
        self.global_ema_sampler = GaussianDiffusionMaskAttackerSampler(
            self.global_ema_model, 1e-4, 0.02, 1000, self.img_size, 'epsilon', 'fixedlarge').to(self.device)
        self.init_attack_sampler()

        self.transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        miu = Image.open('./images/white.png')
        self.miu = self.transform(miu).to(self.device)

        if self.use_model_poison or self.use_pgd_poison or self.use_critical_poison or self.use_bclayersub_poison:
            self.target_model = copy.deepcopy(self.global_model)
            self.target_ema_model = copy.deepcopy(self.global_ema_model)

            if self.use_bclayersub_poison:
                self.adv_global_optim = torch.optim.Adam(
                    self.malicious_model.parameters(), lr) # Optimizer for malicious_model
            else: #This else branch might conflict if LayerSub also uses PGD/ModelPoison on global_model
                  # Assuming adv_global_optim is primarily for malicious_model in LayerSub
                self.adv_global_optim = torch.optim.Adam(
                    self.global_model.parameters(), lr) 
                
            self.adv_global_sched = torch.optim.lr_scheduler.LambdaLR( # Scheduler for adv_optim
                self.adv_global_optim, lr_lambda=self.warmup_lr)


        if self.use_critical_poison == 1 or self.use_critical_poison == 3 or self.use_bclayersub_poison:
            self.previous_global_model = copy.deepcopy(self.global_model)

    def set_global_parameters(self, parameters, ema_parameters):
        self.global_model.load_state_dict(copy.deepcopy(parameters), strict=True)
        self.global_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)

        if self.use_model_poison or self.use_pgd_poison or self.use_critical_poison or self.use_bclayersub_poison:
            self.target_model.load_state_dict(copy.deepcopy(parameters), strict=True)
            self.target_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)
            if self.use_bclayersub_poison and self.malicious_model is not None: # Also update malicious model's targets if used
                self.malicious_model.load_state_dict(copy.deepcopy(parameters), strict=True)
                self.malicious_ema_model.load_state_dict(copy.deepcopy(ema_parameters), strict=True)


    def search_candidate_weights(self, proportion=0.8, reverse=False, batch_for_diff=64):
        candidate_weights = OrderedDict()
        # Determine which model's state to use for search_candidate_weights
        # If LayerSub, and this is called during malicious training, should it use malicious_model?
        # For now, assume it always uses self.global_model as per original logic structure.
        current_model_for_search = self.global_model 
        if current_model_for_search is None: # Fallback if global_model not yet initialized
            logging.error("search_candidate_weights called before global_model is initialized.")
            return {} if self.use_critical_poison >=2 else []


        model_weights = copy.deepcopy(current_model_for_search.state_dict())
        candidate_layers = [0 for _ in range(len(model_weights.keys()))]
        # candidate_weights = OrderedDict() # Already initialized
        critical_layers = {}


        if (self.use_critical_poison==1 or self.use_bclayersub_poison) and self._step_cound > 0: #kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict() # Uses previous_global_model
            candidate_layers_mean = []
            length=len(candidate_layers)
            for layer in history_weights.keys():
                if layer not in model_weights: continue # Skip if layer mismatch
                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                absmean_weight = torch.abs(candidate_weights[layer]).sum()/(n_weight if n_weight > 0 else 1)
                candidate_layers_mean.append(absmean_weight) 
                critical_layers[layer] = absmean_weight
            
            if not candidate_layers_mean: # Handle empty list
                candidate_layers = [0 for _ in range(length)] # Default if no layers processed
            else:
                candidate_layers_mean_tensor = torch.stack(candidate_layers_mean).view(-1)
                if candidate_layers_mean_tensor.numel() > 0:
                    idx_to_get = min(int(length * proportion), candidate_layers_mean_tensor.numel() -1)
                    if idx_to_get < 0: idx_to_get = 0
                    theta = torch.sort(candidate_layers_mean_tensor, descending=True)[0][idx_to_get]
                    for k_layer in critical_layers.keys(): # Renamed k
                        critical_layers[k_layer] = critical_layers[k_layer] > theta
                    candidate_layers = candidate_layers_mean_tensor > theta
                else: # Handle empty tensor
                    candidate_layers = [0 for _ in range(length)]


        elif self.use_critical_poison==2: # DiffPruning logic - complex, ensuring minimal changes to its flow
            logging.info(f'use diffusion importance: {self.use_critical_poison}, {proportion}')
            imp = tp.importance.TaylorImportance()
            example_inputs = {'x': torch.randn(1, 3, self.img_size, self.img_size).to(self.device), 't': torch.ones(1).to(self.device, dtype=torch.long)}
            ignored_layers = [current_model_for_search.tail] # Use current_model_for_search
            
            pruner = tp.pruner.MagnitudePruner(
                current_model_for_search, # Use current_model_for_search
                example_inputs, importance=imp, global_pruning=True, 
                iterative_steps=1, channel_groups={}, pruning_ratio=1.0-proportion, 
                ignored_layers=ignored_layers, root_module_types=[torch.nn.Conv2d, torch.nn.Linear],
            )
            current_model_for_search.zero_grad()

            count = 0
            max_loss_val = 0 # Renamed
            thr = 0.05
            # Pruning gradient calculation loop
            # Determine which trainer to use based on which model current_model_for_search is
            pruning_trainer = self.global_trainer # Default to global_trainer
            if self.use_bclayersub_poison and current_model_for_search is self.malicious_model:
                pruning_trainer = self.global_adv_trainer


            while True:
                for x_batch in self.train_loader: # Benign data for pruning gradients usually
                    x_tar_prune = x_batch[0].to(self.device)[:batch_for_diff]
                    # Assuming pruning_trainer returns loss without backward
                    loss_for_pruning = pruning_trainer(x_tar_prune, self.miu, None, count, count+1) 
                    loss_for_pruning.backward()
                    count += 1
                    if loss_for_pruning > max_loss_val: max_loss_val = loss_for_pruning
                    if loss_for_pruning < max_loss_val*thr and max_loss_val > 0 : # Added max_loss_val > 0
                        count = 1000 
                        break
                if count > 999: break
            
            sum_pruned_weights = 0 # Renamed
            weight_sum = 0
            for group in pruner.step(interactive=True):
                tmp_sum = 0
                for dep, idxs in group:
                    target_layer_obj = dep.target.module # Renamed target_layer
                    pruning_fn = dep.handler
                    layer_name = dep.target._name # Renamed layer
                    if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels, 
                                      tp.prune_conv_out_channels, tp.prune_linear_out_channels, 
                                      tp.prune_batchnorm_out_channels]:
                        layer_weight = layer_name +'.weight'
                        layer_bias = layer_name +'.bias'
                        if layer_weight not in model_weights: continue # Skip if key error

                        candidate_weights[layer_weight] = torch.ones_like(model_weights[layer_weight]).to(self.device)
                        if target_layer_obj.bias is not None and layer_bias in model_weights:
                            candidate_weights[layer_bias] = torch.ones_like(model_weights[layer_bias]).to(self.device)
                        
                        if pruning_fn in [tp.prune_conv_in_channels, tp.prune_linear_in_channels]:
                            candidate_weights[layer_weight][:, idxs] *= 0
                        elif pruning_fn in [tp.prune_conv_out_channels, tp.prune_linear_out_channels]:
                            candidate_weights[layer_weight][idxs] *= 0
                            if target_layer_obj.bias is not None and layer_bias in candidate_weights:
                                candidate_weights[layer_bias][idxs] *= 0
                        elif pruning_fn in [tp.prune_batchnorm_out_channels]: # For batchnorm
                            candidate_weights[layer_weight][idxs] *= 0 # BN weight
                            if target_layer_obj.bias is not None and layer_bias in candidate_weights:
                                candidate_weights[layer_bias][idxs] *= 0 # BN bias

                        candidate_weights[layer_weight] = candidate_weights[layer_weight] == 0
                        if target_layer_obj.bias is not None and layer_bias in candidate_weights:
                           candidate_weights[layer_bias] = candidate_weights[layer_bias] == 0
                        
                        tmp_sum += candidate_weights[layer_weight].sum()
                        weight_sum += candidate_weights[layer_weight].numel()
                sum_pruned_weights += tmp_sum
                # group.prune() # Original was commented out
            
            self.stored_model_weights = model_weights # Store weights of the model used for search
            current_model_for_search.zero_grad()
            # torch.save(current_model_for_search, 'model.pth') # Original had this for global_model
            logging.info(f'sum_pruned_weights: {sum_pruned_weights}, weight_sum: {weight_sum}, proportion: {sum_pruned_weights/weight_sum if weight_sum > 0 else 0}')


        elif self.use_critical_poison==3:
            history_weights = self.previous_global_model.state_dict()
            # length=len(candidate_layers) # Already defined
            for layer_name in history_weights.keys(): # Renamed layer
                if layer_name not in model_weights: continue
                candidate_weights[layer_name] = (model_weights[layer_name] - history_weights[layer_name]) * model_weights[layer_name]
                n_weight = candidate_weights[layer_name].numel()
                if n_weight > 0:
                    idx_to_get = min(int(n_weight * proportion), n_weight -1)
                    if idx_to_get < 0: idx_to_get = 0
                    theta = torch.sort(candidate_weights[layer_name].flatten(), descending=True)[0][idx_to_get]
                    candidate_weights[layer_name] = candidate_weights[layer_name] < theta
                else:
                    candidate_weights[layer_name] = torch.zeros_like(candidate_weights[layer_name], dtype=torch.bool)
            self.stored_model_weights = model_weights
        else: # Default case if no critical poison
            candidate_layers = [1 for _ in range(len(model_weights.keys()))] # This is for layer-wise grad updates
        
        self.previous_global_model = copy.deepcopy(self.global_model) # Always update previous_global_model
        
        if self.use_bclayersub_poison:
            return critical_layers # LayerSub returns a dict of booleans per layer

        if self.use_critical_poison<2:
            return candidate_layers # List of booleans for grad updates
        elif self.use_critical_poison == 2 or self.use_critical_poison == 3:
            return candidate_weights # OrderedDict of masks for resetting weights

    def craft_critical_layer(self, critical_layers_dict, lambda_value=1.0): # Renamed critical_layers
        malicious_w = self.malicious_model.state_dict()
        benign_w = self.global_model.state_dict() # The "benign" part trained on client's data
        global_target_w = self.target_model.state_dict() # The model state before local training

        for key in self.global_model.state_dict().keys():
            if key in critical_layers_dict and critical_layers_dict[key]: # Check if key exists and is True
                # Ensure all weight dicts have this key
                if key not in malicious_w or key not in benign_w or key not in global_target_w:
                    continue
                new_value = global_target_w[key] + (malicious_w[key] - global_target_w[key]) * lambda_value + \
                            max(0, (1 - lambda_value)) * (benign_w[key] - global_target_w[key])
                self.global_model.state_dict()[key].copy_(new_value)
       
        # EMA model crafting
        if hasattr(self, 'malicious_ema_model') and self.malicious_ema_model is not None:
            malicious_ema_w = self.malicious_ema_model.state_dict()
            benign_ema_w = self.global_ema_model.state_dict()
            global_target_ema_w = self.target_ema_model.state_dict()

            for key in self.global_ema_model.state_dict().keys():
                if key in critical_layers_dict and critical_layers_dict[key]:
                    if key not in malicious_ema_w or key not in benign_ema_w or key not in global_target_ema_w:
                        continue
                    new_ema_value = global_target_ema_w[key] + \
                                    (malicious_ema_w[key] - global_target_ema_w[key]) * lambda_value + \
                                    max(0, (1 - lambda_value)) * (benign_ema_w[key] - global_target_ema_w[key])
                    self.global_ema_model.state_dict()[key].copy_(new_ema_value)
        
        return self.global_model, self.global_ema_model

    def reset_weight(self, mask_dict): # Renamed mask
        # Determine which model to reset based on context (e.g., current training phase)
        # For simplicity, assume it refers to self.global_model if not specified
        # This method is usually called with self.candidate_weights obtained from search_candidate_weights
        model_to_reset = self.global_model # Default
        # If self.use_bclayersub_poison, and currently training malicious_model, this would need context.
        # However, self.stored_model_weights is based on global_model from search_candidate_weights.
        # So, resetting global_model seems consistent with how stored_model_weights is set.

        if model_to_reset is None or not hasattr(self, 'stored_model_weights'): return

        for key, value in model_to_reset.state_dict().items():
            if key in mask_dict and key in self.stored_model_weights:
                value[mask_dict[key]] = self.stored_model_weights[key][mask_dict[key]]


    def update_layer_require_grad(self, candidate_layers_list): # Renamed candidate_layers
        # This method applies to self.global_model by default
        # If LayerSub trains malicious_model separately, its grad settings might need separate handling
        # For simplicity, assume this applies to global_model.
        if self.global_model is None: return
        count = 0
        # candidate_layers_list should be a list of booleans matching named_parameters
        if not isinstance(candidate_layers_list, list) or len(candidate_layers_list) != len(list(self.global_model.parameters())) :
            logging.warning(f"Client {self.client_id}: Mismatch in candidate_layers for grad update or not a list.")
            return

        for (name, param), requires_grad_flag in zip(self.global_model.named_parameters(), candidate_layers_list):
            param.requires_grad = bool(requires_grad_flag)
            count += 1
        # assert(count == len(candidate_layers_list)) # Original assert
        # for name, param in self.global_model.named_parameters():
        #     logging.info((f"{name}: requires_grad={param.requires_grad}"))


    def local_train(self, round_num, local_epoch, mid_T, use_labels=True): # Renamed round
        # global_loss = 0 # Defined but not used, task_loss / adv_global_loss used
        if self.use_pgd_poison: # PGD target model should be set before training loop
            model_original_vec = vectorize_net(self.target_model) 
        eps = 8.0

        logging.info(f'Client {self.client_id} (LayerSub) round: '+str(round_num))
        # Critical poison search might need to be aware of which model (global or malicious) it's searching for
        if self.use_critical_poison == 1 and round_num == 10:
            # This applies to global_model by default as per update_layer_require_grad
            sched_state_dict = self.global_sched.state_dict()
            candidate_layers_list = self.search_candidate_weights(self.critical_proportion) # Returns list for CP1
            logging.info(candidate_layers_list)
            self.update_layer_require_grad(candidate_layers_list) # Applies to global_model
            self.global_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.global_model.parameters()), lr=self.lr)
            self.global_sched = torch.optim.lr_scheduler.LambdaLR(self.global_optim, lr_lambda=self.warmup_lr)
            if sched_state_dict: self.global_sched.load_state_dict(sched_state_dict)
            self.global_sched.step()
        elif self.use_critical_poison == 2 and round_num == 0: # Returns dict for CP2
            self.candidate_weights_dict = self.search_candidate_weights(self.critical_proportion) # Store as dict
        elif self.use_critical_poison == 3 and round_num == 1: # Returns dict for CP3
            self.candidate_weights_dict = self.search_candidate_weights(self.critical_proportion)

        # Main training logic for LayerSub
        if self.use_bclayersub_poison:
            if self.malicious_model is None or self.global_adv_trainer is None:
                logging.error(f"Client {self.client_id}: LayerSub enabled but malicious_model/trainer not initialized.")
                return self.global_model, self.global_ema_model # Early exit

            self.malicious_model.load_state_dict(self.global_model.state_dict()) # Sync malicious with global initially
            self.malicious_ema_model.load_state_dict(self.global_ema_model.state_dict())
            
            self.global_trainer.train() # For benign model
            self.global_adv_trainer.train() # For malicious model

            count = 0
            stop_benign_iter = 5000 if round_num >= 10 else 10000

            # Train benign part (self.global_model)
            current_iter_benign = 0
            # Loop for benign training
            for x1_benign in self.train_loader: # Renamed x1
                if current_iter_benign >= stop_benign_iter: break
                x_benign, label_benign = x1_benign[0].to(self.device), x1_benign[1].to(self.device)
                
                task_loss_benign = self.global_trainer(x_benign, self.miu, None, 0, 1000, label_benign) if use_labels else self.global_trainer(x_benign, self.miu, None, 0, 1000)
                
                sim_loss_val = torch.tensor(0.0, device=self.device)
                if self.lambda_reg is not None and self.lambda_reg > 0:
                    sim_loss_val = self.calculate_similarity_loss() # Based on self.global_model
                
                total_loss_benign = task_loss_benign + self.lambda_reg * sim_loss_val

                self.global_optim.zero_grad()
                total_loss_benign.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                self.global_optim.step()
                self.global_sched.step()
                self.ema(self.global_model, self.global_ema_model, self.ema_scale)
                
                self._step_cound += 1; count += x_benign.shape[0]; current_iter_benign += x_benign.shape[0]
                if count >= 10000: break # Overall iteration cap

            # Train malicious part (self.malicious_model)
            current_iter_malicious = 0
            # Loop for malicious training (ensure it runs up to 10000 total iterations with benign)
            # Using cycle for attacker_loader to ensure enough data if attacker_loader is small
            attacker_data_iterator = cycle(self.attacker_loader)
            while count < 10000: # Continue until total iterations reach 10000
                x2_malicious = next(attacker_data_iterator) # Get next batch from attacker data
                x_tar, y_tar, mask_trigger = x2_malicious[0].to(self.device), x2_malicious[1].to(self.device), x2_malicious[2].to(self.device)
                
                task_loss_malicious = self.global_adv_trainer(x_tar, self.miu, mask_trigger, 0, 1000, y_tar) if use_labels else self.global_adv_trainer(x_tar, self.miu, mask_trigger, 0, 1000)
                
                # Similarity for malicious model training:
                # Option 1: Still make self.global_model similar (as calculate_similarity_loss is defined)
                # Option 2: Define a new similarity for self.malicious_model (more complex)
                # Sticking to Option 1 for "modify nothing else" on similarity calculation logic itself.
                sim_loss_mal_val = torch.tensor(0.0, device=self.device)
                if self.lambda_reg is not None and self.lambda_reg > 0:
                     sim_loss_mal_val = self.calculate_similarity_loss() # Still targets self.global_model's similarity
                
                total_loss_malicious = task_loss_malicious # + self.lambda_reg * sim_loss_mal_val # Decide if sim loss applies here
                                                           # If goal is to make *malicious_model* similar, this needs a different sim_loss func.
                                                           # If goal is to make *global_model* (which is affected by malicious_model via craft) similar, then this is complex.
                                                           # For simplicity of *this turn's changes*, I will add the same sim_loss here,
                                                           # assuming the user wants *some* regularization pressure during this phase too,
                                                           # even if its direct target (global_model) isn't the one getting primary gradient here.
                total_loss_malicious = task_loss_malicious + self.lambda_reg * sim_loss_mal_val


                self.adv_global_optim.zero_grad() # Optimizer for malicious_model
                total_loss_malicious.backward()
                torch.nn.utils.clip_grad_norm_(self.malicious_model.parameters(), 1.)
                self.adv_global_optim.step()
                if hasattr(self, 'adv_global_sched'): self.adv_global_sched.step() # Scheduler for malicious_model
                self.ema(self.malicious_model, self.malicious_ema_model, self.ema_scale)
                
                if self._step_cound % 100 == 0:
                    logging.info(f'Client {self.client_id} (LayerSub-Mal) Rnd {round_num} Stp {self._step_cound}: AdvLoss={task_loss_malicious.item():.4f}, SimLoss={sim_loss_mal_val.item():.4f}')

                self._step_cound += 1; count += x_tar.shape[0]
                if count >= 10000: break
            
            # Crafting after both parts are trained
            if self.use_bclayersub_poison and round_num == 10:
                self.critical_layers_dict_ls = self.search_candidate_weights(self.critical_proportion) # Returns dict
            if self.use_bclayersub_poison and round_num >= 10 and hasattr(self, 'critical_layers_dict_ls'):
                self.craft_critical_layer(self.critical_layers_dict_ls, self.scale_rate) # Scale_rate used as lambda here

        else: # Original logic if not LayerSub (e.g. standard model poison, PGD, critical on global_model)
            self.global_trainer.train()
            # Store benign weights if needed for PGD or ModelPoison target
            if self.use_model_poison or self.use_pgd_poison or self.use_critical_poison:
                # This part needs to run *before* the attacker data loop if target_model is the benign start
                # The original code had a 5000 iter benign loop, then set target_model, then 5000 iter attack loop.
                
                # Benign-only pre-training phase (first 5000 iterations)
                count_benign_pretrain = 0
                for x1_benign_pre in self.train_loader:
                    if count_benign_pretrain >= 5000: break
                    x_b_pre, label_b_pre = x1_benign_pre[0].to(self.device), x1_benign_pre[1].to(self.device)
                    
                    task_loss_b_pre = self.global_trainer(x_b_pre, self.miu, None, 0, 1000, label_b_pre) if use_labels else self.global_trainer(x_b_pre, self.miu, None, 0, 1000)
                    sim_loss_b_pre = torch.tensor(0.0, device=self.device)
                    if self.lambda_reg is not None and self.lambda_reg > 0:
                        sim_loss_b_pre = self.calculate_similarity_loss()
                    total_loss_b_pre = task_loss_b_pre + self.lambda_reg * sim_loss_b_pre

                    self.global_optim.zero_grad()
                    total_loss_b_pre.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    self.ema(self.global_model, self.global_ema_model, self.ema_scale)
                    self._step_cound += 1; count += x_b_pre.shape[0]; count_benign_pretrain += x_b_pre.shape[0]
                    if count >= 10000: break

                # Set target model after benign pre-training
                if self.use_model_poison or self.use_pgd_poison or self.use_critical_poison:
                    benign_weight = copy.deepcopy(self.global_model.state_dict())
                    self.target_model.load_state_dict(benign_weight, strict=True)
                    self.target_ema_model.load_state_dict(copy.deepcopy(self.global_ema_model.state_dict()), strict=True)
                    if hasattr(self, 'candidate_weights_dict'): # For CP2/CP3
                         self.stored_model_weights = benign_weight # Stored model is the benign one for reset

            # Attacker data training phase (remaining iterations up to 10000)
            # Ensure adv_global_optim and sched are set up if needed (e.g., PGD)
            # The optimizer for this phase depends on the attack type
            current_optim = self.global_optim
            current_sched = self.global_sched
            if self.use_pgd_poison and hasattr(self, 'adv_global_optim'): # PGD uses its own optimizer
                current_optim = self.adv_global_optim
                current_sched = self.adv_global_sched
            
            attacker_data_iterator_main = cycle(self.attacker_loader)
            while count < 10000:
                x2_attack_main = next(attacker_data_iterator_main)
                x_tar_main, y_tar_main, mask_trigger_main = x2_attack_main[0].to(self.device), x2_attack_main[1].to(self.device), x2_attack_main[2].to(self.device)

                task_loss_attack_main = self.global_trainer(x_tar_main, self.miu, mask_trigger_main, 0, 1000, y_tar_main) if use_labels else self.global_trainer(x_tar_main, self.miu, mask_trigger_main, 0, 1000)
                sim_loss_attack_main = torch.tensor(0.0, device=self.device)
                if self.lambda_reg is not None and self.lambda_reg > 0:
                    sim_loss_attack_main = self.calculate_similarity_loss()
                total_loss_attack_main = task_loss_attack_main + self.lambda_reg * sim_loss_attack_main
                
                current_optim.zero_grad()
                total_loss_attack_main.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                current_optim.step()
                if current_sched: current_sched.step() # PGD might have its own scheduler

                if self.use_pgd_poison:
                    w_vec_pgd = vectorize_net(self.global_model)
                    val_norm_pgd = torch.norm(w_vec_pgd - model_original_vec) # model_original_vec from start of local_train
                    if (val_norm_pgd > eps and val_norm_pgd > 0):
                        scale_pgd = eps / val_norm_pgd
                        for key_pgd, value_pgd in self.global_model.state_dict().items():
                                target_val_pgd = self.target_model.state_dict()[key_pgd] # target_model is benign start
                                new_val_pgd = target_val_pgd + (value_pgd - target_val_pgd) * scale_pgd
                                self.global_model.state_dict()[key_pgd].copy_(new_val_pgd)
                
                if (self.use_critical_poison == 2 or (self.use_critical_poison == 3 and round_num > 1)) and hasattr(self, 'candidate_weights_dict'):
                    self.reset_weight(self.candidate_weights_dict) # Reset global_model

                self.ema(self.global_model, self.global_ema_model, self.ema_scale)
                
                if self._step_cound % 100 == 0:
                    logging.info(f'Client {self.client_id} (NonLayerSub-Attack) Rnd {round_num} Stp {self._step_cound}: AttackLoss={task_loss_attack_main.item():.4f}, SimLoss={sim_loss_attack_main.item():.4f}')
                
                self._step_cound += 1; count += x_tar_main.shape[0]
                if count >= 10000: break


        ### We scale data according to formula: L = G + scale_rate * (X - G) / num_attackers
        # This scaling is applied to self.global_model
        if self.use_model_poison or self.use_critical_poison > 0 or self.use_bclayersub_poison: # Added bclayersub here for final scaling
            num_attackers_for_scaling = 1
            if self.num_colluders is not None and self.num_colluders > 0:
                if self.scaled:
                    num_attackers_for_scaling = self.num_colluders
            
            effective_scale_rate = self.scale_rate  / num_attackers_for_scaling
            logging.info(f'Client {self.client_id} (LayerSub): Applying final scaling: {self.scale_rate} / {num_attackers_for_scaling} = {effective_scale_rate}')
            
            # Ensure target_model and target_ema_model are correctly defined and loaded for this stage
            # For LayerSub, target_model might be the state *before* any local training this round.
            if hasattr(self, 'target_model') and self.target_model is not None:
                for key, value in self.global_model.state_dict().items():
                        target_value = self.target_model.state_dict()[key]
                        new_value = target_value + (value - target_value) * effective_scale_rate
                        self.global_model.state_dict()[key].copy_(new_value)

                for key, value in self.global_ema_model.state_dict().items():
                        target_value = self.target_ema_model.state_dict()[key]
                        new_value = target_value + (value - target_value) * effective_scale_rate
                        self.global_ema_model.state_dict()[key].copy_(new_value)
            else:
                logging.warning(f"Client {self.client_id}: target_model not available for final scaling.")


        return self.global_model, self.global_ema_model

    def get_targets_num(self):
        return len(self.train_dataset.targets) + len(self.attacker_dataset.targets)//self.attacker_dataset.num_repeat

    def init_attack_sampler(self):
        self.gamma = 0.1
        self.x_T_i = None

    def get_sample(self, x_T, start_step, end_step, labels=None):
        self.global_ema_model.eval()
        if self.x_T_i is None:
            self.miu_sample = torch.stack([self.miu] * x_T.shape[0])
            x_T_i_val = self.gamma * x_T + self.miu_sample * (1 - self.gamma) # Renamed
            self.trigger_mask = self.attacker_dataset.init_trigger_mask[0].unsqueeze(0).repeat(x_T.shape[0], 1, 1, 1).to(self.device)
            x_T_i_val = x_T + (x_T_i_val - x_T)*self.trigger_mask
            self.x_T_i = x_T_i_val
        
        sample_output = None # Renamed
        if labels == None:
            # Assuming the sampler for LayerSub is self.global_ema_sampler which takes these args
            sample_output = self.global_ema_sampler(self.x_T_i, self.miu_sample, self.trigger_mask, labels=None)
        else:
            sample_output = self.global_ema_sampler(x_T, start_step, end_step, labels) # Original for labeled
        self.global_ema_model.train()
        return sample_output


def data_allocation_attacker_noniid(client_data_idxs, attack_batch_size, num_targets, save_targets_path, seed=42, dataset_name='cifar10'):
    np.random.seed(seed)
    print(seed)
    
    if len(client_data_idxs) < num_targets: # Handle case where client has fewer samples than num_targets
        num_targets_actual = len(client_data_idxs)
        print(f"Warning: num_targets ({num_targets}) > client samples ({len(client_data_idxs)}). Using {num_targets_actual} targets.")
    else:
        num_targets_actual = num_targets

    if num_targets_actual == 0: # No targets to sample
        attacker_data_target_idxs = np.array([], dtype=int)
        mask = np.ones(len(client_data_idxs), dtype=bool)
    else:
        sampled_idx = np.random.choice(len(client_data_idxs), size=num_targets_actual, replace=False)
        attacker_data_target_idxs = client_data_idxs[sampled_idx]
        attacker_data_target_idxs.sort()
        mask = np.ones(len(client_data_idxs), dtype=bool)
        mask[sampled_idx] = False
        
    print('attacked data: ', len(attacker_data_target_idxs), attacker_data_target_idxs)

    # attacker dataset
    if dataset_name == 'cifar10':
        train_dataset_attacker = BinaryAttackerCIFAR10(
            root='./data', train=True, download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
            num_targets=num_targets_actual # Use actual num_targets
        )
        if num_targets_actual > 0:
            train_dataset_attacker.data = train_dataset_attacker.data[attacker_data_target_idxs]
            train_dataset_attacker.targets = np.array(train_dataset_attacker.targets)[attacker_data_target_idxs].tolist()
        else: # Handle empty attacker dataset
            train_dataset_attacker.data = np.empty((0, *train_dataset_attacker.data.shape[1:]), dtype=train_dataset_attacker.data.dtype)
            train_dataset_attacker.targets = []


    elif dataset_name == 'celeba':
        train_dataset_attacker = BinaryAttackerCELEBA(
            root='./data/celeba',
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            num_targets=num_targets_actual
        )
        train_dataset_attacker.filename = 'celeba' # Add filename attribute if not present
        if num_targets_actual > 0:
            train_dataset_attacker.samples =  [train_dataset_attacker.samples[i] for i in attacker_data_target_idxs]
            # train_dataset_attacker.imgs was removed in original file, assuming samples is key
            # train_dataset_attacker.targets = np.array(train_dataset_attacker.targets)[attacker_data_target_idxs].tolist() # CelebA might not have targets in same way
            # For CelebA, BinaryAttackerCELEBA might handle its own target logic based on num_targets
            train_dataset_attacker.transfer_samples_to_data() # Ensure this works with potentially empty samples
            # If CelebA targets are just indices for BinaryAttacker, ensure they are set.
            # The provided BinaryAttackerCELEBA might not use .targets in the same way as CIFAR10.
            # It uses self.num_targets.
        else:
            train_dataset_attacker.samples = []
            train_dataset_attacker.targets = [] # Or however BinaryAttackerCELEBA represents no targets
            train_dataset_attacker.transfer_samples_to_data()


    else:
        raise NotImplementedError("This dataset has not been implemented yet.")

    print('num of attacker data: ', len(train_dataset_attacker.data) if hasattr(train_dataset_attacker, 'data') else len(train_dataset_attacker.samples), 
                                   len(train_dataset_attacker.targets) if hasattr(train_dataset_attacker, 'targets') else 0)
    
    if num_targets_actual > 0 : train_dataset_attacker.save_targets(save_targets_path)

    if num_targets_actual > 0: # Only repeat if there's data
        num_repeat = (10*attack_batch_size)//num_targets_actual + 1 if num_targets_actual > 0 else 1
        if num_repeat > 1:
            train_dataset_attacker.repeat(num_repeat)

    benign_data_idxs = client_data_idxs[mask]
    print('num of benign data:', len(benign_data_idxs))
    
    if dataset_name == 'cifar10':
        train_dataset_benign = datasets.CIFAR10(
            root='./data', train=True, download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        train_dataset_benign.data = train_dataset_benign.data[benign_data_idxs]
        train_dataset_benign.targets = np.array(train_dataset_benign.targets)[benign_data_idxs].tolist()
    elif dataset_name == 'celeba':
        train_dataset_benign = get_preprocessed_celeba_dataset('./data/celeba')
        # train_dataset_benign.filename = 'celeba' # get_preprocessed_celeba_dataset returns ImageFolder
        # ImageFolder does not have .filename directly. Assuming it's not strictly needed for benign.
        # Subsetting ImageFolder samples
        original_samples = train_dataset_benign.samples
        train_dataset_benign.samples = [original_samples[i] for i in benign_data_idxs]
        train_dataset_benign.targets = [train_dataset_benign.targets[i] for i in benign_data_idxs] # Also subset targets
        train_dataset_benign.imgs = train_dataset_benign.samples # ImageFolder uses imgs = samples

    else:
        raise NotImplementedError("This dataset has not been implemented yet.")
    
    return train_dataset_benign, train_dataset_attacker

class ClientsGroupMultiTargetAttackedNonIID(object):
    # <<< MODIFIED: Added num_colluders, lambda_reg to __init__ signature >>>
    def __init__(self, dataset_name, batch_size, clients_num, num_targets, device, 
                 batch_size_attack_per=0.5, scale_rate=1, use_model_poison=False, 
                 use_pgd_poison=False, use_critical_poison=0, critical_proportion=0.8, 
                 use_bclayersub_poison=False, use_layer_substitution=False, use_no_poison=False, 
                 global_pruning=True, use_adaptive=False, adaptive_lr=0.2, all_num_clients=5, 
                 ema_scale=0.9999, data_distribution_seed=42, scaled=False,
                 num_colluders=None, lambda_reg=None, defense_technique=None): # <<< MODIFIED
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.device = device
        self.clients_set = [] # <<< MODIFIED: Initialize here
        self.scaled = scaled
        self.test_loader = None
        self.use_model_poison = use_model_poison
        self.use_pgd_poison = use_pgd_poison
        self.use_critical_poison = use_critical_poison
        self.critical_proportion = critical_proportion
        self.use_layer_substitution = use_layer_substitution
        self.use_bclayersub_poison = use_bclayersub_poison
        self.scale_rate = scale_rate
        self.num_targets = num_targets
        self.batch_size_attack_per = batch_size_attack_per
        self.dirichlet_alpha = 0.7
        self.data_distribution_seed = data_distribution_seed
        self.ema_scale = ema_scale
        self.all_num_clients = all_num_clients # This was passed but not stored, storing now
        self.global_pruning = global_pruning
        self.use_adaptive = use_adaptive
        self.adaptive_lr = adaptive_lr
        self.defense_technique = defense_technique
        self.use_no_poison = use_no_poison
        self.num_malicious_clients = 2 # Hardcoded as per original structure
        
        # <<< ADDED START >>>
        self.num_colluders = num_colluders
        self.lambda_reg = lambda_reg
        # <<< ADDED END >>>

        self.data_allocation()

    def data_allocation(self):
        if self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root='./data', train=True, download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        elif self.dataset_name == 'celeba':
            train_dataset = get_preprocessed_celeba_dataset('./data/celeba')
        else:
            raise NotImplementedError("This dataset has not been implemented yet.")

        labels = np.array(train_dataset.targets)
        client_idcs = dirichlet_split_noniid(
            labels, alpha=self.dirichlet_alpha, n_clients=self.clients_num, seed=self.data_distribution_seed)
        
        # <<< MODIFIED: First loop to create and store all client objects >>>
        # self.clients_set is already [], will be populated here
        for i in range(self.clients_num):
            client_data_idxs_current = client_idcs[i] # Renamed client_data_idxs
            
            # Create client-specific dataset
            if self.dataset_name == 'cifar10':
                train_dataset_client = datasets.CIFAR10(
                    root='./data', train=True, download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                train_dataset_client.data = train_dataset.data[client_data_idxs_current] # Use original train_dataset.data
                train_dataset_client.targets = np.array(train_dataset.targets)[client_data_idxs_current].tolist()
            elif self.dataset_name == 'celeba':
                train_dataset_client = get_preprocessed_celeba_dataset('./data/celeba') # Fresh dataset object
                original_samples_celeba = list(train_dataset.samples) # Make a mutable copy
                train_dataset_client.samples = [original_samples_celeba[idx_val] for idx_val in client_data_idxs_current]
                train_dataset_client.targets = [train_dataset.targets[idx_val] for idx_val in client_data_idxs_current]
                train_dataset_client.imgs = train_dataset_client.samples # Standard for ImageFolder
            else: # Should not happen due to earlier check
                raise NotImplementedError 

            # Determine if client is attacker and prepare datasets/loaders
            client_obj = None # To store the created client
            if i < self.num_malicious_clients and not self.use_no_poison:
                batch_size_attack_per_eff = self.batch_size_attack_per # Renamed
                attack_batch_size = int(self.batch_size*batch_size_attack_per_eff)
                benign_batch_size = self.batch_size-attack_batch_size
                if self.use_layer_substitution: # LayerSub uses full batch_size for both
                    attack_batch_size = self.batch_size 
                    benign_batch_size = self.batch_size

                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
                
                # Pass client_data_idxs_current to data_allocation_attacker_noniid
                if self.scaled:
                    train_dataset_benign, dataset_attacker = data_allocation_attacker_noniid(
                    client_data_idxs_current, 
                    attack_batch_size=attack_batch_size,
                    num_targets=self.num_targets, 
                    save_targets_path=f'./images/targets_{self.num_targets}_fl_{i}_{formatted_time}_seed_{self.data_distribution_seed}_{self.defense_technique}_scaled_collusion', # Added client index i
                    seed=self.data_distribution_seed, # This seed is for target selection within this client's data
                    dataset_name=self.dataset_name
                )
                else:
                    train_dataset_benign, dataset_attacker = data_allocation_attacker_noniid(
                    client_data_idxs_current, 
                    attack_batch_size=attack_batch_size,
                    num_targets=self.num_targets, 
                    save_targets_path=f'./images/targets_{self.num_targets}_fl_{i}_{formatted_time}_seed_{self.data_distribution_seed}_{self.defense_technique}_collusion', # Added client index i
                    seed=self.data_distribution_seed, # This seed is for target selection within this client's data
                    dataset_name=self.dataset_name
                )
                
                train_loader_client = DataLoader(
                    train_dataset_benign, batch_size=benign_batch_size, shuffle=True, drop_last=True, num_workers=8)
                
                # Ensure dataset_attacker.targets exists and is not empty before calculating min for batch_size
                attacker_bs_actual = attack_batch_size
                if hasattr(dataset_attacker, 'targets') and len(dataset_attacker.targets) > 0:
                     attacker_bs_actual = min(attack_batch_size, len(dataset_attacker.targets))
                elif hasattr(dataset_attacker, 'samples') and len(dataset_attacker.samples) > 0: # For CelebA if targets not used
                     attacker_bs_actual = min(attack_batch_size, len(dataset_attacker.samples))

                if attacker_bs_actual == 0: # Handle empty attacker dataset for loader
                    loader_attacker = DataLoader(dataset_attacker, batch_size=1, shuffle=False) # Dummy loader
                else:
                    loader_attacker = DataLoader(
                        dataset_attacker, batch_size=attacker_bs_actual, shuffle=True, drop_last=True, num_workers=8)

                if self.use_layer_substitution:
                    client_obj = AttackerClientMultiTargetNonIIDLayerSub(
                        i, self.dataset_name, train_dataset_benign, train_loader_client, 
                        dataset_attacker, loader_attacker, self.use_model_poison, self.use_pgd_poison, 
                        self.use_critical_poison, self.critical_proportion, self.use_bclayersub_poison,
                        self.scale_rate, self.device, ema_scale=self.ema_scale,
                        num_colluders=self.num_colluders, lambda_reg=self.lambda_reg) # <<< MODIFIED
                else:
                    client_obj = AttackerClientMultiTargetNonIID(
                        i, self.dataset_name, train_dataset_benign, train_loader_client, 
                        dataset_attacker, loader_attacker, self.use_model_poison, self.use_pgd_poison, 
                        self.use_critical_poison, self.critical_proportion,
                        self.scale_rate, self.global_pruning, self.use_adaptive, self.adaptive_lr,
                        self.device, all_num_clients=self.all_num_clients, ema_scale=self.ema_scale,
                        num_colluders=self.num_colluders, lambda_reg=self.lambda_reg) # <<< MODIFIED
            else: # Benign client
                train_loader_client = DataLoader(
                    train_dataset_client, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=8)
                client_obj = ClientNonIID(
                    i, self.dataset_name, train_dataset_client, train_loader_client, self.device)

            self.clients_set.append(client_obj)

        # <<< ADDED: Second loop to set colluder references for attackers >>>
        colluder_ids = list(range(self.num_malicious_clients)) # IDs of malicious clients
        for i in range(self.clients_num):
            # Check if the current client is an attacker and has the set_colluder_refs method
            if i in colluder_ids and hasattr(self.clients_set[i], 'set_colluder_refs') and not self.use_no_poison :
                other_colluders_refs = []
                for j_colluder_id in colluder_ids: # Iterate through all potential colluder IDs
                    if i == j_colluder_id:
                        continue # Don't add self
                    # Add reference if the other client is also an attacker type (has global_model)
                    if hasattr(self.clients_set[j_colluder_id], 'global_model'):
                         other_colluders_refs.append(self.clients_set[j_colluder_id])
                
                self.clients_set[i].set_colluder_refs(other_colluders_refs)


import matplotlib.pyplot as plt
if __name__ == "__main__":
    n_clients = 5
    dirichlet_alpha = 0.7
    train_data = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))

    classes = train_data.classes
    n_classes = len(classes)

    labels = np.array(train_data.targets)
    # dataset = train_data # This line was in original but dataset var not used later in if __name__

    client_idcs_main = dirichlet_split_noniid( # Renamed client_idcs
        labels, alpha=dirichlet_alpha, n_clients=n_clients, seed=42)
    print(client_idcs_main)

    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc_list in enumerate(client_idcs_main): # Renamed idc
        for idx_val_main in idc_list: # Renamed idx
            label_distribution[labels[idx_val_main]].append(c_id)

    plt.hist(label_distribution, stacked=True,
                bins=np.arange(-0.5, n_clients + 1.5, 1),
                label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                        c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.savefig('./dirichlet_42.png')