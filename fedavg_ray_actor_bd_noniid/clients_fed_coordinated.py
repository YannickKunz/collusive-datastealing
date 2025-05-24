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

# Shared state for colluding attackers
class ColludersSharedState:
    def __init__(self):
        self.round_data = {}
        self.aggregated_result = None

    def submit(self, client_id, f_value):
        self.round_data[client_id] = f_value
        logging.info(f"[ColluderState] Client {client_id} submitted f_value: {f_value}")


    def aggregate(self, adaptive_k, all_num_clients):
        logging.info(f"[ColluderState AGGREGATE] Received round_data: {self.round_data}")

        tars_j_list = []
        processed_f_values = [] # To log f_values that were processed

        # adaptive_k should be a single numeric value.
        # If it's passed as a tuple/list from the client (e.g., self.adaptive_k could be),
        # ensure you get the numeric value. Usually, adaptive_k is a single hyperparameter.
        if isinstance(adaptive_k, (tuple, list)):
            if adaptive_k: # Check if not empty
                adaptive_k_val = adaptive_k[0]
                logging.warning(f"[ColluderState AGGREGATE] adaptive_k was a list/tuple, using first element: {adaptive_k_val}")
            else:
                logging.error("[ColluderState AGGREGATE] adaptive_k is an empty list/tuple! Cannot proceed.")
                self.aggregated_result = None
                return
        else:
            adaptive_k_val = adaptive_k

        if not isinstance(adaptive_k_val, (int, float)):
            logging.error(f"[ColluderState AGGREGATE] adaptive_k_val is not a number: {adaptive_k_val}! Cannot proceed.")
            self.aggregated_result = None
            return


        for client_id, f_j_submitted_val in self.round_data.items():
            processed_f_values.append(f_j_submitted_val)
            if f_j_submitted_val is not None and not isinstance(f_j_submitted_val, (int, float)):
                logging.warning(f"[ColluderState AGGREGATE] Client {client_id}: f_j={f_j_submitted_val} is type {type(f_j_submitted_val)}, expected number. Skipping.")
                continue

            if f_j_submitted_val is not None and f_j_submitted_val != 1:
                try:
                    logging.info(f"[ColluderState AGGREGATE DEBUG] For Client {client_id}: adaptive_k_val={adaptive_k_val}, f_j_submitted_val={f_j_submitted_val}")
                    tars = (adaptive_k_val - 1) / (f_j_submitted_val - 1)
                    logging.info(f"[ColluderState AGGREGATE] Client {client_id}: f_j={f_j_submitted_val}, calculated_tars_j={tars}")
                    if 0 < tars <= 2 * all_num_clients:
                        tars_j_list.append(tars)
                    else:
                        logging.info(f"[ColluderState AGGREGATE] Client {client_id}: tars_j={tars} is OUT OF RANGE (0, {2 * all_num_clients}]")
                except ZeroDivisionError:
                    logging.warning(f"[ColluderState AGGREGATE] Client {client_id}: f_j={f_j_submitted_val} caused ZeroDivisionError (f_j - 1 was 0). Skipping.")
                except TypeError as e:
                    logging.error(f"[ColluderState AGGREGATE] Client {client_id}: TypeError for f_j={f_j_submitted_val}, adaptive_k={adaptive_k_val}. Error: {e}. Skipping.")

            else:
                logging.info(f"[ColluderState AGGREGATE] Client {client_id}: f_j={f_j_submitted_val} is None or 1, skipping tars_j calculation.")

        if len(tars_j_list) > 0:
            self.aggregated_result = float(np.mean(tars_j_list))
            logging.info(f"[ColluderState AGGREGATE] Valid tars_j values for aggregation: {tars_j_list}")
            logging.info(f"[ColluderState AGGREGATE] Corresponding f_values processed: {processed_f_values}")
            logging.info(f"[ColluderState AGGREGATE] Final tars_coll (aggregated_result): {self.aggregated_result}")
        else:
            self.aggregated_result = None
            logging.info(f"[ColluderState AGGREGATE] No valid tars_j values found from f_values {processed_f_values}. tars_coll is None.")

    def get_aggregated(self):
        return self.aggregated_result

    def reset(self):
        logging.info(f"[ColluderState RESET] Resetting round_data and aggregated_result.")
        self.round_data = {}
        self.aggregated_result = None



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
        global_loss = 0
        iterations = 0
        while True: ### max iteration is 10000
        # for epoch in range(local_epoch):
            # with tqdm(self.train_loader, dynamic_ncols=True,
            #           desc=f'round:{round+1} client:{self.client_id}') as pbar:
                # for x, label in pbar:
            for x, label in self.train_loader:
                x, label = x.to(self.device), label.to(self.device)
                if use_labels:
                    global_loss = self.global_trainer(x, 0, 1000, label)
                else:
                    global_loss = self.global_trainer(x, 0, 1000)

                # global update
                self.global_optim.zero_grad()
                global_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                self.global_optim.step()
                self.global_sched.step()
                self.ema(self.global_model, self.global_ema_model, 0.9999)

                # log
                # pbar.set_postfix(global_loss='%.3f' % global_loss, lr='%.6f' % self.global_sched.get_last_lr()[-1])
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

class AttackerClient(object):
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
        self.global_trainer = GaussianDiffusionAttackerTrainer(
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
        global_loss = 0
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
                    global_loss = self.global_trainer(x, self.miu, 0, 1000, label)
                else:
                    global_loss = self.global_trainer(x, self.miu, 0, 1000)

                # global update
                self.global_optim.zero_grad()
                global_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                self.global_optim.step()
                self.global_sched.step()
                self.ema(self.global_model, self.global_ema_model, 0.9999)

                # log
                # pbar.set_postfix(global_loss='%.3f' % global_loss, lr='%.6f' % self.global_sched.get_last_lr()[-1])
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
                 shared_state=None):
        self.dataset_name = dataset_name
        if dataset_name == 'cifar10':
            self.img_size = 32
        elif dataset_name == 'celeba':
            self.img_size = 64
        self.img_size = attacker_dataset.img_size
        self.client_id = client_id
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
        self.global_pruning = global_pruning
        self.use_adaptive = use_adaptive
        self.shared_state = shared_state

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
                for x in self.train_loader:
                    x_tar = x[0].to(self.device)[:batch_for_diff]
                    # global_loss = self.global_trainer(x_tar, self.miu, mask_trigger, count, count+1)
                    global_loss = self.global_trainer(x_tar, self.miu, None, count, count+1)
                    global_loss = global_loss.mean()

                    ### record gradient and hessian matrix
                    if self.use_adaptive:
                        grad = torch.autograd.grad(global_loss,
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
                        
                        # for idx, c in enumerate(curv):
                        #     if c is not None:
                        #         if c.reshape(-1).shape[0]>512:
                        #             min_values, min_indices = torch.topk(torch.abs(c.reshape(-1)), k=512, largest=False)
                        #             min_values = min_values.cpu().numpy()  # 转换为NumPy数组
                        #             min_indices = min_indices.cpu().numpy()
                        #             logging.info(f'param: {last_layers_params[idx].reshape(-1)[min_indices[:5]]}')
                        #             logging.info(f'grad: {allgrad[idx].reshape(-1)[min_indices[:5]]}')
                        #             logging.info(min_values[:5])
                        #         logging.info(f'{idx}, {c.shape}')
                        #     else:
                        #         logging.info(idx)

                        curv = curv[nolayer]
                        gradient += grad.detach().cpu()
                        curvature += curv.detach().cpu()

                    global_loss.backward()
                    count += 1
                    if count > 998:
                        break
                    if global_loss>max_loss:
                        max_loss = global_loss
                    if global_loss<max_loss*thr:
                        count = 1000
                        break
                if count > 998:
                    break

            # end_event.record()
            # torch.cuda.synchronize()
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # logging.info(f"GPU运行时间: {elapsed_time_ms:.2f} ms")

            idx = 0
            
            # base_macs, base_nparams = tp.utils.count_ops_and_params(self.global_model, example_inputs)
            sum = 0
            weight_sum = 0
            num_pruned_layer = 0
            # for group in pruner.DG.get_all_groups(ignored_layers=pruner.ignored_layers, root_module_types=pruner.root_module_types):
            for group in pruner.step(interactive=True):
                # logging.info(group)
                # for i, (dep, idxs) in enumerate(group):
                #     logging.info(f"Dep: {dep}, Idxs: {idxs}")
                # group = pruner._downstream_node_as_root_if_attention(group) # use a downstream node as the root node for attention layers
                # ch_groups = pruner._get_channel_groups(group)
                # imp = pruner.estimate_importance(group)
                # dim_imp = imp.view(ch_groups, -1).mean(dim=0)
                ### importance -> weight mask
                # os.makedirs('tmp/pruning_logs_benign', exist_ok=True)
                # # draw bar for imp
                # plt.figure()
                # plt.bar(range(len(imp)), imp.cpu().numpy())
                # plt.savefig(f'tmp/pruning_logs_benign/imp_{idx}.png')
                # plt.close()
                # idx += 1
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
                        # logging.info(layer)
                        # logging.info(pruning_fn)
                        # logging.info(f'{idxs}_{candidate_weights[layer_weight].shape}')

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
                        candidate_weights[layer_bias] = candidate_weights[layer_bias] == 0
                        # logging.info(f'{candidate_weights[layer_weight].sum()}, {candidate_weights[layer_weight].sum() == candidate_weights[layer_weight].numel()}')
                        
                        if self.use_adaptive and layer_weight == self.indicator_layer_name:
                            self.indicator_candidate = copy.deepcopy(candidate_weights[layer_weight])
                        
                        tmp_sum += candidate_weights[layer_weight].sum()
                        weight_sum += candidate_weights[layer_weight].numel()

                num_pruned_layer += 1
                sum += tmp_sum
                logging.info(f'{group[0][0].target._name}, tmp_sum: {tmp_sum}')
                # group.prune()

            if self.use_adaptive:
                gradient = torch.abs(gradient.reshape(-1))
                curvature = torch.abs(curvature.reshape(-1))
                if not self.indicator_layer_name in candidate_weights.keys(): # if not prunned, select from all params
                    self.indicator_candidate = torch.ones(128, 256, 3, 3)
                self.indicator_candidate = self.indicator_candidate.reshape(-1)
                min_values, min_indices = torch.topk(gradient, k=512, largest=False)
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
                        elif curvature[ind] == 0:
                            temp = val_cur.index(max(val_cur))
                            if gradient[ind] < gradient[self.indicator_indices[temp]]:
                                val_cur[temp] = curvature[ind]
                                self.indicator_indices[temp] = ind

                for idx in self.indicator_indices:
                    i0 = torch.div(idx, (256*3*3), rounding_mode='trunc')
                    i1 = torch.div(idx%(256*3*3), (3*3), rounding_mode='trunc')
                    i2 = torch.div(idx%(3*3),3, rounding_mode='trunc')
                    i3 = idx%3
                    self.indicator_indices_mat.append([i0,i1,i2,i3])
                    # if prunned: set not update gradient = False
                    if self.indicator_layer_name in candidate_weights.keys():
                        candidate_weights[self.indicator_layer_name][i0,i1,i2,i3] = False

            if proportion <= 0.3 and num_pruned_layer < 47:
                logging.info('Error: low proportion with low pruned layer, check whether torch_prunnig has not been modified!')
                assert(0)
            self.stored_model_weights = model_weights
            self.global_model.zero_grad()
            # torch.save(self.global_model, 'model.pth')
            logging.info(f'sum: {sum}, weight_sum: {weight_sum}, proportion: {sum/weight_sum}')
            # logging.info(self.global_model)
            # macs, nparams = tp.utils.count_ops_and_params(self.global_model, example_inputs)
            # logging.info(
            #             "Params: %.2f M => %.2f M"
            #             % (base_nparams / 1e6, nparams / 1e6)
            #         )
            # logging.info(
            #             "MACs: %.2f G => %.2f G"
            #             % (base_macs / 1e9, macs / 1e9)
            #         )

        elif self.use_critical_poison==3:
            history_weights = self.previous_global_model.state_dict()
            length=len(candidate_layers)
            ### keep proportion weight [AAAI 2023 On the Vulnerability of Backdoor Defenses for Federated Learning]
            for layer in history_weights.keys():
                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                ### keep proportion weight
                theta = torch.sort(candidate_weights[layer].flatten(), descending=True)[0][int(n_weight * proportion)]
                candidate_weights[layer] = candidate_weights[layer] < theta
                # candidate_weights[layer][candidate_weights[layer] != 1] = 0
            self.stored_model_weights = model_weights
        else:
            candidate_layers = [1 for _ in range(len(model_weights.keys()))]
            # for layer in self.global_model.state_dict().keys():
            #     candidate_weights[layer] = model_weights[layer]
            #     candidate_weights[layer][candidate_weights[layer] != 1] = 1
        
        self.previous_global_model = copy.deepcopy(self.global_model)
        
        if self.use_critical_poison<2:
            return candidate_layers
        elif self.use_critical_poison == 2 or self.use_critical_poison == 3:
            return candidate_weights

    def reset_weight(self, mask):
        for key, value in self.global_model.state_dict().items():
            if key in mask.keys():
                value[mask[key]] = self.stored_model_weights[key][mask[key]]
                # logging.info(f'exists: {key}')
                # logging.info(mask[key])
            # logging.info((value - self.global_model.state_dict()[key]).sum())

    def update_layer_require_grad(self, candidate_layers):
        count = 0
        for (name, param), requires_grad in zip(self.global_model.named_parameters(), candidate_layers):
            param.requires_grad = bool(requires_grad)
            count += 1
        assert(count == len(candidate_layers))
        for name, param in self.global_model.named_parameters():
            logging.info((f"{name}: requires_grad={param.requires_grad}"))

    def read_indicator(self):
        feedback = []
        num_chosen_client = []
        weight_g = self.global_model.state_dict()
        # self.his_scale_rate.append(self.scale_rate)
        # Calculate feedback for each indicator (here, only one is used)
        for i in range(1):
            idx = self.indicator_indices_mat[i]
            # Compute the change in the indicator parameter
            delta = weight_g[self.indicator_layer_name][idx].item() - self.indicator_param[i][0]
            delta_x = self.indicator_param[i][1]
            
            # Feedback: ratio of global update to local update
            feedback.append( delta / delta_x) # real global update / real local update
            logging.info(f'delta: {delta}')
            logging.info(delta_x)
            logging.info(feedback)

        # Estimate the number of clients that would have produced the observed feedback
        for i, f in enumerate(feedback):
            idx = self.indicator_indices_mat[i]
            if f > 1:
                tmp_num = (self.adaptive_k - 1) / (f - 1)
                # Only consider reasonable client numbers
                if 0 < tmp_num <= 2 * self.all_num_clients:
                    num_chosen_client.append(tmp_num)
                logging.info(tmp_num)

        if len(num_chosen_client) > 0:
            # If feedback is valid, accept and update scale rate
            self.his_accepted.append(copy.deepcopy(self.scale_rate))
            self.his_accepted.sort()
            # New scale is the average estimated client number
            new_scale = sum(num_chosen_client) / len(num_chosen_client)

            # Optionally limit new scale by the 20th percentile of rejected scale rates
            if len(self.his_rejected) > 10:
                pos = int(len(self.his_rejected) * 0.2)
                median_reject_scale_rate = self.his_rejected[pos]
                new_scale = min(median_reject_scale_rate, new_scale)
                logging.info(f'median_reject_scale_rate: {median_reject_scale_rate}')

            # Update scale_rate with momentum (adaptive_lr)
            self.scale_rate = self.scale_rate * (1 - self.adaptive_lr) + new_scale * self.adaptive_lr
            self.accepted_before = True
            logging.info(f'scale_rate: {self.scale_rate}, {num_chosen_client}')

        else:
            # If feedback is not valid, reject and decrease scale rate
            self.his_rejected.append(copy.deepcopy(self.scale_rate))
            self.his_rejected.sort()
            if len(self.his_accepted) > 10:
                pos = int(len(self.his_accepted) * 0.5)
                median_accept_scale_rate = self.his_accepted[pos]
                # Use the median of accepted scale rates to avoid dropping too low
                self.scale_rate = max(self.scale_rate * (1 - self.adaptive_lr), median_accept_scale_rate)
            else:
                self.scale_rate = self.scale_rate * (1 - self.adaptive_lr)
            # If previously accepted but now rejected, decrease learning rate for finer search
            if self.accepted_before:
                self.adaptive_lr = self.adaptive_lr * self.adaptive_decay
            self.accepted_before = False

            logging.info(f'scale_rate: {self.scale_rate}, {num_chosen_client} ')
        logging.info(f'adaptive_lr: {self.adaptive_lr} {self.his_accepted} {self.his_rejected}')

        # Ensure scale_rate does not go below 0.8
        self.scale_rate = max(self.scale_rate, 0.8)

        # Reset indicator weights to their original value plus the local update
        for i in range(1):
            idx = self.indicator_indices_mat[i]
            weight_g[self.indicator_layer_name][idx] = self.indicator_param[i][0] + self.indicator_param[i][1]

        # Remove the used indicator index from the list
        self.indicator_indices_mat = self.indicator_indices_mat[1:]

    # def set_final_scale_rate(self):
    #     valid_v = []
    #     for acc, v in zip(self.his_accepted, self.his_scale_rate):
    #         if acc:
    #             valid_v.append(v)
    #     valid_v.sort()
    #     valid_large = valid_v[len(valid_v)//2:]
    #     self.scale_rate = valid_large[len(valid_large)//2] # 50% median

    def local_train(self, round, local_epoch, mid_T, use_labels=True):
        self.round = round

        if self.use_adaptive and round > 0:
            #self.read_indicator()
            self._update_scale_from_tars_coll()

        self.global_trainer.train()
        global_loss = 0
        if self.use_pgd_poison:
            model_original_vec = vectorize_net(self.target_model)
        eps = 8.0
        # if self.use_lora_poison > 0:
        #     self.init_lora(0.0002)
        logging.info('round: '+str(round))
        if self.use_critical_poison == 1 and round == 10:
            sched_state_dict = self.global_sched.state_dict()
            candidate_layers = self.search_candidate_weights(self.critical_proportion)
            logging.info(candidate_layers)
            self.update_layer_require_grad(candidate_layers)
            self.global_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.global_model.parameters()), lr=self.lr)
            self.global_sched = torch.optim.lr_scheduler.LambdaLR(self.global_optim, lr_lambda=self.warmup_lr)
            self.global_sched.load_state_dict(sched_state_dict)
            # self.global_sched._last_lr = last_lr
            self.global_sched.step()
        # elif self.use_critical_poison == 2 and round == 0:
        elif (self.use_critical_poison == 2 and round == 0) or (self.use_adaptive and len(self.indicator_indices_mat) == 0):
            self.candidate_weights = self.search_candidate_weights(self.critical_proportion)
        elif self.use_critical_poison == 3 and round == 1:
            self.candidate_weights = self.search_candidate_weights(self.critical_proportion)

        count = 0
        while True:
            for x1, x2 in zip(self.train_loader, cycle(self.attacker_loader)):
                x, label = x1[0].to(self.device), x1[1].to(self.device)
                ### add attack samples
                x_tar, y_tar, mask_trigger = x2[0].to(self.device), x2[1].to(self.device), x2[2].to(self.device)
                x = torch.cat([x, x_tar], dim=0)
                label= torch.cat([label, y_tar], dim=0)

                if use_labels:
                    global_loss = self.global_trainer(x, self.miu, mask_trigger, 0, 1000, label)
                else:
                    global_loss = self.global_trainer(x, self.miu, mask_trigger, 0, 1000)

                self.global_optim.zero_grad()
                # global update
                if self.use_pgd_poison:
                    self.adv_global_optim.zero_grad()
                    global_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.adv_global_optim.step()
                    self.adv_global_sched.step()
                    # pgd_l2_projection()
                    # do l2_projection
                    w_vec = vectorize_net(self.global_model)
                    # make sure you project on last iteration otherwise, high LR pushes you really far
                    val_norm = torch.norm(w_vec - model_original_vec)
                    # logging.info(f'val_norm: {val_norm}')
                    
                    if (val_norm > eps):
                        # project back into norm ball
                        # w_proj_vec = eps * (w_vec - model_original_vec) / torch.norm(
                        #     w_vec - model_original_vec) + model_original_vec
                        scale = eps / val_norm
                        # plug w_proj back into model
                        # load_model_weight(self.global_model, w_proj_vec)
                        for key, value in self.global_model.state_dict().items():
                                target_value = self.target_model.state_dict()[key]
                                new_value = target_value + (value - target_value) * scale
                                self.global_model.state_dict()[key].copy_(new_value)

                else:
                    global_loss = global_loss.mean()
                    global_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), 1.)
                    self.global_optim.step()
                    self.global_sched.step()
                    
                if self.use_critical_poison == 2 or (self.use_critical_poison == 3 and round > 1):
                    self.reset_weight(self.candidate_weights)

                self.ema(self.global_model, self.global_ema_model, self.ema_scale)
                # log
                # logging.info((f'step: %d, loss: %.3f, lr: %f' % 
                #               (self._step_cound, global_loss, self.global_sched.get_last_lr()[0])))
                
                self._step_cound += 1
                count += x.shape[0]
                if count + x.shape[0] > 10000: # keep the same iteration with other clients
                    break
            if count + x.shape[0] > 10000: # keep the same iteration with other clients
                break
        
        ### We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
        if self.use_model_poison or self.use_critical_poison > 0:
            for key, value in self.global_model.state_dict().items():
                    target_value = self.target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * self.scale_rate
                    self.global_model.state_dict()[key].copy_(new_value)

            for key, value in self.global_ema_model.state_dict().items():
                    target_value = self.target_ema_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * self.scale_rate
                    self.global_ema_model.state_dict()[key].copy_(new_value)
            # return self.global_model.state_dict(), self.global_ema_model.state_dict()
            # return self.global_model, self.global_ema_model
        # else:   
            # return self.global_model.state_dict(), self.global_ema_model.state_dict()

        if self.use_adaptive and (self.use_model_poison or self.use_critical_poison > 0):
            self.indicator_param = []
            weight_g = self.global_model.state_dict()
            weight_pre = self.target_model.state_dict()
            ### when to remove the affect of adaptive_k?
            for index in [self.indicator_indices_mat[0]]:
                # avoid zero value
                if (weight_g[self.indicator_layer_name][index] - weight_pre[self.indicator_layer_name][index]) == 0:
                    self.indicator_param.append((copy.deepcopy(weight_pre[self.indicator_layer_name][index]), (1e-4/self.adaptive_k)))
                    weight_g[self.indicator_layer_name][index].add_(1e-4)
                    
                else:   
                    # save (G(r-1), delta_x))
                    delta_x = (weight_g[self.indicator_layer_name][index] - weight_pre[self.indicator_layer_name][index]) / self.scale_rate
                    self.indicator_param.append((copy.deepcopy(weight_pre[self.indicator_layer_name][index]), delta_x))
                    weight_g[self.indicator_layer_name][index] = weight_pre[self.indicator_layer_name][index] + delta_x * self.adaptive_k
                    
                
            logging.info(f'indicator_param: {self.indicator_param}')
            logging.info(f'indicator_param_update: {(weight_g[self.indicator_layer_name][index] - weight_pre[self.indicator_layer_name][index])}')

        logging.info(f'scale rate: {self.scale_rate}')
        # Colluder communication: submit f_j
        if self.shared_state is not None and self.use_adaptive and (self.use_model_poison or self.use_critical_poison > 0):
            f_j = self._calculate_f_value()
            self.shared_state.submit(self.client_id, f_j)

        return self.global_model, self.global_ema_model

    def prepare_for_next_round(self):
        self.prepare_for_collusion_round()
    

    def _calculate_f_value(self):
            """Calculate f_j for collusion."""
            weight_g = self.global_model.state_dict()
            weight_pre_indicator_val = self.indicator_param[0][0] # This is G_r-1_indicator
            delta_x_local = self.indicator_param[0][1] # This is (w_r-1_indicator - G_r-1_indicator) / scale_rate_r-1

            idx_coords = self.indicator_indices_mat[0] # Use tuple indexing for clarity if it's a list of lists/tuples
            # Assuming idx_coords is [i0, i1, i2, i3]
            current_global_indicator_val = weight_g[self.indicator_layer_name][idx_coords[0], idx_coords[1], idx_coords[2], idx_coords[3]].item()

            delta_global = current_global_indicator_val - weight_pre_indicator_val.item() # G_r_indicator - G_r-1_indicator
            # weight_pre = self.target_model.state_dict()
            # idx = self.indicator_indices_mat[0]
            # # Compute the change in the indicator parameter
            # delta_global = weight_g[self.indicator_layer_name][idx].item() - self.indicator_param[0][0]
            # delta_x_local = self.indicator_param[0][1]
            if delta_x_local == 0:
                f_j = None
                logging.warning(f"[Client {self.client_id} CALC_F_VALUE] delta_x_local is 0. f_j is None.")
            else:
                f_j = delta_global / delta_x_local.item() # Ensure delta_x_local is also a scalar if tensor
            
            logging.info(f"[Client {self.client_id} CALC_F_VALUE] Round {self.round}: G_r_ind={current_global_indicator_val:.4f}, G_r-1_ind={weight_pre_indicator_val.item():.4f}, "
                         f"delta_global={delta_global:.4f}, delta_x_local_from_prev_round={delta_x_local.item():.4f}, calculated_f_j={f_j}")
            return f_j

    def prepare_for_collusion_round(self):
        """Get the aggregated tars_coll from shared state."""
        if self.shared_state is not None:
            self.current_tars_coll = self.shared_state.get_aggregated()
            logging.info(f"[Client {self.client_id} PREPARE_COLLUSION] Round {self.round}: Fetched current_tars_coll: {self.current_tars_coll}")
        else:
            self.current_tars_coll = None
            logging.info(f"[Client {self.client_id} PREPARE_COLLUSION] Round {self.round}: No shared state, current_tars_coll is None.")

    def _update_scale_from_tars_coll(self):
        """Update scale_rate and adaptive_lr based on colluders' aggregation."""
        logging.info(f"[Client {self.client_id} UPDATE_SCALE] Round {self.round}: Received current_tars_coll: {self.current_tars_coll}")
        logging.info(f"[Client {self.client_id} UPDATE_SCALE] Before update: scale_rate={self.scale_rate:.4f}, adaptive_lr={self.adaptive_lr:.4f}, accepted_before={self.accepted_before}")
        logging.info(f"[Client {self.client_id} UPDATE_SCALE] History: accepted={self.his_accepted}, rejected={self.his_rejected}")
        num_chosen_client = []
        if self.current_tars_coll is not None:
            tars = self.current_tars_coll
            if 0 < tars <= 2 * self.all_num_clients:
                num_chosen_client.append(tars)
                logging.info(f"[Client {self.client_id} UPDATE_SCALE] tars_coll={tars:.4f} is valid and added.")
            else:
                logging.info(f"[Client {self.client_id} UPDATE_SCALE] tars_coll={tars:.4f} is invalid (out of range (0, {2*self.all_num_clients}]).")

        if len(num_chosen_client) > 0:
            self.his_accepted.append(copy.deepcopy(self.scale_rate))
            self.his_accepted.sort()
            new_scale = sum(num_chosen_client) / len(num_chosen_client)
            logging.info(f"[Client {self.client_id} UPDATE_SCALE] ACCEPTED state. new_scale_target (from tars_coll): {new_scale:.4f}")
            if len(self.his_rejected) > 10:
                pos = int(len(self.his_rejected) * 0.2)
                median_reject_scale_rate = self.his_rejected[pos]
                new_scale = min(median_reject_scale_rate, new_scale)
                logging.info(f"[Client {self.client_id} UPDATE_SCALE] Capped new_scale_target by 20th percentile of rejected: {median_reject_scale_rate:.4f} -> {new_scale:.4f}")

            self.scale_rate = self.scale_rate * (1 - self.adaptive_lr) + new_scale * self.adaptive_lr
            self.accepted_before = True
        else:
            self.his_rejected.append(copy.deepcopy(self.scale_rate))
            self.his_rejected.sort()
            logging.info(f"[Client {self.client_id} UPDATE_SCALE] REJECTED state.")
            if len(self.his_accepted) > 10:
                pos = int(len(self.his_accepted) * 0.5)
                median_accept_scale_rate = self.his_accepted[pos]
                self.scale_rate = max(self.scale_rate * (1 - self.adaptive_lr), median_accept_scale_rate)
                logging.info(f"[Client {self.client_id} UPDATE_SCALE] Lower bounded scale_rate by median of accepted: {median_accept_scale_rate:.4f}. scale_rate_prime was {scale_rate:.4f}")
            else:
                self.scale_rate = self.scale_rate * (1 - self.adaptive_lr)
            if self.accepted_before:
                self.adaptive_lr = self.adaptive_lr * self.adaptive_decay
                logging.info(f"[Client {self.client_id} UPDATE_SCALE] adaptive_lr decayed to: {self.adaptive_lr:.4f}")
            self.accepted_before = False
        self.scale_rate = max(self.scale_rate, 0.8)

        logging.info(f"[Client {self.client_id} UPDATE_SCALE] After update: scale_rate={self.scale_rate:.4f}, adaptive_lr={self.adaptive_lr:.4f}, accepted_before={self.accepted_before}")

        # Consume the indicator
        if self.indicator_indices_mat:
            consumed_indicator = self.indicator_indices_mat.pop(0)
            logging.info(f"[Client {self.client_id} UPDATE_SCALE] Consumed indicator: {consumed_indicator}. Remaining: {len(self.indicator_indices_mat)}")
        else:
            logging.warning(f"[Client {self.client_id} UPDATE_SCALE] Attempted to consume indicator, but indicator_indices_mat is empty!")

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
            # sample = self.global_ema_sampler(x_T, start_step, end_step)
        else:
            sample = self.global_ema_sampler(x_T, start_step, end_step, labels)
        self.global_ema_model.train()
        return sample

def data_allocation_attacker_noniid(client_data_idxs, attack_batch_size, num_targets, save_targets_path, seed=42, dataset_name='cifar10'):
    ''' 
    split a training dataset into two parts.
     - one is for normal training
     - another is for attaker
    '''
    np.random.seed(seed)
    print(seed)
    sampled_idx = np.random.choice(len(client_data_idxs), size=num_targets, replace=False)
    attacker_data_target_idxs = client_data_idxs[sampled_idx]
    attacker_data_target_idxs.sort()
    print('attacked data: ', len(attacker_data_target_idxs), attacker_data_target_idxs)
    mask = np.ones(len(client_data_idxs), dtype=bool)
    mask[sampled_idx] = False

    # attacker dataset
    if dataset_name == 'cifar10':
        train_dataset_attacker = BinaryAttackerCIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]),
            num_targets=num_targets
        )
        train_dataset_attacker.data = train_dataset_attacker.data[attacker_data_target_idxs]

    elif dataset_name == 'celeba':
        train_dataset_attacker = BinaryAttackerCELEBA(
            root='./data/celeba',
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            num_targets=num_targets
        )
        train_dataset_attacker.filename = 'celeba'
        train_dataset_attacker.samples =  [train_dataset_attacker.samples[i] for i in attacker_data_target_idxs]
        train_dataset_attacker.imgs =  [train_dataset_attacker.imgs[i] for i in attacker_data_target_idxs]
        train_dataset_attacker.transfer_samples_to_data()
    else:
        raise NotImplementedError("This dataset has not been implemented yet.")

    ### set dataset to attacker_data_target_idxs
    train_dataset_attacker.targets = np.array(train_dataset_attacker.targets)[
        attacker_data_target_idxs].tolist()
    print('num of attacker data: ', len(train_dataset_attacker.data), len(train_dataset_attacker.targets))
    
    train_dataset_attacker.save_targets(save_targets_path)

    # 10 x times for quicker load
    num_repeat = 10*attack_batch_size//len(attacker_data_target_idxs) + 1
    if num_repeat > 1:
        train_dataset_attacker.repeat(num_repeat)

    # datasets of benign training
    benign_data_idxs = client_data_idxs[mask]
    print('num of benign data:', len(benign_data_idxs))
    
    if dataset_name == 'cifar10':
        train_dataset_benign = datasets.CIFAR10(
            root='./data',
            train=True,
            download=False,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        train_dataset_benign.data = train_dataset_benign.data[benign_data_idxs]
    elif dataset_name == 'celeba':
        train_dataset_benign = get_preprocessed_celeba_dataset('./data/celeba')
        train_dataset_benign.filename = 'celeba'
        train_dataset_benign.samples =  [train_dataset_benign.samples[i] for i in benign_data_idxs]
        train_dataset_benign.imgs =  [train_dataset_benign.imgs[i] for i in benign_data_idxs]
    else:
        raise NotImplementedError("This dataset has not been implemented yet.")

    train_dataset_benign.targets = np.array(train_dataset_benign.targets)[
        benign_data_idxs].tolist()
    
    return train_dataset_benign, train_dataset_attacker

class ClientsGroupMultiTargetAttackedNonIID(object):

    def __init__(self, dataset_name, batch_size, clients_num, num_targets, device, 
                 batch_size_attack_per=0.5, scale_rate=1, use_model_poison=False, 
                 use_pgd_poison=False, use_critical_poison=0, critical_proportion=0.8, 
                 use_bclayersub_poison=False, use_layer_substitution=False, use_no_poison=False, 
                 global_pruning=True, use_adaptive=False, adaptive_lr=0.2, all_num_clients=5, 
                 ema_scale=0.9999, data_distribution_seed=42):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.clients_num = clients_num
        self.device = device
        self.clients_set = []
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
        self.all_num_clients = all_num_clients
        self.global_pruning = global_pruning
        self.use_adaptive = use_adaptive
        self.adaptive_lr = adaptive_lr
        self.use_no_poison = use_no_poison

        self.colluding_clients = 2
        self.colluders_shared_state = ColludersSharedState()
        self.data_allocation()
    def data_allocation(self):
        if self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root='./data',
                train=True,
                download=False,
                transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
        elif self.dataset_name == 'celeba':
            # self.seed = 30
            train_dataset = get_preprocessed_celeba_dataset('./data/celeba')
        else:
            raise NotImplementedError("This dataset has not been implemented yet.")

        labels = np.array(train_dataset.targets)

        # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
        client_idcs = dirichlet_split_noniid(
            labels, alpha=self.dirichlet_alpha, n_clients=self.clients_num, seed=self.data_distribution_seed)
        for i in client_idcs:
            # i.sort # this will lead to unconsistent images
            print(i, len(i))

        for i in range(self.clients_num):
            # non-iid per client
            client_data_idxs = client_idcs[i]
            print(len(client_data_idxs))
            
            if self.dataset_name == 'cifar10':
                train_dataset_client = datasets.CIFAR10(
                    root='./data',
                    train=True,
                    download=False,
                    transform=transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    ]))
                train_dataset_client.data = train_dataset_client.data[client_data_idxs]
            elif self.dataset_name == 'celeba':
                train_dataset_client = get_preprocessed_celeba_dataset('./data/celeba')
                train_dataset_client.samples =  [train_dataset_client.samples[i] for i in client_data_idxs]
                train_dataset_client.imgs =  [train_dataset_client.imgs[i] for i in client_data_idxs]
            else:
                raise NotImplementedError("This dataset has not been implemented yet.")

            train_dataset_client.targets = np.array(train_dataset_client.targets)[
                client_data_idxs].tolist()
            
            if i < self.colluding_clients and self.use_no_poison == False:
                batch_size_attack_per = self.batch_size_attack_per
                attack_batch_size = int(self.batch_size*batch_size_attack_per)
                benign_batch_size = self.batch_size-attack_batch_size
                current_time = datetime.datetime.now()
                formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
                train_dataset_benign, dataset_attacker = data_allocation_attacker_noniid(client_data_idxs, \
                                                                                  attack_batch_size=attack_batch_size,
                                                                                  num_targets=self.num_targets, \
                                                                                  save_targets_path=f'./images/targets_{self.num_targets}_fl_{formatted_time}_seed_{self.data_distribution_seed}',
                                                                                  seed=self.data_distribution_seed,
                                                                                  dataset_name=self.dataset_name)
                
                train_loader_client = DataLoader(
                    train_dataset_benign,
                    batch_size=benign_batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=8)
                # sampler = RandomSampler(dataset_attacker, replacement=True, num_samples=len(dataset_attacker.targets))
                loader_attacker = DataLoader(
                    dataset_attacker,
                    batch_size=attack_batch_size,#min(attack_batch_size,len(dataset_attacker.targets)),
                    # sampler=sampler,
                    shuffle=True,
                    drop_last=True,
                    num_workers=8)

                client = AttackerClientMultiTargetNonIID(i, self.dataset_name, train_dataset_benign, train_loader_client, 
                                dataset_attacker, loader_attacker, self.use_model_poison, self.use_pgd_poison, 
                                self.use_critical_poison, self.critical_proportion,
                                self.scale_rate, self.global_pruning, self.use_adaptive, self.adaptive_lr,
                                self.device, all_num_clients=self.all_num_clients, ema_scale=self.ema_scale,
                                shared_state=self.colluders_shared_state)
            else:
                train_loader_client = DataLoader(
                    train_dataset_client,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=8)
                client = ClientNonIID(i, self.dataset_name, train_dataset_client,
                                train_loader_client, self.device)

            self.clients_set.append(client)

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
    dataset = train_data

    # 我们让每个client不同label的样本数量不同，以此做到Non-IID划分
    client_idcs = dirichlet_split_noniid(
        labels, alpha=dirichlet_alpha, n_clients=n_clients, seed=42)
    print(client_idcs)

    # 展示不同client上的label分布
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

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

    # 展示不同label划分到不同client的情况
    # plt.figure(figsize=(12, 8))
    # plt.hist([labels[idc]for idc in client_idcs], stacked=True,
    #          bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
    #          label=["Client {}".format(i) for i in range(n_clients)],
    #          rwidth=0.5)
    # plt.xticks(np.arange(n_classes), train_data.classes)
    # plt.xlabel("Label type")
    # plt.ylabel("Number of samples")
    # plt.legend(loc="upper right")
    # plt.title("Display Label Distribution on Different Clients")
    # plt.savefig('./dirichlet_42_label.png')