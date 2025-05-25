import copy
import json
import os
import sys
import math
from random import sample
import warnings
from absl import app, flags

import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange, tqdm

from diffusion_fed import GaussianDiffusionTrainer, GaussianDiffusionSampler
from clients_fed_single_coll import ClientsGroupMultiTargetAttackedNonIID
from model import UNet
from score.both import get_inception_and_fid_score

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from defense import *


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', None, help='dataset name')
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load ckpt.pt and evaluate FID and IS')

flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')

flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', [
                  'xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', [
                  'fixedlarge', 'fixedsmall'], help='variance type')

flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')

flags.DEFINE_integer('mid_T', 500, help='mid T split local global')
flags.DEFINE_bool('use_labels', False, help='use labels') 
flags.DEFINE_integer('num_labels', None, help='num of classes') 
flags.DEFINE_integer('local_epoch', 1, help='local epoch')
flags.DEFINE_integer('total_round', 300, help='total round')
flags.DEFINE_integer('client_num', 5, help='client num')
flags.DEFINE_integer('save_round', 100, help='save round')

flags.DEFINE_string(
    'logdir', None, help='log directory')
flags.DEFINE_integer('sample_size', 100, "sampling size of images") 
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling') 

flags.DEFINE_integer(
    'save_step', 50000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer(
    'eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000,
                     help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', None, help='FID cache') 

# scaling flags
flags.DEFINE_bool('scaled', False, help='use scaled dataset')

# collusion flags
flags.DEFINE_integer('num_colluders', 2, help='number of colluders must be ≤ client_num')
flags.DEFINE_float('lambda_reg', 0.1, help='regularization strength for colluding attacker similarity (set > 0 to enable)')


flags.DEFINE_string('defense_technique', 'no-defense', help='defense technique: no-defense/krum/multi-metrics')
flags.DEFINE_integer('part_nets_per_round', 5, help='Number of active users that are sampled per FL round to participate.')
flags.DEFINE_float('stddev', 0.158, help="choose std_dev for weak-dp defense")
flags.DEFINE_float('norm_bound', 2, help="describe norm bound of norm-clipping|weak-dp|")
flags.DEFINE_integer('num_targets', 20, help='Number of attack targets, should be 10*n for class balance.')
flags.DEFINE_float('batch_size_attack_per', 0.5, help="train with batch_size_attack_per attack samples in a batch")
flags.DEFINE_string('poison_type', 'model_poison', help='use data_poison/model_poison/pgd_poison/diff_poison/no_poison')
flags.DEFINE_bool('use_model_poison', None, help='use_model_poison[only need set poison type]')
flags.DEFINE_bool('use_pgd_poison', None, help='use_pgd_poison[only need set poison type]')
flags.DEFINE_float('multi_p', 0.6, help='percentage of included clients')
flags.DEFINE_float('model_poison_scale_rate', 5.0, help='scale of model poison')
flags.DEFINE_float('ema_scale', 0.9999, help='scale of ema')
flags.DEFINE_integer('use_critical_poison', None, help='algo: 1, algo:2, algo:3...[only need set poison type]')
flags.DEFINE_float('critical_proportion', 0.8, help='the proportion of critical layers in total layers.')
flags.DEFINE_bool('use_layer_substitution', False, help='training with layer substitution policy')
flags.DEFINE_bool('use_bclayersub_poison', False, help='training with BC layer substitution')
flags.DEFINE_bool('use_no_poison', False, help='training with no poison')

flags.DEFINE_bool('global_pruning', False, help='training with global pruning, need to modify the torch_pruning/pruner/algorithms/metapruner.py and deprecate "self.DG.check_pruning_group(group)"') 
flags.DEFINE_bool('use_adaptive', False, help='train with adaptive scale') 
flags.DEFINE_float('adaptive_lr', 0.2, help='the learning rate of adaptive scale rate.')
flags.DEFINE_integer('data_distribution_seed', 42, help='the data distribution seed.')

device = torch.device('cuda:0')

def fed_avg_aggregator(net_list, net_freq):
    sum_parameters = None
    sum_ema_parameters = None
    client_idx = [i for i in range(len(net_list))]

    for c in client_idx:
        global_parameters_tuple = net_list[c] 
        if not (isinstance(global_parameters_tuple, tuple) and len(global_parameters_tuple) == 2):
            print(f"Warning: Item in net_list for client {c} is not a tuple of two models. Skipping.")
            continue
        
        current_model, current_ema_model = global_parameters_tuple
        global_parameters_sd = current_model.state_dict()
        global_ema_parameters_sd = current_ema_model.state_dict()

        if sum_parameters is None:
            sum_parameters = {}
            for key, var in global_parameters_sd.items():
                sum_parameters[key] = var.clone() * net_freq[c]
        else:
            for var in sum_parameters:
                if var in global_parameters_sd: 
                    sum_parameters[var].add_(global_parameters_sd[var] * net_freq[c])

        if sum_ema_parameters is None:
            sum_ema_parameters = {}
            for key, var in global_ema_parameters_sd.items():
                sum_ema_parameters[key] = var.clone() * net_freq[c]
        else:
            for var in sum_ema_parameters:
                if var in global_ema_parameters_sd: 
                    sum_ema_parameters[var].add_(global_ema_parameters_sd[var] * net_freq[c])
    
    if sum_parameters is None or sum_ema_parameters is None :
        print("Warning: Aggregation resulted in None parameters. Did any clients participate?")
        return None, None

    return sum_parameters, sum_ema_parameters

def init_defender(FLAGS):
    defense_technique = FLAGS.defense_technique
    if defense_technique == "no-defense":
        _defender = None
    elif defense_technique == "norm-clipping" or defense_technique == "norm-clipping-adaptive":
        _defender = WeightDiffClippingDefense(norm_bound=FLAGS.norm_bound)
    elif defense_technique == "weak-dp":
        _defender = WeightDiffClippingDefense(norm_bound=FLAGS.norm_bound) 
    elif defense_technique == "multi-metrics":
        _defender = Multi_metrics(num_workers=FLAGS.part_nets_per_round, num_adv=2, p=FLAGS.multi_p) # num_adv is hardcoded
    elif defense_technique == "krum":
        _defender = Krum(mode='krum', num_workers=FLAGS.part_nets_per_round, num_adv=2) # num_adv is hardcoded
    elif defense_technique == "multi-krum":
        _defender = Krum(mode='multi-krum', num_workers=FLAGS.part_nets_per_round, num_adv=2) # num_adv is hardcoded
    elif defense_technique == "rfa":
        _defender = RFA()
    elif defense_technique == "foolsgold":
        _defender = FoolsGold()
    else:
        raise NotImplementedError("Unsupported defense method !")
    
    return _defender

    
def train():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    if FLAGS.dataset_name == 'cifar10':
        FLAGS.logdir = './logs/cifar10_fedavg_ray_actor_att_mul_uncond_def_noniid' 
        FLAGS.img_size = 32
    elif FLAGS.dataset_name == 'celeba':
        FLAGS.logdir = './logs/celeba_fedavg_att_mul_uncond_def_noniid'
        FLAGS.img_size = 64
    else:
        raise NotImplementedError("Dataset not implemented in main.")


    FLAGS.use_model_poison = (FLAGS.poison_type == 'model_poison')
    FLAGS.use_pgd_poison = (FLAGS.poison_type == 'pgd_poison')
    if FLAGS.poison_type == 'critical_poison': FLAGS.use_critical_poison = 1
    elif FLAGS.poison_type == 'diff_poison': FLAGS.use_critical_poison = 2
    elif FLAGS.poison_type == 'wfreeze_poison': FLAGS.use_critical_poison = 3
    else: FLAGS.use_critical_poison = 0

    FLAGS.use_bclayersub_poison = (FLAGS.poison_type == 'bclayersub')
    FLAGS.use_layer_substitution = (FLAGS.poison_type == 'bclayersub')
    FLAGS.use_no_poison = (FLAGS.poison_type == 'no_poison')


    if FLAGS.poison_type == 'model_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+
                                    str(FLAGS.num_targets)+'_'+str(FLAGS.batch_size_attack_per)+
                                    '_modpoi_'+str(FLAGS.model_poison_scale_rate)+'_ema_'+str(FLAGS.ema_scale))
    elif FLAGS.poison_type == 'pgd_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+
                                    str(FLAGS.num_targets)+'_'+str(FLAGS.batch_size_attack_per)+
                                    '_pgdpoi_8.0'+'_ema_'+str(FLAGS.ema_scale)) 
    elif FLAGS.poison_type in ['critical_poison', 'diff_poison', 'wfreeze_poison']:
        poison_prefix = {'critical_poison': 'cripoi', 'diff_poison': 'diffpoi', 'wfreeze_poison': 'wfreepoi'}[FLAGS.poison_type]
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'
                                    +str(FLAGS.batch_size_attack_per)+f'_{poison_prefix}'+f'_proportion_{FLAGS.critical_proportion}'
                                    +f'_scale_{FLAGS.model_poison_scale_rate}'+'_ema_'+str(FLAGS.ema_scale))
    elif FLAGS.poison_type == 'bclayersub':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'
                                    +str(FLAGS.batch_size_attack_per)+'_bclayersub'+f'_scale_{FLAGS.model_poison_scale_rate}'
                                    +f'_proportion_{FLAGS.critical_proportion}'+'_ema_'+str(FLAGS.ema_scale))
    elif FLAGS.poison_type == 'data_poison':
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'+
                                    str(FLAGS.batch_size_attack_per)+'_datapoi'+'_ema_'+str(FLAGS.ema_scale))
    elif FLAGS.poison_type == 'no_poison': 
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'+
                                    str(FLAGS.batch_size_attack_per)+'_nopoi'+'_ema_'+str(FLAGS.ema_scale))
    else: 
        FLAGS.logdir = os.path.join(FLAGS.logdir, FLAGS.defense_technique+'_'+str(FLAGS.num_targets)+'_'+
                                    str(FLAGS.batch_size_attack_per)+f'_{FLAGS.poison_type}'+'_ema_'+str(FLAGS.ema_scale))


    if FLAGS.defense_technique == 'multi-metrics':
        FLAGS.logdir = FLAGS.logdir + '_' + str(FLAGS.multi_p)
 

    if FLAGS.global_pruning:
        FLAGS.logdir = FLAGS.logdir + '_global'
    
    if FLAGS.use_adaptive:
        if FLAGS.poison_type == 'diff_poison': 
            FLAGS.logdir = FLAGS.logdir + f'_adaptive_{FLAGS.adaptive_lr}'


    if FLAGS.num_colluders is not None: 
        FLAGS.logdir = FLAGS.logdir + f'_collusion_{FLAGS.num_colluders}_lambda_{FLAGS.lambda_reg}'

    if FLAGS.scaled and FLAGS.num_targets == 500: 
        FLAGS.logdir = FLAGS.logdir + f'_single_dataseed_{FLAGS.data_distribution_seed}_scaled'
    else:
        FLAGS.logdir = FLAGS.logdir + f'_single_dataseed_{FLAGS.data_distribution_seed}'


    print('poison_type: ', FLAGS.poison_type)
    print('use_model_poison: ', FLAGS.use_model_poison)
    print('model_poison_scale_rate: ', FLAGS.model_poison_scale_rate)
    print('use_pgd_poison: ', FLAGS.use_pgd_poison)
    print('use_critical_poison (algo type): ', FLAGS.use_critical_poison)
    print('use_layer_substitution: ', FLAGS.use_layer_substitution) 
    print('use_bclayersub_poison: ', FLAGS.use_bclayersub_poison)
    print('global_pruning: ', FLAGS.global_pruning)
    print('use_adaptive: ', FLAGS.use_adaptive, FLAGS.adaptive_lr if FLAGS.use_adaptive else "")
    print('logdir: ', FLAGS.logdir)
    
    # Load global checkpoint
    global_ckpt_path = None
    if FLAGS.dataset_name == 'cifar10':
        global_ckpt_path = './logs/cifar10_fedavg_uncond_noniid_0325/global_ckpt_round2000.pt'
    elif FLAGS.dataset_name == 'celeba':
        global_ckpt_path = './logs/celeba_fedavg_uncond_noniid_0423/global_ckpt_round1200.pt'
    
    global_ckpt = None
    if global_ckpt_path and os.path.exists(global_ckpt_path):
        global_ckpt = torch.load(global_ckpt_path, map_location=torch.device('cpu'))
        print(f"Loaded global checkpoint from {global_ckpt_path}")
    else:
        print(f"Warning: Global checkpoint not found at {global_ckpt_path}. Models will initialize from scratch if no other ckpt logic.")


    # model setup
    net_model_global = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels)
    
    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    x_T_sample = torch.randn(FLAGS.sample_size if FLAGS.num_labels is None else FLAGS.num_labels, 
                             3, FLAGS.img_size, FLAGS.img_size) 
    x_T_sample = x_T_sample.to(device) 
    
    writer = SummaryWriter(FLAGS.logdir)
    
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())

    # Initialize Client Group
    clients_group = ClientsGroupMultiTargetAttackedNonIID(
        FLAGS.dataset_name, FLAGS.batch_size, FLAGS.client_num, FLAGS.num_targets, device, 
        batch_size_attack_per=FLAGS.batch_size_attack_per, 
        scale_rate=FLAGS.model_poison_scale_rate, 
        use_model_poison=FLAGS.use_model_poison, 
        use_pgd_poison=FLAGS.use_pgd_poison, 
        use_critical_poison=FLAGS.use_critical_poison, 
        critical_proportion=FLAGS.critical_proportion, 
        use_bclayersub_poison=FLAGS.use_bclayersub_poison,
        use_layer_substitution=FLAGS.use_layer_substitution, 
        use_no_poison=FLAGS.use_no_poison, 
        global_pruning=FLAGS.global_pruning, 
        use_adaptive=FLAGS.use_adaptive,
        adaptive_lr=FLAGS.adaptive_lr, 
        all_num_clients=FLAGS.client_num, # Pass total clients
        ema_scale=FLAGS.ema_scale, 
        data_distribution_seed=FLAGS.data_distribution_seed,
        num_colluders=FLAGS.num_colluders, # Pass collusion flags
        lambda_reg=FLAGS.lambda_reg,      # Pass collusion flags
        defense_technique=FLAGS.defense_technique,
        scaled=FLAGS.scaled, # Pass scaled flag
    )
    
    # init local_parameters for each client
    for i in range(FLAGS.client_num):
        # Pass the actual net_model_global object, not a deepcopy yet, init handles copy
        clients_group.clients_set[i].init(net_model_global, FLAGS.lr, FLAGS.parallel, global_ckpt=global_ckpt)

    # For norm-clipping/weak-dp, server keeps a copy of the global model for diff calculation
    net_avg_server_copy = None # Store the (model, ema_model) tuple for defense
    if FLAGS.defense_technique == 'weak-dp' or FLAGS.defense_technique == 'norm-clipping':
        # These models represent the state *before* client updates for this round
        server_model_for_diff = copy.deepcopy(net_model_global)
        server_ema_model_for_diff = copy.deepcopy(net_model_global) # Start EMA same as model
        if global_ckpt: # Load from checkpoint if available
            server_model_for_diff.load_state_dict(global_ckpt['global_model'], strict=True)
            server_ema_model_for_diff.load_state_dict(global_ckpt['global_ema_model'], strict=True)
        net_avg_server_copy = (server_model_for_diff.to(device), server_ema_model_for_diff.to(device))


    client_idx_all = list(range(FLAGS.client_num)) # Renamed client_idx
    
    clients_targets_num = [] # Renamed clients_targets
    for c_idx in client_idx_all: # Renamed c
        clients_targets_num.append(clients_group.clients_set[c_idx].get_targets_num())
    
    train_data_sum = sum(clients_targets_num)
    if train_data_sum == 0: train_data_sum = 1 # Avoid division by zero if no data

    # g_user_indices = client_idx_all # In this version, all clients participate each round

    _defender = init_defender(FLAGS)
    norm_diff_collector_adaptive = [] # For norm-clipping-adaptive, if used

    # start training
    for round_idx in tqdm(range(0, FLAGS.total_round), desc="Federated Rounds"): # Renamed round
        net_freq = [num_data/train_data_sum for num_data in clients_targets_num]
        
        net_list_round = [] # Renamed net_list
        # Participating clients for this round (all clients in this setup)
        current_round_participants_indices = client_idx_all 
        
        for c_idx_participant in current_round_participants_indices: # Renamed c
            # Client's local_train returns (model_object, ema_model_object)
            client_model_tuple = clients_group.clients_set[c_idx_participant].local_train( 
                                                        round_idx, 
                                                        FLAGS.local_epoch, 
                                                        mid_T=FLAGS.mid_T, 
                                                        use_labels=FLAGS.use_labels)
            net_list_round.append(client_model_tuple)


        # Apply defense mechanisms
        # Defenses might modify net_list_round or net_freq
        if FLAGS.defense_technique == "no-defense":
            pass
        elif FLAGS.defense_technique == "norm-clipping":
            if net_avg_server_copy is None: # Should have been initialized
                print("Error: net_avg_server_copy not initialized for norm-clipping. Skipping defense.")
            else:
                for net_idx, client_model_tuple_def in enumerate(net_list_round):
                    # Defender expects client_model=(model, ema_model), global_model=(model_server, ema_model_server)
                    _defender.exec(client_model=client_model_tuple_def, global_model=net_avg_server_copy)
        elif FLAGS.defense_technique == "norm-clipping-adaptive":
            if not norm_diff_collector_adaptive: # First round or no diffs collected
                 _defender.norm_bound = FLAGS.norm_bound # Use initial norm_bound
            else:
                 _defender.norm_bound = np.mean(norm_diff_collector_adaptive) if norm_diff_collector_adaptive else FLAGS.norm_bound
            # print(f"#### Norm Diff Collector : {norm_diff_collector_adaptive}; Mean: {np.mean(norm_diff_collector_adaptive) if norm_diff_collector_adaptive else 'N/A'}")
            
            current_norm_diffs_this_round = [] # Collect diffs for *next* round's adaptive bound
            if net_avg_server_copy is None:
                print("Error: net_avg_server_copy not initialized for norm-clipping-adaptive. Skipping defense.")
            else:
                for net_idx, client_model_tuple_def_adapt in enumerate(net_list_round):
                     _defender.exec(client_model=client_model_tuple_def_adapt, global_model=net_avg_server_copy)

        elif FLAGS.defense_technique == "weak-dp": # Clipping part
            if net_avg_server_copy is None:
                print("Error: net_avg_server_copy not initialized for weak-dp clipping. Skipping defense.")
            else:
                for net_idx, client_model_tuple_def_wdp in enumerate(net_list_round):
                    _defender.exec(client_model=client_model_tuple_def_wdp, global_model=net_avg_server_copy)
            # Noise addition happens *after* aggregation for weak-dp

        elif FLAGS.defense_technique in ["multi-metrics", "krum", "multi-krum", "rfa", "foolsgold"]:
            if FLAGS.defense_technique == "krum": # Single Krum
                net_list_round, net_freq, krum_epoch_choice, krum_global_choice = _defender.exec(
                                                        client_models=net_list_round,
                                                        num_dps=clients_targets_num,
                                                        g_user_indices=current_round_participants_indices,
                                                        device=device)
                print(f"Krum selected client index (local to this round's participants): {krum_epoch_choice}")
                print(f"Krum selected client index (global): {krum_global_choice}")

            elif FLAGS.defense_technique == "multi-krum": # Multi Krum
                net_list_round, net_freq = _defender.exec( # <<< ONLY UNPACK 2 VALUES
                                                        client_models=net_list_round,
                                                        num_dps=clients_targets_num,
                                                        g_user_indices=current_round_participants_indices,
                                                        device=device)
            elif FLAGS.defense_technique == "multi-metrics":
                 net_list_round, net_freq = _defender.exec(
                                                        client_models=net_list_round,
                                                        num_dps=clients_targets_num,
                                                        g_user_indices=current_round_participants_indices,
                                                        device=device)
            elif FLAGS.defense_technique == "rfa":
                 net_list_round, net_freq = _defender.exec(
                                                        client_models=net_list_round, net_freq=net_freq,
                                                        device=device) # RFA might have different params
            elif FLAGS.defense_technique == "foolsgold":
                 net_list_round, net_freq = _defender.exec(
                                                        client_models=net_list_round, net_freq=net_freq,
                                                        names=current_round_participants_indices, device=device)
        else:
            # This 'else' implies a defense was specified but not handled above.
            # init_defender should raise error for unsupported, so this might not be reached.
            pass 
        
        sum_params_sd, sum_ema_params_sd = fed_avg_aggregator(net_list_round, net_freq)

        if sum_params_sd is None or sum_ema_params_sd is None:
            print(f"Round {round_idx}: Aggregation failed, no valid models to aggregate. Skipping update.")
            # Potentially load previous round's model to continue or handle error
            if global_ckpt: # Fallback to last saved checkpoint
                sum_params_sd = global_ckpt['global_model']
                sum_ema_params_sd = global_ckpt['global_ema_model']
            else: # Critical error if no fallback
                raise ValueError("Aggregation failed and no checkpoint to fallback to.")


        # Update server's model state for norm-clipping/weak-dp for *next* round's diff
        if FLAGS.defense_technique == 'norm-clipping' or FLAGS.defense_technique == "weak-dp" or FLAGS.defense_technique == "norm-clipping-adaptive":
            if net_avg_server_copy is not None:
                net_avg_server_copy[0].load_state_dict(copy.deepcopy(sum_params_sd), strict=True)
                net_avg_server_copy[1].load_state_dict(copy.deepcopy(sum_ema_params_sd), strict=True)
        
        # For weak-dp, add noise AFTER aggregation and potential clipping defense updates
        if FLAGS.defense_technique == "weak-dp":
            # Create a temporary model tuple from aggregated state dicts to add noise
            temp_agg_model = UNet(T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                                  num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels).to(device)
            temp_agg_ema_model = UNet(T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
                                      num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels).to(device)
            
            temp_agg_model.load_state_dict(sum_params_sd)
            temp_agg_ema_model.load_state_dict(sum_ema_params_sd)
            
            noise_adder = AddNoise(stddev=FLAGS.stddev) # AddNoise needs to be defined from defense.py
            # AddNoise.exec should modify the models in-place
            noise_adder.exec(client_model=(temp_agg_model, temp_agg_ema_model), device=device) 
            
            # Get the state dicts back after noise addition
            sum_params_sd = temp_agg_model.state_dict()
            sum_ema_params_sd = temp_agg_ema_model.state_dict()


        # Distribute aggregated global model (state dicts) to all clients
        for c_idx_distribute in client_idx_all: # Renamed c
            clients_group.clients_set[c_idx_distribute].set_global_parameters(sum_params_sd, sum_ema_params_sd)

        # Sampling and Logging
        if (round_idx + 1) % FLAGS.sample_step == 0 or round_idx == FLAGS.total_round -1 :
            samples_list = []
            if len(clients_group.clients_set) > 0 and clients_group.clients_set[0].global_ema_model is not None:
                client_for_sampling = clients_group.clients_set[0] # Sample from client 0
                with torch.no_grad():
                    if FLAGS.num_labels is None: # Unconditional
                        x_0_sampled = client_for_sampling.get_sample(x_T_sample, 0, FLAGS.T-1) # Correct end_step
                    else: # Conditional
                        labels_for_sample = torch.arange(FLAGS.num_labels, device=device).repeat_interleave(FLAGS.sample_size // FLAGS.num_labels if FLAGS.sample_size >= FLAGS.num_labels else 1)
                        # Adjust x_T_sample shape for conditional if needed, or sample one by one
                        x_T_cond_sample = torch.randn(labels_for_sample.size(0), 3, FLAGS.img_size, FLAGS.img_size, device=device)
                        x_0_sampled = client_for_sampling.get_sample(x_T_cond_sample, 0, FLAGS.T-1, labels_for_sample)
                    samples_list.append(x_0_sampled.cpu())
            
                if samples_list:
                    samples_tensor = torch.cat(samples_list, dim=0)
                    grid = (make_grid(samples_tensor, nrow= int(math.sqrt(samples_tensor.size(0))) ) + 1) / 2 # Adjust nrow
                    path = os.path.join(FLAGS.logdir, 'sample', f'{round_idx+1}.png')
                    save_image(grid, path)
                    writer.add_image('sample/global_model_sample', grid, round_idx+1)
                else:
                    print(f"Round {round_idx+1}: No samples generated (client 0 might not be ready or conditional sampling issue).")
            else:
                print(f"Round {round_idx+1}: Client 0 not available for sampling.")


        # Save global model checkpoint
        if FLAGS.save_round > 0 and (round_idx+1) % FLAGS.save_round == 0:
            global_ckpt_save = { # Renamed global_ckpt
                'global_model': sum_params_sd,
                'global_ema_model': sum_ema_params_sd,
                'round': round_idx + 1
            }
            # global_model_path = FLAGS.logdir # Path is already FLAGS.logdir
            torch.save(global_ckpt_save, os.path.join(
                FLAGS.logdir, f'global_ckpt_round{round_idx+1}.pt'))
            # Update global_ckpt for fallback if aggregation fails next round
            global_ckpt = global_ckpt_save 


    writer.close()


def eval_main(): # Renamed eval to avoid conflict with built-in eval
    # Simplified eval function, assuming FID/IS calculation based on a saved checkpoint
    # This part needs to be adapted from your original full eval logic if it was more complex
    print("Evaluation function (eval_main) called. Implement your FID/IS calculation here.")
    if not FLAGS.logdir or not os.path.exists(os.path.join(FLAGS.logdir, f'global_ckpt_round{FLAGS.total_round}.pt')):
        print(f"No checkpoint found at {os.path.join(FLAGS.logdir, f'global_ckpt_round{FLAGS.total_round}.pt')} for evaluation.")
        return

    # model setup
    eval_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, num_labels=FLAGS.num_labels
    ).to(device)
    
    # Load the EMA model from the checkpoint for evaluation
    ckpt_to_eval = torch.load(os.path.join(FLAGS.logdir, f'global_ckpt_round{FLAGS.total_round}.pt'), map_location=device)
    eval_model.load_state_dict(ckpt_to_eval['global_ema_model'])
    
    sampler_eval = GaussianDiffusionSampler(
        eval_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size, 
        FLAGS.mean_type, FLAGS.var_type
    ).to(device)

    # Generate images
    # (Your image generation loop for FID/IS here)
    # For example:
    # all_images = []
    # for i in trange(0, FLAGS.num_images, FLAGS.batch_size, desc="Generating images for FID/IS"):
    #     x_T_eval = torch.randn(FLAGS.batch_size, 3, FLAGS.img_size, FLAGS.img_size).to(device)
    #     generated_images = sampler_eval(x_T_eval, 0, FLAGS.T-1) # unconditional
    #     all_images.append(generated_images.cpu())
    # all_images = torch.cat(all_images, dim=0)
    
    # Calculate FID/IS
    # (IS, FID) = get_inception_and_fid_score(all_images, device, FLAGS.fid_use_torch, FLAGS.fid_cache)
    # print(f"Inception Score: {IS}, FID: {FID}")
    # writer_eval = SummaryWriter(os.path.join(FLAGS.logdir, 'eval_results'))
    # writer_eval.add_scalar('IS', IS, 0)
    # writer_eval.add_scalar('FID', FID, 0)
    # writer_eval.close()


def main(argv):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train()
    if FLAGS.eval:
        eval_main() # Call renamed eval
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)