import os
from glob import glob

import sys
import torch

device = sys.argv[1] #"cuda:0"
# logs_name = 'cifar10_cond_0304'
# logs_name = 'cifar10_fedavg_iid_ray_actor'
# logs_name = 'cifar10_cond_0304'
# logs_name = 'cifar10_0309'
# logs_name = 'cifar10_fedavg_iid_ray_actor_uncond_0310'
logs_name = 'cifar10_fedavg_ray_actor_att_mul_uncond_def_noniid'
# logs_name = 'cifar10_fedavg_iid_rayact_attack_mul_uncond_modp_v1'
defense = sys.argv[2] #'multi-metrics_20_0.75_datapoi' # ''/'krum'...
prefix = 'global_ckpt_round'
available_ckpts = ['300']
all_ckpts = glob(f'./logs/{logs_name}/{defense}/*.pt')
print(all_ckpts)
ckpt = [ckpt for ckpt in available_ckpts if any(f'{prefix}{ckpt}.pt' in path for path in all_ckpts)]
if not ckpt:
    print("No matching checkpoints found.")
    sys.exit(1)
print(f"Available checkpoints: {ckpt}")

log_file = 'res_cifar10'
fid_outname = []

for i in ckpt:
    j = f'./logs/{logs_name}/{defense}/{prefix}{i}.pt'
    if os.path.exists(j):
        save_dir = f'./results/{logs_name}_{defense}_{i}'
        
        # Generate samples
        cmd = (
            f'python sample_images_uncond.py '
            f'--ckpt_path {j} --save_dir {save_dir} '
            f'--device {device}'
        )
        os.system(cmd)
        fid_outname.append(save_dir)

        # Compute FID and log
        cmd_fid = (
            f'pytorch-fid stats/cifar10.train.npz {save_dir} '
            f'--device {device} >> {log_file}'
        )
        os.system(cmd_fid)
        os.system(f'echo -----------{save_dir}------------- >> {log_file}')
        os.system(f'echo ================================= >> {log_file}')
    else:
        print(j, ' not exist')
        assert(0)
