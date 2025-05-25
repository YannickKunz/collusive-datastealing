# This script runs C-ADASCP with different configurations for CIFAR10 dataset.
# To change the number of malicious clients, modify inside clients_fed_coordinated the number of colluding clients.

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_coordinated.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique no-defense \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_coordinated.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique krum \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_coordinated.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique multi-krum \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_coordinated.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique multi-metrics \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_coordinated.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique rfa \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_coordinated.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique foolsgold \
          --num_targets 1000 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42
