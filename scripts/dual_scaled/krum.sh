#!/bin/bash

#SBATCH --job-name="krum_single"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=3G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx3090:1

# Commands to execute:
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate datastealing

CUDA_VISIBLE_DEVICES=0 python fedavg_ray_actor_bd_noniid/main_fed_uncond_multitarget_defense_single_2.py \
          --train --flagfile ./config/CIFAR10_uncond.txt \
          --batch_size_attack_per 0.5 \
          --poison_type diff_poison \
          --model_poison_scale_rate 5 \
          --defense_technique krum \
          --num_targets 500 \
          --critical_proportion 0.4 \
          --global_pruning \
          --use_adaptive \
          --adaptive_lr 0.2 \
          --data_distribution_seed 42 \
          --scaled