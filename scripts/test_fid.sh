#!/bin/bash

#SBATCH --job-name="testing4"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=rtx4090:1

# Commands to execute:
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate datastealing

# 2 attackers colluding scaled

python bash_test_fid_multi_defense.py "cuda:0" 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1'

# 2 attackers colluding no-scale

python bash_test_fid_multi_defense.py "cuda:0" 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1'
python bash_test_fid_multi_defense.py "cuda:0" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1'

# 2 attackers no-collusion scaled

python bash_test_fid_multi_defense.py "cuda:0" 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_foolsgold_two_attackers_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_krum_two_attackers_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_multi-krum_two_attackers_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_multi-metrics_two_attackers_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_no-defense_two_attackers_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_rfa_two_attackers_scaled'

# 2 attackers no-collusion no-scale

python bash_test_fid_multi_defense.py "cuda:0" 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_foolsgold_two_attackers'
python bash_test_fid_multi_defense.py "cuda:0" 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_krum_two_attackers'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_multi-krum_two_attackers'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_multi-metrics_two_attackers'
python bash_test_fid_multi_defense.py "cuda:0" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_no-defense_two_attackers'
python bash_test_fid_multi_defense.py "cuda:0" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_rfa_two_attackers'

# 1 attacker

python bash_test_fid_multi_defense.py "cuda:0" 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_foolsgold'
python bash_test_fid_multi_defense.py "cuda:0" 'krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_krum'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_multi-krum'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_multi-metrics'
python bash_test_fid_multi_defense.py "cuda:0" 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_no-defense'
python bash_test_fid_multi_defense.py "cuda:0" 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_rfa'

# 2 attackers colluding scaled 2

python bash_test_fid_multi_defense.py "cuda:0" 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled'
python bash_test_fid_multi_defense.py "cuda:0" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled'