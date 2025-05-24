import os
from glob import glob
import sys

device = sys.argv[1] #"cuda:3"
logs_name = 'cifar10_fedavg_ray_actor_att_mul_uncond_def_noniid'

num_targets = int(sys.argv[2])
defense = sys.argv[3]
full_defense_str = sys.argv[3]
defense2 = sys.argv[3].split('_')[0] # ''/'krum'...
print(num_targets)
seed = int(sys.argv[4])
if seed == 42: # All examples use seed 42
    # Category: 2 attackers colluding scaled (num_targets == 500)
    if full_defense_str == 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'foolsgold', "Foolsgold coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_145749_seed_42_foolsgold_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_145753_seed_42_foolsgold_scaled_collusion_cifar10'
    elif full_defense_str == 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'krum', "Krum coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_201805_seed_42_krum_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_201809_seed_42_krum_scaled_collusion_cifar10'
    elif full_defense_str == 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-krum', "Multi-krum coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_145749_seed_42_multi-krum_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_145753_seed_42_multi-krum_scaled_collusion_cifar10'
    elif full_defense_str == 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-metrics', "Multi-metrics coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_201841_seed_42_multi-metrics_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_201850_seed_42_multi-metrics_scaled_collusion_cifar10'
    elif full_defense_str == 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'no-defense', "No defense coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_145758_seed_42_no-defense_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_145801_seed_42_no-defense_scaled_collusion_cifar10'
    elif full_defense_str == 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_scaled_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'rfa', "RFA coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_150945_seed_42_rfa_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_150952_seed_42_rfa_scaled_collusion_cifar10'

    # Category: 2 attackers colluding no-scale (num_targets == 500)
    elif full_defense_str == 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'foolsgold', "Foolsgold coll no scaled"
        target_path = 'images/targets_500_fl_0_20250515_225235_seed_42_foolsgold_coll2_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250515_225259_seed_42_foolsgold_coll2_cifar10'
    elif full_defense_str == 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'krum', "Krum coll no scaled"
        target_path = 'images/targets_500_fl_0_20250516_214508_seed_42_krum_coll2_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250516_214547_seed_42_krum_coll2_cifar10'
    elif full_defense_str == 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-krum', "Multi-krum coll no scaled"
        target_path = 'images/targets_500_fl_0_20250516_230216_seed_42_multi-krum_coll2_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250516_230732_seed_42_multi-krum_coll2_cifar10'
    elif full_defense_str == 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-metrics', "Multi-metrics coll no scaled"
        target_path = 'images/targets_500_fl_0_20250517_001133_seed_42_multi-metrics_coll2_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250517_001137_seed_42_multi-metrics_coll2_cifar10'
    elif full_defense_str == 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'no-defense', "No defense coll no scaled"
        target_path = 'images/targets_500_fl_0_20250517_055225_seed_42_no-defense_coll2_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250517_055421_seed_42_no-defense_coll2_cifar10'
    elif full_defense_str == 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_collusion_2_lambda_0.1' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'rfa', "RFA coll no scaled"
        target_path = 'images/targets_500_fl_0_20250517_065336_seed_42_rfa_coll2_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250517_065355_seed_42_rfa_coll2_cifar10'

    # Category: 2 attackers no-collusion scaled (num_targets == 500)
    elif full_defense_str == 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_foolsgold_two_attackers_scaled' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'foolsgold', "Foolsgold 2 scaled"
        target_path = 'images/targets_500_fl_20250515_180757_client_0_seed_42_foolsgold_two_attackers_scaled_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_180758_client_1_seed_42_foolsgold_two_attackers_scaled_cifar10'
    elif full_defense_str == 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_krum_two_attackers_scaled' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'krum', "Krum 2 scaled"
        target_path = 'images/targets_500_fl_20250515_221445_client_0_seed_42_krum_two_attackers_scaled_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_221457_client_1_seed_42_krum_two_attackers_scaled_cifar10'
    elif full_defense_str == 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_multi-krum_two_attackers_scaled' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-krum', "Multi-krum 2 scaled"
        target_path = 'images/targets_500_fl_20250515_221445_client_0_seed_42_multi-krum_two_attackers_scaled_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_221457_client_1_seed_42_multi-krum_two_attackers_scaled_cifar10'
    elif full_defense_str == 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_multi-metrics_two_attackers_scaled' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-metrics', "Multi-metrics 2 scaled"
        target_path = 'images/targets_500_fl_20250515_221446_client_0_seed_42_multi-metrics_two_attackers_scaled_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_221505_client_1_seed_42_multi-metrics_two_attackers_scaled_cifar10'
    elif full_defense_str == 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_no-defense_two_attackers_scaled' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'no-defense', "No defense 2 scaled"
        target_path = 'images/targets_500_fl_20250515_221445_client_0_seed_42_no-defense_two_attackers_scaled_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_221505_client_1_seed_42_no-defense_two_attackers_scaled_cifar10'
    elif full_defense_str == 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_rfa_two_attackers_scaled' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'rfa', "RFA 2 scaled"
        target_path = 'images/targets_500_fl_20250515_221445_client_0_seed_42_rfa_two_attackers_scaled_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_221505_client_1_seed_42_rfa_two_attackers_scaled_cifar10'

    # Category: 2 attackers no-collusion no-scale (num_targets == 500) - ACTIVE IN BASH SCRIPT
    elif full_defense_str == 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_foolsgold_two_attackers' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'foolsgold', "Foolsgold 2 no scaled"
        target_path = 'images/targets_500_fl_20250515_180757_client_0_seed_42_foolsgold_two_attackers_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_180758_client_1_seed_42_foolsgold_two_attackers_cifar10'
    elif full_defense_str == 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_krum_two_attackers' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'krum', "Krum 2 no scaled"
        target_path = 'images/targets_500_fl_20250515_221251_client_0_seed_42_krum_two_attackers_cifar10'
        target_path2 = 'images/targets_500_fl_20250515_221259_client_1_seed_42_krum_two_attackers_cifar10'
    elif full_defense_str == 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_multi-krum_two_attackers' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-krum', "Multi-krum 2 no scaled"
        target_path = 'images/targets_500_fl_20250516_070812_client_0_seed_42_multi-krum_two_attackers_cifar10'
        target_path2 = 'images/targets_500_fl_20250516_070814_client_1_seed_42_multi-krum_two_attackers_cifar10'
    elif full_defense_str == 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_multi-metrics_two_attackers' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'multi-metrics', "Multi-metrics 2 no scaled"
        target_path = 'images/targets_500_fl_20250516_102400_client_0_seed_42_multi-metrics_two_attackers_cifar10'
        target_path2 = 'images/targets_500_fl_20250516_102409_client_1_seed_42_multi-metrics_two_attackers_cifar10'
    elif full_defense_str == 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_no-defense_two_attackers' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'no-defense', "No defense 2 no scaled"
        target_path = 'images/targets_500_fl_20250516_113436_client_0_seed_42_no-defense_two_attackers_cifar10'
        target_path2 = 'images/targets_500_fl_20250516_113437_client_1_seed_42_no-defense_two_attackers_cifar10'
    elif full_defense_str == 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_rfa_two_attackers' and num_targets == 500:
        # Corresponds to original: num_targets == 500, defense2 == 'rfa', "RFA 2 no scaled"
        target_path = 'images/targets_500_fl_20250516_114951_client_0_seed_42_rfa_two_attackers_cifar10'
        target_path2 = 'images/targets_500_fl_20250516_114953_client_1_seed_42_rfa_two_attackers_cifar10'

    # Category: 1 attacker (num_targets == 1000)
    elif full_defense_str == 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_foolsgold' and num_targets == 1000:
        # Corresponds to original: num_targets == 1000, defense2 == 'foolsgold'
        target_path = 'images/targets_1000_fl_20250515_173025_seed_42_foolsgold_cifar10'
        # target_path2 is not used for 1000 targets
    elif full_defense_str == 'krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_krum' and num_targets == 1000:
        # Corresponds to original: num_targets == 1000, defense2 == 'krum'
        target_path = 'images/targets_1000_fl_20250515_221509_seed_42_krum_cifar10'
    elif full_defense_str == 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_multi-krum' and num_targets == 1000:
        # Corresponds to original: num_targets == 1000, defense2 == 'multi-krum'
        target_path = 'images/targets_1000_fl_20250515_221512_seed_42_multi-krum_cifar10'
    elif full_defense_str == 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42_multi-metrics' and num_targets == 1000:
        # Corresponds to original: num_targets == 1000, defense2 == 'multi-metrics'
        target_path = 'images/targets_1000_fl_20250515_221516_seed_42_multi-metrics_cifar10'
    elif full_defense_str == 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_no-defense' and num_targets == 1000:
        # Corresponds to original: num_targets == 1000, defense2 == 'no-defense'
        target_path = 'images/targets_1000_fl_20250516_160125_seed_42_no-defense_cifar10'
    elif full_defense_str == 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42_rfa' and num_targets == 1000:
        # Corresponds to original: num_targets == 1000, defense2 == 'rfa'
        target_path = 'images/targets_1000_fl_20250516_193448_seed_42_rfa_cifar10'
    
    # Category: 2 attackers colluding scaled 2 (This category was named "collusion_2_lambda_0.1_single_dataseed_42_scaled" in bash)
    # It seems these map to the same "coll scaled" paths as the first category if the defense type matches.
    # Let's re-verify based on the original script logic and naming. The distinguishing factor here is the suffix
    # "_collusion_2_lambda_0.1_single_dataseed_42_scaled" versus "_single_dataseed_42_scaled_collusion_2_lambda_0.1"
    # Assuming these are distinct experiments that happen to use the same target images IF their simpler defense name and scenario matches.
    # If they are meant to be different target images, the original script doesn't show different paths for these specific long names.
    # For now, I'll map them to the same "coll scaled" target paths based on the defense prefix.
    # If these were intended to be *different* target image sets, you'd need to add new entries in the original script's target path logic.
    # Given the current structure, the simplest is to assume they map to the "coll scaled" paths.

    elif full_defense_str == 'foolsgold_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled' and num_targets == 500:
        # Assuming this maps to: num_targets == 500, defense2 == 'foolsgold', "Foolsgold coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_145749_seed_42_foolsgold_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_145753_seed_42_foolsgold_scaled_collusion_cifar10'
    elif full_defense_str == 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled' and num_targets == 500:
        # Assuming this maps to: num_targets == 500, defense2 == 'krum', "Krum coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_201805_seed_42_krum_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_201809_seed_42_krum_scaled_collusion_cifar10'
    elif full_defense_str == 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled' and num_targets == 500:
        # Assuming this maps to: num_targets == 500, defense2 == 'multi-krum', "Multi-krum coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_145749_seed_42_multi-krum_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_145753_seed_42_multi-krum_scaled_collusion_cifar10'
    elif full_defense_str == 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled' and num_targets == 500:
        # Assuming this maps to: num_targets == 500, defense2 == 'multi-metrics', "Multi-metrics coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_201841_seed_42_multi-metrics_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_201850_seed_42_multi-metrics_scaled_collusion_cifar10'
    elif full_defense_str == 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled' and num_targets == 500:
        # Assuming this maps to: num_targets == 500, defense2 == 'no-defense', "No defense coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_145758_seed_42_no-defense_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_145801_seed_42_no-defense_scaled_collusion_cifar10'
    elif full_defense_str == 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_collusion_2_lambda_0.1_single_dataseed_42_scaled' and num_targets == 500:
        # Assuming this maps to: num_targets == 500, defense2 == 'rfa', "RFA coll scaled"
        target_path = 'images/targets_500_fl_0_20250519_150945_seed_42_rfa_scaled_collusion_cifar10'
        target_path2 = 'images/targets_500_fl_1_20250519_150952_seed_42_rfa_scaled_collusion_cifar10'
else:
    assert(0)
# logs_name = 'cifar10_fedavg_iid_rayact_attack_mul_uncond_modp_v1'

# defense = 'multi-metrics_20_0.5_datapoi' # ''/'krum'...
# defense = 'no-defense_20_0.75' # ''/'krum'...
prefix = 'global_ckpt_round'
available_ckpts = ['300']
all_ckpts = glob(f'./logs/{logs_name}/{defense}/*.pt')
print(f'./logs/{logs_name}/{defense}/*.pt')
print(all_ckpts)

ckpt = [ckpt for ckpt in available_ckpts if any(f'{prefix}{ckpt}.pt' in path for path in all_ckpts)]
if not ckpt:
    print("No matching checkpoints found.")
    sys.exit(1)
print(f"Available checkpoints: {ckpt}")

log_file = 'res_att_mse_cifar10'
fid_outname = []

for i in ckpt:
    for j in all_ckpts:
        if prefix + str(i) + '.pt' in j:
            save_dir = f'./results_attack/{logs_name}_{defense}_{i}'
            print(save_dir)

            # First attack run
            cmd1 = (
                f'python sample_images_multi_attack_uncond_binary_mask.py '
                f'--ckpt_path {j} --save_dir {save_dir} '
                f'--device {device} --target_path {target_path}'
            )
            os.system(cmd1)

            # Second attack run, only if num_targets == 500
            if num_targets == 500:
                cmd2 = (
                    f'python sample_images_multi_attack_uncond_binary_mask.py '
                    f'--ckpt_path {j} --save_dir {save_dir} '
                    f'--device {device} --target_path {target_path2}'
                )
                os.system(cmd2)

            fid_outname.append(save_dir)
            os.system(f'echo -----------{save_dir}------------- >> {log_file}')

            # First MSE test
            cmd3 = (
                f'python test_mse_multi_targets.py '
                f'--data_dir {save_dir} --targets_dir {target_path} '
                f'>> {log_file}'
            )
            os.system(cmd3)

            # Second MSE test, only if num_targets == 500
            if num_targets == 500:
                cmd4 = (
                    f'python test_mse_multi_targets.py '
                    f'--data_dir {save_dir} --targets_dir {target_path2} '
                    f'>> {log_file}'
                )
                os.system(cmd4)

            os.system(f'echo -----------{save_dir}------------- >> {log_file}')
