# This script test the FID and MSE for the malicious clients.

# Test FID for different defense methods
# No Defense
#python bash_test_fid_multi_defense_nd.py "cuda:0" 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_nd.py "cuda:0" 'no-defense_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_nd.py "cuda:0" 'no-defense_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_nd.py "cuda:0" 'no-defense_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_nd.py "cuda:0" 'no-defense_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'

# Krum
#python bash_test_fid_multi_defense_k.py "cuda:0" 'krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_k.py "cuda:0" 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_k.py "cuda:0" 'krum_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_k.py "cuda:0" 'krum_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_k.py "cuda:0" 'krum_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'

# Krum with 0.7 scale
#python bash_test_fid_multi_defense_k07.py "cuda:0" 'krum_500_0.7_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'

# Multi-Krum
#python bash_test_fid_multi_defense_mk.py "cuda:0" 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mk.py "cuda:0" 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mk.py "cuda:0" 'multi-krum_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mk.py "cuda:0" 'multi-krum_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mk.py "cuda:0" 'multi-krum_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'

# Multi-Metrics
#python bash_test_fid_multi_defense_mm.py "cuda:0" 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mm.py "cuda:0" 'multi-metrics_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mm.py "cuda:0" 'multi-metrics_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mm.py "cuda:0" 'multi-metrics_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_mm.py "cuda:0" 'multi-metrics_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42'

# RFA
#python bash_test_fid_multi_defense_rfa.py "cuda:0" 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_rfa.py "cuda:0" 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_rfa.py "cuda:0" 'rfa_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_rfa.py "cuda:0" 'rfa_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'
#python bash_test_fid_multi_defense_rfa.py "cuda:0" 'rfa_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'

# Fool's Gold
#python bash_test_fid_multi_defense_fg.py "cuda:0" 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'





# Test MSE for different defense methods
# No Defense
#python bash_test_diffusion_attack_uncond_multi_mask_seed_nodef.py "cuda:0" 1000 'no-defense_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_nodef.py "cuda:0" 333 'no-defense_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_nodef.py "cuda:0" 250 'no-defense_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_nodef.py "cuda:0" 200 'no-defense_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42

# Krum
#python bash_test_diffusion_attack_uncond_multi_mask_seed_krum.py "cuda:0" 1000 'krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_krum.py "cuda:0" 500 'krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_krum.py "cuda:0" 333 'krum_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_krum.py "cuda:0" 250 'krum_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_krum.py "cuda:0" 200 'krum_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42

# Krum with 0.7 scale
#python bash_test_diffusion_attack_uncond_multi_mask_seed_krum.py "cuda:0" 500 'krum_500_0.7_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_fid_multi_defense_k07b.py "cuda:0" 'krum_500_0.7_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42'

# Multi-Krum
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mkrum.py "cuda:0" 1000 'multi-krum_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mkrum.py "cuda:0" 500 'multi-krum_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mkrum.py "cuda:0" 333 'multi-krum_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mkrum.py "cuda:0" 250 'multi-krum_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mkrum.py "cuda:0" 200 'multi-krum_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42

# Multi-Metrics
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mm.py "cuda:0" 1000 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mm.py "cuda:0" 500 'multi-metrics_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mm.py "cuda:0" 333 'multi-metrics_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mm.py "cuda:0" 250 'multi-metrics_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_mm.py "cuda:0" 200 'multi-metrics_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42

# RFA
#python bash_test_diffusion_attack_uncond_multi_mask_seed_rfa.py "cuda:0" 1000 'rfa_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_rfa.py "cuda:0" 500 'rfa_500_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_rfa.py "cuda:0" 333 'rfa_333_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_rfa.py "cuda:0" 250 'rfa_250_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_rfa.py "cuda:0" 200 'rfa_200_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42

# Fool's Gold
#python bash_test_diffusion_attack_uncond_multi_mask_seed_fg.py "cuda:0" 1000 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_0.6_global_adaptive_0.2_single_dataseed_42' 42
#python bash_test_diffusion_attack_uncond_multi_mask_seed_fg.py "cuda:0" 1000 'foolsgold_1000_0.5_diffpoi_proportion_0.4_scale_5.0_ema_0.9999_global_adaptive_0.2_single_dataseed_42' 42

