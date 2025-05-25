# Collusive DataStealing: Enhancing Data Exfiltration from Federated Diffusion Models

This repository contains the code and experimental setup for investigating N-party collusive attacks that extend the "DataStealing" framework [1] for exfiltrating private data from diffusion models in Federated Learning (FL). We introduce and evaluate two primary collusive strategies: one based on a **Model Similarity Objective** and another termed **Coordinated AdaSCP (C-AdaSCP)** leveraging Distributed Indicator Probing.

Our work demonstrates that coordinated adversaries can significantly amplify data exfiltration success and improve stealth against common FL defenses compared to single-attacker baselines.

## Table of Contents

- [Collusive DataStealing: Enhancing Data Exfiltration from Federated Diffusion Models](#collusive-datastealing-enhancing-data-exfiltration-from-federated-diffusion-models)
  - [Table of Contents](#table-of-contents)
  - [Original DataStealing Framework](#original-datastealing-framework)
  - [Our Proposed Collusive Extensions](#our-proposed-collusive-extensions)
    - [1. Collusion via Model Similarity](#1-collusion-via-model-similarity)
    - [2. Coordinated AdaSCP (C-AdaSCP)](#2-coordinated-adascp-c-adascp)
  - [Repository Structure](#repository-structure)
  - [Setup and Dependencies](#setup-and-dependencies)
  - [Dataset Preparation](#dataset-preparation)
  - [Running Experiments](#running-experiments)
    - [Training Baseline and Collusive Models](#training-baseline-and-collusive-models)
    - [Evaluating Attack Success (MSE)](#evaluating-attack-success-mse)
    - [Evaluating Model Utility (FID)](#evaluating-model-utility-fid)
  - [Results Summary Table](#results-summary-table)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)

## Original DataStealing Framework

This work builds upon the "DataStealing: Steal Data from Diffusion Models in Federated Learning with Multiple Trojans" paper by Gan et al. [1]. We reimplement and extend their core single-attacker mechanisms, including:
*   **Combinatorial Triggers (ComboTs):** For mapping multiple target images to distinct backdoor triggers.
*   **Adaptive Scale Critical Parameters (AdaSCP):** For identifying critical model parameters and adaptively scaling malicious updates to bypass defenses, utilizing an indicator mechanism.

For full details on the original framework, please refer to their paper and repository:
*   **Original Paper:** [DataStealing: Steal Data from Diffusion Models in Federated Learning with Multiple Trojans](https://openreview.net/forum?id=792txRlKit&referrer=%5Bthe%20profile%20of%20Jiaxu%20Miao%5D(%2Fprofile%3Fid%3D~Jiaxu_Miao2))
*   **Original Code:** [https://github.com/yuangan/DataStealing](https://github.com/yuangan/DataStealing)

## Our Proposed Collusive Extensions

We introduce two novel N-party collusive strategies designed to enhance the efficacy of the DataStealing attack:

### 1. Collusion via Model Similarity

This strategy involves `N` colluding attackers who, in addition to standard DataStealing techniques, incorporate a model similarity regularization term (`L_sim = λ_reg * || M_i - M_avg_colluders ||²_2`) into their local training objective. This encourages their individual malicious models to converge, aiming for more consistent and potent aggregated malicious updates. This approach is primarily implemented in:
*   `clients_fed_single_coll.py`
*   `main_fed_single_coll.py`

### 2. Coordinated AdaSCP (C-AdaSCP)

C-AdaSCP modifies the original AdaSCP for a collusive setting. Key aspects include:
*   **Distributed Indicator Probing:** Each of the `N` colluding clients independently identifies and implants its own indicator parameter.
*   **Shared Feedback Aggregation:** Colluders share feedback derived from their individual probes to collectively compute a more robust target scaling factor (`tars_collective`).
*   **Synchronized Scale Optimization:** All colluders use this `tars_collective` to update their individual malicious update scale factors.
This approach is primarily implemented in:
*   `clients_fed_coordinated.py`
*   `main_fed_coordinated.py`

## Repository Structure

```
collusive-datastealing/
├── config/                                              # CIFAR10_uncond.txt
├── data/                                                # CIFAR-10 dataset
├── fedavg_ray_actor_bd_noniid/
│   ├── clients_fed_single_1000.py                       # Client logic for single attacker (baseline)
│   ├── clients_fed_single_2.py                          # Client logic for uncoordinated 2 attackers (if distinct from single)
│   ├── clients_fed_single_coll.py                       # Client logic for Model Similarity Collusion
│   ├── clients_fed_coordinated.py                       # Client logic for Coordinated AdaSCP (C-AdaSCP)
│   ├── main_fed_single_1000.py                          # Main script for single attacker
│   ├── main_fed_single_2.py                             # Main script for uncoordinated 2 attackers
│   ├── main_fed_single_coll.py                          # Main script for Model Similarity Collusion
│   ├── main_fed_coordinated.py                          # Main script for Coordinated AdaSCP
│   ├── diffusion_fed.py                                 # Core diffusion and federated training utilities
├── images/                                              # Images folders will be generated here during training
├── logs/
│   ├── cifar10_fedavg_ray_att_mul_uncond_def_noniid/    # Your train models will be saved here
│   ├── cifar10_fedavg_uncond_noniid_0325/               # Download the pretrained FL models with 5 clients from original paper and put it here
├── results/                                             # For storing FID scores from training runs
├── results_attack/                                      # For storing MSE scores and reconstructed images
├── score/                                               # Scripts for FID/IS calculation
├── scripts/                                             # Shell scripts for running experiments
│   ├── collusion_noscale/                               # Training scripts for Collusion, 2 attackers without scaling
│   ├── collusion_scaled/                                # Training scripts for Collusion, 2 attackers with scaling
│   ├── dual_noscale/                                    # Training scripts for No-Collusion, 2 attackers without scaling
│   ├── dual_scaled/                                     # Training scripts for No-Collusion, 2 attackers with scaling
│   ├── single_no_scaling/                               # Training scripts for 1 attacker
│   ├── collaborative_adascp/                            # Training scripts for Collaborative AdaSCP attack
│   ├── test_fid.sh                                      # Combined FID evaluation script
│   ├── test_mses.sh                                     # Combined MSE evaluation script
├── stats/                                               # CIFAR-10 train images as npz
├── tmp/                                             # C-ADASCP scale rate logs
│   ├── CADASCP3/...                                     # C-ADASCP scale rate logs
│   ├── attacker.log                                     # C-ADASCP scale rate logs
│   ├── attacker_foolsgold.log                           # C-ADASCP scale rate logs
│   ├── attacker_krum.log                                # C-ADASCP scale rate logs
│   ├── attacker_multi-krum.log                          # C-ADASCP scale rate logs
│   ├── attacker_multi-metrics.log                       # C-ADASCP scale rate logs
│   ├── attacker_no-defense.log                          # C-ADASCP scale rate logs
│   ├── attacker_rfa.log                                 # C-ADASCP scale rate logs
├── attackerDataset.py                                   # Attacker dataset prep
├── bash_test_diffusion_attack_uncond_multi_mask_seed.py # MSE evaluation file, need to update this file based on your logs/ and images/ folders
├── bash_test_fid_multi_defense.py                       # FID evaluation file
├── defense.py                                           # Defense mechanism implementations
├── diffusion                                            # Gaussian diffusion
├── geometric_median.py                                  # Geometric median calculation for evaluation
├── LICENSE                                              # MIT License
├── model.py                                             # UNet model definition
├── README.md                                            # This file
├── res_att_mse_cifar10_results                          # Compiled MSE results from our paper
├── res_cifar10_results                                  # Compiled FID results from our paper
├── sample_images_multi_attack_uncond_binary_mask.py     # Add trigger for AdaSCP
├── sample_images_uncond.py                              # Add trigger for AdaSCP
├── test_mse_multi_targets.py                            # Testing MSE
├── visualize.py                                         # Image visualizer
```

## Setup and Dependencies

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YannickKunz/collusive-datastealing.git
    cd collusive-datastealing
    ```
2.  **Create a Conda environment (recommended):**
    ```bash
    conda create -n collusive_ds python=3.9
    conda activate collusive_ds
    ```
3.  **Install dependencies:**
    ```bash
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
    pip install -r requirements.txt
    ```
4.  **Data preparation:**
    ```bash
    cd ./data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xf cifar-10-python.tar.gz
    ```

## Dataset Preparation

This project uses the CIFAR-10 dataset.
*   **CIFAR-10:** The dataset will be automatically downloaded by torchvision the first time you run a script that requires it. Ensure you have an internet connection. Data is typically saved in a `./data/` directory.
*   **Target Images for `D_backdoor`:** The scripts are designed to select target images from the dataset based on the `data_distribution_seed` and client data splits for reproducibility. If you used a fixed set of pre-selected target images for all experiments, please specify their location or provide them.

## Running Experiments

All commands for training and evaluation used in our paper can be found in the `scripts/` directory. Please adapt paths, GPU IDs, and other parameters as necessary for your environment.

### Training Baseline and Collusive Models

Refer to the example commands in `scripts/`. Key flags to configure include:

*   `--dataset_name cifar10`
*   `--train`
*   `--poison_type diff_poison` (for AdaSCP-based attacks)
*   `--defense_technique [no-defense|krum|multi-krum|foolsgold|rfa|multi-metrics]`
*   `--num_targets` (e.g., 1000 for single attacker, 500 per attacker for 2 attackers)
*   `--batch_size_attack_per` (e.g., 0.5)
*   `--model_poison_scale_rate` (e.g., 5.0, effectively `s` in AdaSCP)
*   `--critical_proportion` (e.g., 0.4)
*   `--adaptive_lr` (e.g., 0.2 for AdaSCP's scale optimization)
*   `--data_distribution_seed` (e.g., 42)
*   `--logdir` (specify unique path for each experiment run)
*   **For Collusive Attacks:**
    *   `--num_colluders [N]` (e.g., 2)
    *   `--lambda_reg [value]` (e.g., 0.1, for Model Similarity Collusion)
    *   `--scaled` (Add this flag for scaled multi-attacker scenarios)


### Evaluating Attack Success (MSE)

Use `bash_test_diffusion_attack_uncond_multi_mask_seed.sh` to evaluate MSE.  
Check the scripts folder to see usage.  
Ensure the script correctly handles paths to the checkpoint and outputs MSE scores.  

### Evaluating Model Utility (FID)

Use `bash_test_fid_multi_defense.sh` to calculate FID.  
Check the scripts folder to see usage. 

## Results Summary Table

Our experiments on CIFAR-10 demonstrate the heightened threat of collusive attacks:

| Attack Scenario                               | Defense         | MSE (Attacker 1)      | MSE (Attacker 2)      | MSE (Average)      | FID          |
| :-------------------------------------------- | :-------------- | :-------------------- | :-------------------- | :----------------- | :----------- |
| **1 Attacker (Baseline AdaSCP)**              | No-Defense      | 0.0133                | N/A                   | 0.0133             | 13.37        |
|                                               | Krum            | 0.0506                | N/A                   | 0.0506             | 33.94        |
|                                               | Multi-Krum      | 0.1233                | N/A                   | 0.1233             | 8.02         |
|                                               | Foolsgold       | 0.0208                | N/A                   | 0.0208             | 21.39        |
|                                               | RFA             | 0.1255                | N/A                   | 0.1255             | 8.34         |
|                                               | Multi-Metrics   | 0.0684                | N/A                   | 0.0684             | 11.84        |
|                                               | **Average**     | **0.0670**            | **N/A**               | **0.0670**         | **16.15**    |
| **2 Attackers (No Collusion, No Scale)**      | No-Defense      | 0.0638                | 0.0728                | 0.0683             | 102.41       |
|                                               | Krum            | 0.1466                | 0.1524                | 0.1495             | 41.55        |
|                                               | Multi-Krum      | 0.1025                | 0.1026                | 0.1026             | 12.17        |
|                                               | Foolsgold       | 0.0657                | 0.0757                | 0.0707             | 251.66       |
|                                               | RFA             | 0.1131                | 0.1236                | 0.1184             | 9.89         |
|                                               | Multi-Metrics   | 0.1097                | 0.1183                | 0.1140             | 16.03        |
|                                               | **Average**     | **0.1002**            | **0.1076**            | **0.1039**         | **72.29**    |
| **2 Attackers (No Collusion, Scaled)**        | No-Defense      | 0.2507                | 0.2633                | 0.2570             | 10.93        |
|                                               | Krum            | 0.0323                | 0.1183                | 0.0753             | 32.83        |
|                                               | Multi-Krum      | 0.1133                | 0.1178                | 0.1156             | 12.55        |
|                                               | Foolsgold       | 0.2013                | 0.2087                | 0.2050             | 13.31        |
|                                               | RFA             | 0.1099                | 0.1209                | 0.1154             | 9.56         |
|                                               | Multi-Metrics   | 0.0293                | 0.1152                | 0.0723             | 11.86        |
|                                               | **Average**     | **0.1228**            | **0.1574**            | **0.1401**         | **15.17**    |
| **2 Attackers (Collusion λ=0.1, No Scale)**   | No-Defense      | 0.2241                | 0.2402                | 0.2322             | 93.61        |
|                                               | Krum            | 0.0515                | 0.1190                | 0.0853             | 34.13        |
|                                               | Multi-Krum      | 0.1189                | 0.1262                | 0.1226             | 11.18        |
|                                               | Foolsgold       | 0.0607                | 0.0768                | 0.0688             | 31.72        |
|                                               | RFA             | 0.1074                | 0.1180                | 0.1127             | 9.94         |
|                                               | Multi-Metrics   | 0.0425                | 0.1068                | 0.0747             | 13.71        |
|                                               | **Average**     | **0.1009**            | **0.1312**            | **0.1160**         | **32.38**    |
| **2 Attackers (Collusion λ=0.1, Scaled)**     | No-Defense      | 0.0630                | 0.0713                | 0.0672             | 11.53        |
|                                               | Krum            | 0.1475                | 0.0898                | 0.1187             | 29.45        |
|                                               | Multi-Krum      | 0.0697                | 0.1232                | 0.0965             | 12.78        |
|                                               | Foolsgold       | 0.0609                | 0.1145                | 0.0877             | 22.47        |
|                                               | RFA             | 0.1097                | 0.1205                | 0.1151             | 9.60         |
|                                               | Multi-Metrics   | 0.1540                | 0.0895                | 0.1218             | 13.58        |
|                                               | **Average**     | **0.1008**            | **0.1015**            | **0.1011**         | **16.57**    |
  
Coordinated Adaptive Scale Critical Parameters Attacks results with varying number of colluders:  
  
| Defense       | O.AdaSCP     | C-AdaSCP(1)  | C-AdaSCP(2)  | C-AdaSCP(3)  | C-AdaSCP(4)  | C-AdaSCP(5)  |
|---------------|--------------|--------------|--------------|--------------|--------------|--------------|
|               | FID/MSE      | FID/MSE      | FID/MSE      | FID/MSE      | FID/MSE      | FID/MSE      |
| **No-Defense**| 12.93/0.0117 | 8.28/0.1231  | 9.53/0.1219  | 9.49/0.1215  | 11.63/0.1184 | 12.06/0.1244 |
| **Krum**      | 30.68/0.0861 | 35.71/0.1045 | 24.98/0.1147 | 32.68/0.0052 | 26.45/0.1111 | 13.69/0.1841 |
| **Multi-Krum**| 8.23/0.1271  | 8.82/0.1304  | 10.21/0.1173 | 17.17/0.0902 | 24.53/0.1055 | N/A          |
| **Foolsgold** | 24.21/0.0129 | 8.33/0.1843  | 8.61/0.1261  | N/A          | N/A          | N/A          |
| **RFA**       | 8.22/0.1233  | 8.71/0.1268  | 9.35/0.1215  | 11.27/0.1153 | 11.44/0.1171 | 12.59/0.1082 |
| **Multi-Metrics**| 15.04/0.0328 | 9.24/0.1226  | 9.46/0.1229  | 9.49/0.1185  | 10.77/0.1209 | 13.69/0.1051 |
| **Average**   | **16.55/0.0657** | **13.18/0.1320** | **12.02/0.1207** | **16.02/0.0901** | **16.96/0.1146** | **13.01/0.1305** |

Key observations:
*   Simply adding uncoordinated attackers can degrade attack success and model utility.
*   Scaling individual contributions for multiple attackers improves model utility (FID) but can reduce attack effectiveness (MSE) without coordination.
*   **Collusion via Model Similarity (Scaled)** often provides a better balance, improving MSE against some robust defenses like Multi-Krum compared to single or uncoordinated attackers, while maintaining good FID.
*   **Coordinated AdaSCP (C-AdaSCP)** consistently adapts to a stealthy scale factor of ≈1.0 enhancing model utility (FID), and while typically yielding higher MSE than the original baseline, it achieved a remarkable MSE of 0.0052 against Krum with three colluders.

For a comprehensive analysis and results against all defenses, please refer to our paper: [Link to paper](https://drive.google.com/file/d/1ivFbPe6RBW_rfQJlGd3xd6QWaVRHi5sa/view?usp=sharing).

## Citation

The original DataStealing paper:
```bibtex
@article{gan2025datastealing,
  title={DataStealing: Steal Data from Diffusion Models in Federated Learning with Multiple Trojans},
  author={Gan, Yuan and Miao, Jiaxu and Yang, Yi},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={132614--132646},
  year={2025}
}
```

## Acknowledgements

This work is an extension of and builds upon the codebase and insights from the original [DataStealing repository](https://github.com/yuangan/DataStealing) by Gan et al.

## License

This project is licensed under the [MIT License]. See the `LICENSE` file for details.
