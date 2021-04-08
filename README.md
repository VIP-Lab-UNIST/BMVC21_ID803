# Context-Aware Unsupervised Clustering for Person Search 
This repository hosts our code for ICCV2021 "Context-Aware Unsupervised Clustering for Person Search". The code is based on Chen et al's code of NAE4PS(https://github.com/DeanChan/NAE4PS).

## Preparation
1. Clone this repo

   ```bash
   https://github.com/VIP-Lab-UNIST/CUCPS.git && cd CUCPS
   ```

2. Build environment with [conda](https://docs.anaconda.com/anaconda/install/linux/)

   ```bash
   conda env create --prefix <your_conda_env_path> -f environment.yml
   ```

## Experiments
1.  Datasets

    Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](https://github.com/liangzheng06/PRW-baseline) to your directory and change the path in `./lib/datasets/__init__.py`.
    
2. Train
    ```bash
   ./train.sh
   ```
   The training results will be stored under `./logs/`.
   
3. Evaluation
   
    Add the checkpoint directories that you want to evaluate into *serach_dirs* in `auto_eval*.py`. To evaluate for the PRW dataset, you have options `auto_eval-regular.py` or `auto_eval-multiview.py` and CUHK-SYSU dataset, `auto_eval.py`.
    
    ```bash
    python auto_eval*.py
    ```

   It computes mAP and Top-1 scores and records them in *checkpoint_name.json* of the checkpoint folder.
   
