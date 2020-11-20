# Person search 

This repository hosts our code for CVPR2021

## Preparation

This code is based on Chen et al's code for NAE4PS.
Follow all process introduced in their [repository](https://github.com/DeanChan/NAE4PS).

## Experiments

1.  Datasets

    Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](https://github.com/liangzheng06/PRW-baseline) to `/root/workspace/datasets/PersonSearch/`.
    
    You can change the dataset directory in `./lib/datasets/__init__.py`.
    
2. Train
    ```bash
   ./train.sh
   ```
   The training results will be stored under `./logs/`.
   
3. Evaluation
   
    Add the checkpoint directories that you want to evaluate or visualize into *serach_dirs* in auto_eval.py or auto_eval.py.
    
    ```bash
   python auto_eval.py
   ```
   It computes mAP and Top-1 scores and records them in *checkpoint_name.json* of the checkpoint folder. The performance chart is saved in *performance.png*.
   
