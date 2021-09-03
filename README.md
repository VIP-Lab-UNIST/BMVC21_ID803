# Person search 

Official pytorch implementation for BMVC 2021 [Context-Aware Unsupervised Clustering for Person Search]()

This code is based on Chen et al's code for [NAE4PS](https://github.com/DeanChan/NAE4PS).

## Preparation

1.  Datasets

    Download [CUHK-SYSU](https://github.com/ShuangLI59/person_search) and [PRW](https://github.com/liangzheng06/PRW-baseline) to your location and set the dataset path in `./lib/datasets/__init__.py`.

## Train
    
1. Train
    ```bash
   ./train.sh
   ```
   The training results will be stored under `./logs/`.
   
## Test

1. Test CUHK-SYSU
   
    Add the checkpoint directories that you want to evaluate or visualize into *serach_dirs* in auto_eval.py or auto_eval.py.
    
    ```bash
   python auto_eval.py
   ```
   It computes mAP and Top-1 scores and records them in *checkpoint_name.json* of the checkpoint folder. The performance chart is saved in *performance.png*.
   
2. Test PRW(regular gallery)

    If you want to evaluate on the multi-view gallery, set `ignore_cam_id=False` and `remove_unlabel=False` on `runs/test.py` 
3. Test PRW(multi-view gallery)