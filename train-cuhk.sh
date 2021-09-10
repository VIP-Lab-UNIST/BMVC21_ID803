python -B runs/train.py  \
        --dataset CUHK-SYSU \
        --num_pids 5532 \
        --num_cq_size 5000 \
        --lr_warm_up \
        --lr_decay_step 20 \
        --batch_size 4 \
        --start_epoch 0 \
        --path ./logs/cuhk-sysu/
                       