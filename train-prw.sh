python -B runs/train.py  \
        --dataset PRW \
        --num_pids 482 \
        --num_cq_size 500 \
        --lr_decay_step 14 \
        --batch_size 4 \
        --start_epoch 6 \
        --path ./logs/tmp/
                       