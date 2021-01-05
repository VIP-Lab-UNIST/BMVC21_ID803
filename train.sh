
                        
# CUDA_VISIBLE_DEVICES=0 python -B scripts/train_NAE.py  \
#                         --reid_loss oim \
#                         --batch_size 4 \
#                         --lr_decay_gamma 0.1 \
#                         --epochs 20 \
#                         --lr 0.003 \
#                         --lr_decay_step 12 \
#                         --w_RCNN_loss_bbox 10.0 \
#                         --oim_scalar 30.0 \
#                         --cls_scalar 1.0 \
#                         --part_num 4 \
#                         --part_cls_scalar 10.0 \
#                         --num_cq_size 5000 \
#                         --path ./logs/cuhk/bs4-partScalar10-partNum4-clsscalar1-lr003-step12 \
#                         --num_pids 5532 \
#                         --oim_momentum 0.5 \
#                         --momentum 0.9 \
#                         --batch_size_test 1 \
#                         --dataset CUHK-SYSU \
#                         --device cuda \
#                         --rpn_post_nms_top_n 2000 \
#                         --rpn_pre_nms_top_n_test 6000 \
#                         --w_RPN_loss_box 1.0 \
#                         --anchor_scales 32 64 128 256 512 \
#                         --rpn_pre_nms_top_n 12000 \
#                         --rpn_fg_fraction 0.5 \
#                         --min_size 900 \
#                         --rpn_min_size_test 16 \
#                         --fg_thresh 0.5 \
#                         --bg_thresh_hi 0.5 \
#                         --fg_fraction 0.5 \
#                         --nw 0 \
#                         --weight_decay 0.0005 \
#                         --num_features 256 \
#                         --rpn_batch_size 256 \
#                         --max_size 1500 \
#                         --bg_thresh_lo 0.1 \
#                         --disp_interval 10 \
#                         --rpn_nms_thresh_test 0.7 \
#                         --max_size_test 1500 \
#                         --rpn_positive_overlap 0.7 \
#                         --w_RPN_loss_cls 1.0 \
#                         --min_size_test 900 \
#                         --clip_gradient 10.0 \
#                         --w_RCNN_loss_cls 1.0 \
#                         --seed 1 \
#                         --nms_test 0.4 \
#                         --start_epoch 0 \
#                         --rcnn_batch_size 128 \
#                         --rpn_min_size 8 \
#                         --rpn_nms_thresh 0.7 \
#                         --box_regression_weights 10.0 10.0 5.0 5.0 \
#                         --rpn_negative_overlap 0.3 \
#                         --net resnet50 \
#                         --checkpoint_interval 1 \
#                         --rpn_post_nms_top_n_test 300 \
#                         --aspect_grouping -1 \
#                         --anchor_ratios 0.5 1.0 2.0 \
#                         --lr_warm_up \
#                         --w_OIM_loss_oim 1.0

                        # --path ./logs/prw/bs4/uniq_v1/ \
                        # --path ./logs/prw/bs4/base/f256 \
                        # --path ./logs/prw/tmp \
                        # --resume ./logs/prw/bs4/base/f256/Dec27_14-51-14/checkpoint_epoch22.pth \

CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python -B scripts/train_NAE.py  \
                        --reid_loss oim \
                        --dataset PRW \
                        --batch_size 4 \
                        --lr_decay_step 18 \
                        --lr 0.003 \
                        --lr_decay_gamma 0.1 \
                        --epochs 35 \
                        --oim_scalar 30.0 \
                        --cls_scalar 1.0 \
                        --w_RCNN_loss_bbox 10.0 \
                        --anchor_scales 32 64 128 256 512 \
                        --num_cq_size 500 \
                        --w_OIM_loss_oim 1.0 \
                        --part_num 4 \
                        --part_cls_scalar 10 \
                        --num_features 256 \
                        --num_pids 482 \
                        --oim_momentum 0.5 \
                        --batch_size_test 1 \
                        --device cuda \
                        --rpn_post_nms_top_n 2000 \
                        --rpn_pre_nms_top_n_test 6000 \
                        --w_RPN_loss_box 1.0 \
                        --rpn_pre_nms_top_n 12000 \
                        --rpn_fg_fraction 0.5 \
                        --min_size 900 \
                        --rpn_min_size_test 16 \
                        --fg_thresh 0.5 \
                        --nw 0 \
                        --momentum 0.9 \
                        --rpn_negative_overlap 0.3 \
                        --rpn_batch_size 256 \
                        --max_size 1500 \
                        --bg_thresh_lo 0.1 \
                        --disp_interval 10 \
                        --rpn_nms_thresh_test 0.7 \
                        --bg_thresh_hi 0.5 \
                        --max_size_test 1500 \
                        --rpn_positive_overlap 0.7 \
                        --w_RPN_loss_cls 1.0 \
                        --fg_fraction 0.5 \
                        --clip_gradient 10.0 \
                        --w_RCNN_loss_cls 1.0 \
                        --seed 1 \
                        --nms_test 0.4 \
                        --start_epoch 0 \
                        --rcnn_batch_size 128 \
                        --rpn_min_size 8 \
                        --rpn_nms_thresh 0.7 \
                        --box_regression_weights 10.0 10.0 5.0 5.0 \
                        --weight_decay 0.0005 \
                        --net resnet50 \
                        --checkpoint_interval 1 \
                        --rpn_post_nms_top_n_test 300 \
                        --aspect_grouping -1 \
                        --anchor_ratios 0.5 1.0 2.0 \
                        --lr_warm_up \
                        --min_size_test 900 \
                        --path ./logs/prw/bs4/tc06sc05num10 \
                        --co_thrd 0.6 \
                        --co_scale 0.5 \
                        --num_neg 10 
                       
                        # --path ./logs/prw/tmp \
                        # --path ./logs/prw/bs4/uniq_v1/ \
                        # --resume ./logs/prw/bs4/base/f256/Dec27_14-51-14/checkpoint_epoch12.pth \
                        # --embedding_feat_fuse 

                        
