import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

path = './logs/prw/detector/Jan07_14-31-36'
# path = './logs/cuhk/detector/Jan07_14-43-20'
file_name = 'checkpoint_epoch12.pth'
# file_name = 'checkpoint_epoch18.pth'
checkpoint = os.path.join(path, file_name)
print(os.path.join(dname, checkpoint))
args_file = os.path.join(path, 'args.json')
with open(args_file, 'r') as f:
    args = json.load(f)

command = " python -B scripts/crop_detection.py \
            -p %s \
            --reid_loss %s \
            --dataset %s \
            --lr %s \
            --batch_size %s \
            --num_pids %s \
            --num_cq_size %s \
            --oim_scalar %s \
            --cls_scalar %s \
            --checkpoint_name %s" % (
                path, 
                args['reid_loss'],
                args['dataset'],
                args['train.lr'],
                args['train.batch_size'],
                args['num_pids'],
                args['num_cq_size'],
                args['oim_scalar'],
                args['cls_scalar'],
                file_name)

os.system(command)
os.system('rm -rf performance.png')
os.system('python auto_draw.py')
