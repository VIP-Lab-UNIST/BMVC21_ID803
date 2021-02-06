import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

search_dirs = [ 
    # 'logs/prw/v1/lambda01/Feb02_20-39-04',
    # 'logs/prw/v1/lambda05/Feb02_20-37-50',
    'logs/prw/v2/Feb03_15-15-45',
    'logs/prw/v3/Feb03_15-36-27',
    'logs/prw/v4/temp5/Feb04_11-48-10',
    'logs/prw/v4/temp10/Feb04_11-47-26',

    'logs/prw/v5/Feb06_02-30-35',
    'logs/prw/v5/coscale0.15/Feb06_11-17-22',
    'logs/prw/v5/coscale0.2/Feb06_20-10-49',
    'logs/prw/v5/coscale0.25/Feb06_22-39-37',
    'logs/prw/v7/coscale0.1/Feb06_14-51-21',
    'logs/prw/v7/coscale0.15/Feb06_20-08-03',
    'logs/prw/v8/coscale0.1/Feb06_20-52-19',
    'logs/prw/v8/coscale0.2/Feb06_20-56-47',
    ]

random.shuffle(search_dirs)
while True:
    cnt = 0 
    for root in search_dirs:
        for (path, dirs, files) in os.walk(root):
            files = files[::-1]
            for file_name in files:
                if ('.pth' in file_name) and ('checkpoint' in file_name) and ('last' not in file_name):
                    if (file_name.replace('.pth', '.json')) not in files:
                        checkpoint = os.path.join(path, file_name)
                        print(os.path.join(dname, checkpoint))
                        args_file = os.path.join(path, 'args.json')
                        with open(args_file, 'r') as f:
                            args = json.load(f)
                        
                        tmp = checkpoint.replace('.pth', 'cache.txt')
                        if not os.path.exists(tmp):
                            with open(tmp, 'w') as f:
                                f.write('tmp')

                            command = " python -B scripts/test_NAE.py \
                                        -p %s \
                                        --reid_loss %s \
                                        --dataset %s \
                                        --lr %s \
                                        --batch_size %s \
                                        --num_pids %s \
                                        --num_cq_size %s \
                                        --oim_scalar %s \
                                        --cls_scalar %s \
                                        --part_num %s \
                                        --part_cls_scalar %s \
                                        --checkpoint_name %s" % (
                                            root, 
                                            args['reid_loss'],
                                            args['dataset'],
                                            args['train.lr'],
                                            args['train.batch_size'],
                                            args['num_pids'],
                                            args['num_cq_size'],
                                            args['oim_scalar'],
                                            args['cls_scalar'],
                                            args['part_num'],
                                            args['part_cls_scalar'],
                                            file_name)

                            os.system(command)
                            os.system('rm -rf %s' % tmp)
                            os.system('rm -rf performance.png')
                            os.system('python auto_draw.py')
                            cnt += 1

    if cnt == 0:
        time.sleep(300)
    else:
        time.sleep(10)