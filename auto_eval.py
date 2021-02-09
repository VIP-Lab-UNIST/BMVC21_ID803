import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

search_dirs = [ 


    'logs/prw/v5/coscale0.1/Feb07_14-00-10',
    'logs/prw/v5/coscale0.15/Feb07_14-00-44',
    'logs/prw/v5/coscale0.2/Feb07_14-01-02',

    # 'logs/prw/v9/coscale0.1/Feb07_14-11-32',
    # 'logs/prw/v9/coscale0.15/Feb07_14-12-13',

    # 'logs/prw/v5-th/coscale0.1/Feb08_08-47-26',
    
    # 'logs/prw/v10/coscale0.1-probTemp0.1/Feb08_18-24-07',
    # 'logs/prw/v10/coscale0.1-probTemp0.2/Feb08_18-24-51',
    # 'logs/prw/v10/coscale0.1-probTemp0.5/Feb08_18-25-34',

    # 'logs/prw/v11/coscale0.1-temp0.1/Feb08_18-37-09',

    'logs/prw/v12/coscale0.1-probTemp0.1/Feb09_14-41-19',
    'logs/prw/v12/coscale0.1-probTemp0.2/Feb09_14-42-03',
    'logs/prw/v12/coscale0.1-probTemp0.5/Feb09_14-42-26',
    
    'logs/prw/v14/coscale0.1/Feb09_15-30-53',
    'logs/prw/v14/coscale0.2/Feb09_15-29-56',

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