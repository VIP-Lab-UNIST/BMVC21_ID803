import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

search_dirs = [
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-15-21',
    # # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-16-32',
    # # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-16-43',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar14_10-34-01',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar14_10-34-21',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.6-neg01/Mar14_10-37-13',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01-iter4/Mar14_10-38-01',

    # 'logs/prw/v41/ablation/hnmX-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-17-39',
    # 'logs/prw/v41/ablation/hnmO-hpmX-coscale0.1-simthrd0.6-neg01/Mar13_23-18-01',
    # 'logs/prw/v41/ablation/hnmX-hpmX-coscale0.1-simthrd0.6-neg01/Mar13_23-18-25',

    # 'logs/prw/v41/ablation/hnmO-hpmX-coscale0.1-simthrd0.6-neg01/Mar14_19-12-32',
    # 'logs/prw/v41/ablation/hnmO-hpmX-coscale0.1-simthrd0.6-neg01/Mar14_19-12-39',
    
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.05-simthrd0.6-neg01/Mar15_07-39-10',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.01-simthrd0.6-neg01/Mar15_07-39-34',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.2-simthrd0.6-neg01/Mar14_19-39-27',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.25-simthrd0.6-neg01/Mar14_20-05-33',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.3-simthrd0.6-neg01/Mar14_22-02-10',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.35-simthrd0.6-neg01/Mar14_22-04-19',

    # 'logs/prw/v41/ablation/hnmX-hpmO-coscale0.15-simthrd0.6-neg01/Mar15_07-52-39',
    # 'logs/prw/v41/ablation/hnmO-hpmX-coscale0.15-simthrd0.6-neg01/Mar15_07-45-56',
    # 'logs/prw/v41/ablation/hnmX-hpmX-coscale0.15-simthrd0.6-neg01/Mar15_07-45-17',
    
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.6-neg01-iter2/Mar15_08-04-33',

    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.65-neg01/Mar15_10-04-51',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.70-neg01/Mar15_10-05-09',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.75-neg01/Mar15_10-06-22',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.80-neg01/Mar15_10-07-14',
    
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.55-neg01/Mar15_15-13-36',

    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.6-neg001/Mar15_15-15-36',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.6-neg005/Mar15_15-17-03',
    
    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01-decay20/Mar16_00-30-44',
    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_00-08-25',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_00-15-52',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_00-16-05',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_00-16-12',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_00-16-18',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_00-16-24',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.55-neg01/Mar16_00-22-50',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.65-neg01/Mar16_00-23-04',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.70-neg01/Mar16_00-23-22',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.75-neg01/Mar16_00-23-40',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.80-neg01/Mar16_00-24-00',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg001/Mar16_00-26-52',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg005/Mar16_00-27-25',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg02/Mar16_00-27-40',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg03/Mar16_13-52-46',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg04/Mar16_14-12-16',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg05/Mar16_15-46-18',
    
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.40-simthrd0.60-neg01/Mar16_16-31-33',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_18-14-54',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_18-16-15',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_18-28-02',

    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.6-neg01/Mar16_18-28-27',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.6-neg01/Mar16_18-28-46',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.50-simthrd0.6-neg01/Mar16_19-18-39',
    

    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01-decay20/Mar16_20-09-31',
    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01-decay20/Mar16_20-09-43',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.6-neg01/Mar16_21-45-28',
    
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.6-neg01/Mar17_10-18-47',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.6-neg01/Mar17_10-19-01',

    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.6-neg01/Mar17_10-25-19',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.6-neg01/Mar17_10-39-51',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.6-neg01/Mar17_10-42-46',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.6-neg01/Mar17_12-43-00',

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

                            command = " python -B scripts/test.py \
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
                            os.system('python -B auto_draw.py')
                            
                            # os.system('rm -rf performance_cuhk.png')
                            # os.system('python -B auto_draw_cuhk.py')

                            cnt += 1

    if cnt == 0:
        time.sleep(300)
    else:
        time.sleep(10)