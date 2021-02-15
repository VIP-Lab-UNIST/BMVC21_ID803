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

    # 'logs/prw/v12/coscale0.1-probTemp0.1/Feb09_14-41-19',
    # 'logs/prw/v12/coscale0.1-probTemp0.2/Feb09_14-42-03',
    # 'logs/prw/v12/coscale0.1-probTemp0.5/Feb09_14-42-26',
    
    # 'logs/prw/v14/coscale0.1/Feb09_15-30-53',
    # 'logs/prw/v14/coscale0.2/Feb09_15-29-56',

    # 'logs/prw/v14/coscale0.3/Feb10_13-19-19',
    # 'logs/prw/v14/coscale0.3-weight0.1/Feb10_13-23-34',

    # 'logs/prw/v14/coscale0.4/Feb10_13-19-45',
    # 'logs/prw/v14/coscale0.4-weight0.3/Feb10_13-22-13',
    # 'logs/prw/v14/coscale0.4-weight0.1/Feb10_13-23-00',

    # 'logs/prw/v14/coscale0.1-weight0.1/Feb11_11-33-16',
    # 'logs/prw/v14/coscale0.1-weight0.5/Feb11_11-33-34',
    # 'logs/prw/v14/coscale0.1-weight1.0/Feb11_11-34-03',
    # 'logs/prw/v14/coscale0.1-weight2.0/Feb11_11-34-28',
    # 'logs/prw/v14/coscale0.1-weight3.0/Feb11_11-34-50',

    # 'logs/prw/v14/coscale0.1-weight4.0/Feb12_02-22-08',
    'logs/prw/v14/coscale0.1-weight5.0/Feb12_02-22-29',
    # 'logs/prw/v14/coscale0.1-weight6.0/Feb12_02-22-55',
    # 'logs/prw/v14/coscale0.1-weight7.0/Feb12_02-23-15',
    # 'logs/prw/v14/coscale0.1-weight8.0/Feb12_02-23-33',

    # 'logs/prw/v17/coscale0.1-weight1.0/Feb13_01-43-34',
    # 'logs/prw/v17/coscale0.1-weight2.0/Feb13_01-43-59',
    # 'logs/prw/v17/coscale0.1-weight3.0/Feb13_01-44-21',
    # 'logs/prw/v17/coscale0.1-weight4.0/Feb13_01-44-44',
    # 'logs/prw/v17/coscale0.1-weight5.0/Feb13_01-45-09',
    # 'logs/prw/v17/coscale0.1-weight6.0/Feb13_01-45-28',
    # 'logs/prw/v17/coscale0.1-weight7.0/Feb13_01-45-56',

    # 'logs/prw/v18/coscale0.1-weight0.5/Feb13_13-30-19',
    # 'logs/prw/v18/coscale0.1-weight1.0/Feb13_13-30-47',
    # 'logs/prw/v18/coscale0.1-weight2.0/Feb13_13-31-07',
    # 'logs/prw/v18/coscale0.1-weight3.0/Feb13_13-31-29',
    # 'logs/prw/v18/coscale0.1-weight4.0/Feb13_13-31-52',
    # 'logs/prw/v18/coscale0.1-weight5.0/Feb13_13-32-23',
    # 'logs/prw/v18/coscale0.1-weight0.1/Feb13_13-33-03',

    # 'logs/prw/v19/coscale0.1-weight0.1/Feb13_23-32-45',
    # 'logs/prw/v19/coscale0.1-weight1.0/Feb13_23-33-42',
    # 'logs/prw/v19/coscale0.1-weight2.0/Feb13_23-34-02',
    # 'logs/prw/v19/coscale0.1-weight3.0/Feb13_23-34-23',
    # 'logs/prw/v19/coscale0.1-weight4.0/Feb13_23-34-41',
    # 'logs/prw/v19/coscale0.1-weight5.0/Feb13_23-35-03',


    # 'logs/prw/v20/coscale0.1-weight0.1/Feb14_13-15-45',
    # 'logs/prw/v20/coscale0.1-weight0.5/Feb14_13-16-15',
    # 'logs/prw/v20/coscale0.1-weight1.0/Feb14_13-16-38',
    # 'logs/prw/v20/coscale0.1-weight2.0/Feb14_13-16-55',
    # 'logs/prw/v20/coscale0.1-weight3.0/Feb14_13-17-20',
    # 'logs/prw/v20/coscale0.1-weight4.0/Feb14_13-17-39',
    # 'logs/prw/v20/coscale0.1-weight5.0/Feb13_23-55-07',

    # 'logs/prw/v20/coscale0.01-weight1.0/Feb15_05-25-11',
    # 'logs/prw/v20/coscale0.05-weight1.0/Feb15_05-25-47',
    # 'logs/prw/v20/coscale0.15-weight1.0/Feb15_05-26-04',
    # 'logs/prw/v22/coscale0.1-weight1.0-normalized/Feb15_05-57-53',
    # 'logs/prw/v22/coscale0.1-weight1.0-scale0.3/Feb15_05-58-48',
    
    'logs/prw/v23/coscale0.1-weight5.0/Feb15_15-04-52',
    'logs/prw/v23/coscale0.1-weight3.0/Feb15_16-00-32',
    'logs/prw/v23/coscale0.1-weight1.0/Feb15_15-03-37',
    'logs/prw/v23/coscale0.1-weight0.5/Feb15_15-07-10',
    'logs/prw/v23/coscale0.1-weight0.1/Feb15_15-07-46',
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