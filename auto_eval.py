import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)



search_dirs = [
    
    # 'logs/prw/v25/coscale0.1/Feb22_15-14-32',
    # 'logs/prw/v25/coscale0.1/Feb22_15-14-45',
    # 'logs/prw/v25/coscale0.1/Feb22_15-14-53',
    # 'logs/prw/v25/coscale0.1/Feb22_15-15-11',
    # 'logs/prw/v25/coscale0.1/Feb22_15-15-19',

    # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-28',
    # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-36',
    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-42',
    # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-46',
    # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-50',

    # # 'logs/prw/v25-woCoapp/coscale0.1/Feb22_21-13-07',
    # 'logs/prw/v25-woCoapp/coscale0.1/Feb22_21-13-16',
    # # 'logs/prw/v25-woCoapp/coscale0.1/Feb22_21-13-25',

    # # 'logs/prw/v25-coAppIter3-noCycle/coscale0.1/Feb24_09-37-21',
    # 'logs/prw/v25-coAppIter3-noCycle/coscale0.1/Feb24_09-37-30',
    # # 'logs/prw/v25-coAppIter3-noCycle/coscale0.1/Feb24_09-37-37',

    # # 'logs/prw/v25-noCycle/coscale0.1/Feb23_13-09-39',
    # # # 'logs/prw/v25-noCycle/coscale0.1/Feb23_16-54-10',
    # # # 'logs/prw/v25-noCycle/coscale0.1/Feb23_16-54-23',
    
    # 'logs/prw/v25-noCycle-noCoap/coscale0.1/Feb24_09-32-39',
    # # 'logs/prw/v25-noCycle-noCoap/coscale0.1/Feb24_09-32-51',
    # # 'logs/prw/v25-noCycle-noCoap/coscale0.1/Feb24_17-39-18',

    # 'logs/prw/v32/pathTemp10/Feb24_21-07-13',
    # 'logs/prw/v32/pathTemp1/Feb24_21-08-01',
    # 'logs/prw/v32/pathTemp0.1/Feb24_21-08-49',

    # 'logs/prw/v33/pathTemp10/Feb25_14-01-51',
    # 'logs/prw/v33/pathTemp1/Feb25_14-02-08',
    # 'logs/prw/v33/pathTemp0.1/Feb25_14-02-23',

    # 'logs/prw/v34/pathTemp10-ratio5/Feb25_18-15-02',
    # 'logs/prw/v34/pathTemp10-ratio10/Feb25_18-16-01',    
    # 'logs/prw/v34/pathTemp1-ratio10/Feb25_18-18-41',
    # 'logs/prw/v34/pathTemp1-ratio5/Feb25_18-19-22',

    # 'logs/prw/v35/pathTemp1-ratio5/Feb25_19-52-15',
    # 'logs/prw/v35/pathTemp1-ratio0.2/Feb25_19-53-15',

    # 'logs/prw/v36/pathTemp0.1-ratio10/Feb26_10-51-30',
    # 'logs/prw/v36/pathTemp0.1-ratio1/Feb26_10-51-54',
    # 'logs/prw/v36/pathTemp0.1-ratio0.1/Feb26_10-52-24',
    # 'logs/prw/v36/pathTemp10-ratio10/Feb26_10-41-10',
    # 'logs/prw/v36/pathTemp10-ratio1/Feb26_10-41-41',
    # 'logs/prw/v36/pathTemp10-ratio0.1/Feb26_10-42-31',
    # 'logs/prw/v36/pathTemp1-ratio10/Feb26_10-43-28',
    # 'logs/prw/v36/pathTemp1-ratio1/Feb26_10-44-05',
    # 'logs/prw/v36/pathTemp1-ratio0.1/Feb26_10-48-46',
    
    # 'logs/prw/v36/pathTemp1-ratio0.1/Feb27_09-17-45',
    # 'logs/prw/v36/pathTemp1-ratio0.1/Feb27_09-17-57',
    # 'logs/prw/v37/pathTemp10/Feb27_10-28-35',
    # 'logs/prw/v37/pathTemp1/Feb27_10-28-57',
    # 'logs/prw/v37/pathTemp0.1/Feb27_10-29-33',

    # 'logs/prw/v38/pathTemp10/Feb27_10-23-23',
    # 'logs/prw/v38/pathTemp1/Feb27_10-23-44',
    # 'logs/prw/v38/pathTemp0.1/Feb27_10-24-04',

    # 'logs/prw/v25-allNeg/noCoap-noCycle/Feb28_15-04-08',
    # 'logs/prw/v25-allNeg/noCoap-noCycle/Feb28_15-15-32',
    # 'logs/prw/v25-allNeg/noCoap/Feb28_15-04-36',
    # 'logs/prw/v25-allNeg/noCycle/Feb28_15-05-00',
    # 'logs/prw/v25-allNeg/Feb28_15-08-17',

    # 'logs/prw/v37/pathTemp30/Feb28_15-21-42',
    # 'logs/prw/v37/pathTemp40/Feb28_15-22-06',
    # 'logs/prw/v37/pathTemp50/Feb28_15-22-32',
    # 'logs/prw/v37/pathTemp60/Feb28_15-22-56',

    # 'logs/prw/v25/coscale0.1-cothrd0.7/Mar01_05-07-39',
    # 'logs/prw/v25/coscale0.1-cothrd0.8/Mar01_05-07-54',
    # 'logs/prw/v25/coscale0.1-cothrd0.9/Mar01_05-08-14',
    # 'logs/prw/v25/coscale0.1-cothrd0.5/Mar01_05-08-28',
    
    # 'logs/prw/v25/coscale0.05-cothrd0.6/Mar01_05-14-06',
    # 'logs/prw/v25/coscale0.2-cothrd0.6/Mar01_05-14-31',
    # 'logs/prw/v25/coscale0.3-cothrd0.6/Mar01_05-14-45',
    # 'logs/prw/v25/coscale0.15-cothrd0.6/Mar01_05-15-21',
    # 'logs/prw/v25/coscale0.25-cothrd0.6/Mar01_05-15-35',
    
    # 'logs/prw/v25/coscale0.1-cothrd0.6-neg02/Mar01_21-41-28',
    # 'logs/prw/v25/coscale0.1-cothrd0.6-neg03/Mar01_21-41-51',
    # 'logs/prw/v25/coscale0.1-cothrd0.6-neg04/Mar01_21-42-10',
    # 'logs/prw/v25/coscale0.1-cothrd0.6-neg05/Mar01_21-42-27',
    # 'logs/prw/v25/coscale0.1-cothrd0.6-neg10/Mar01_21-42-50',

    # 'logs/prw/v25/ablation/coscale0.1-cothrd0.3-neg01/Mar04_21-45-20',
    # 'logs/prw/v25/ablation/coscale0.1-cothrd0.4-neg01/Mar04_19-40-53',
    'logs/prw/v25/ablation/coscale0.1-cothrd0.6-neg005/Mar05_14-37-26',
    'logs/prw/v25/ablation/coscale0.1-cothrd0.6-neg001/Mar05_14-37-45',


    'logs/prw/v25/ablation/coscale0.1-cothrd0.55-neg01/Mar05_14-48-09',
    'logs/prw/v25/ablation/coscale0.1-cothrd0.65-neg01/Mar05_10-17-19',
    'logs/prw/v25/ablation/coscale0.1-cothrd0.75-neg01/Mar05_10-17-32',
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
                            
                            os.system('rm -rf performance_cuhk.png')
                            os.system('python auto_draw_cuhk.py')

                            cnt += 1

    if cnt == 0:
        time.sleep(300)
    else:
        time.sleep(10)