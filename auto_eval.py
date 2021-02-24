import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

search_dirs = [
    
    # 'logs/cuhk/v25/coscale0.1/Feb20_21-27-00',
    # 'logs/cuhk/v25/coscale0.1-decay18/Feb20_21-28-33',

    'logs/prw/v25/coscale0.1/Feb22_15-14-32',
    'logs/prw/v25/coscale0.1/Feb22_15-14-45',
    'logs/prw/v25/coscale0.1/Feb22_15-14-53',
    'logs/prw/v25/coscale0.1/Feb22_15-15-11',
    'logs/prw/v25/coscale0.1/Feb22_15-15-19',


    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-28',
    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-36',
    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-42',
    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-46',
    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-50',

    'logs/prw/v25-woCoapp/coscale0.1/Feb22_21-13-07',
    'logs/prw/v25-woCoapp/coscale0.1/Feb22_21-13-16',
    'logs/prw/v25-woCoapp/coscale0.1/Feb22_21-13-25',

    'logs/prw/v25-noCycle/coscale0.1/Feb23_13-09-39',
    'logs/prw/v25-noCycle/coscale0.1/Feb23_16-54-10',
    'logs/prw/v25-noCycle/coscale0.1/Feb23_16-54-23',
    'logs/prw/v25-noCycle-noCoap/coscale0.1/Feb23_16-55-04',
    'logs/prw/v25-noCycle-noCoap/coscale0.1/Feb24_09-32-39',
    'logs/prw/v25-noCycle-noCoap/coscale0.1/Feb24_09-32-51',
    
    'logs/prw/v25-coAppIter3-noCycle/coscale0.1/Feb24_09-37-21',
    'logs/prw/v25-coAppIter3-noCycle/coscale0.1/Feb24_09-37-30',
    'logs/prw/v25-coAppIter3-noCycle/coscale0.1/Feb24_09-37-37',
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