import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

search_dirs = [

    'logs/prw/paper/'
    
]   

random.shuffle(search_dirs)
while True:
    cnt = 0 
    for root in search_dirs:
        for (path, dirs, files) in os.walk(root):
            files = files[::-1]
            for file_name in files:
                if ('.pth' in file_name) and ('checkpoint' in file_name) and ('last' not in file_name):
                    if (file_name.replace('.pth', '-prw_multiview.json')) not in files:
                        checkpoint = os.path.join(path, file_name)
                        print(os.path.join(dname, checkpoint))
                        args_file = os.path.join(path, 'args.json')
                        with open(args_file, 'r') as f:
                            args = json.load(f)
                        
                        tmp = checkpoint.replace('.pth', '-prw_multiview_cache.txt')
                        if not os.path.exists(tmp):
                            with open(tmp, 'w') as f:
                                f.write('tmp')

                            command = " python -B runs/test_prw_multiview.py \
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
                                            root, 
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
                            os.system('rm -rf %s' % tmp)
                            os.system('rm -rf performance.png')
                            os.system('python auto_draw_prw.py')
                            
                            # os.system('rm -rf performance_cuhk.png')
                            # os.system('python auto_draw_cuhk.py')

                            cnt += 1

    if cnt == 0:
        time.sleep(300)
    else:
        time.sleep(10)