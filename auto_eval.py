import os
import time
import json
import random

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

search_dirs = [ 

    # './logs/prw/v3/bjhan/no_prior_negonly/Jan29_09-30-59', \

    # './logs/prw/v3/bjhan/no_prior_hnall/Jan28_22-21-40',\
    # './logs/prw/v3/bjhan/no_prior_posall/Jan28_22-14-45', \
    # './logs/prw/v3/bjhan/no_prior_posall_hnall/Jan28_22-36-58',\
    # './logs/prw/v3/bjhan/no_prior_no_multilabel/Jan29_11-40-31', \


    # './logs/prw/v8/no_prior_keep_rate/Jan29_22-27-10',\
    # './logs/prw/v8/base_keep_rate/Jan29_22-40-45', \
    # './logs/prw/v8/no_prior_easy/Jan30_10-24-25',\

    # './logs/prw/v8/no_prior/Jan29_21-04-28', \
    # './logs/prw/v8/base/Jan29_21-05-03', \
    # './logs/prw/v8/base_cycle_th05/Jan29_21-15-25', \

    # './logs/prw/v9/base_nocycle/Jan30_21-16-54', \
    # './logs/prw/v9/base/Jan30_21-17-40', \
    # './logs/prw/v9/base_th05/Jan30_21-18-27', \
    # './logs/prw/v9/base_th04/Jan30_21-19-10', \
    # './logs/prw/v9/base_th07/Jan30_21-19-41', \

    # './logs/prw/v9/no_prior/Jan31_11-54-06', \
    # './logs/prw/v9/base_nocycle/Jan31_11-55-07', \
    # './logs/prw/v9/base/Jan31_11-56-09', \
    # './logs/prw/v9/no_prior_hn0008/Jan31_11-57-35', \
    # './logs/prw/v9/no_prior_hn0005/Jan31_11-58-22', \

    # './logs/prw/v10/base/Feb01_11-37-16', \
    # './logs/prw/v9/no_prior/Feb01_09-58-41', \
    # './logs/prw/v9/base_nocycle/Feb01_09-59-48', \
    # './logs/prw/v9/base/Feb01_10-00-14', \
    './logs/prw/v10/base_nocycle/Feb01_11-36-44',\

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