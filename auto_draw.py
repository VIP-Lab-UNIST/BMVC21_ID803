
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure

search_dirs_all = [

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-15-21',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-16-32',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-16-43',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar14_10-34-01',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar14_10-34-21',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01-iter4/Mar14_10-38-01',

    'logs/prw/v41/ablation/hnmX-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-17-39',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar13_23-18-01',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.60-neg01/Mar13_23-18-25',

    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar14_19-12-32',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar14_19-12-39',


    ## Beta
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.01-simthrd0.60-neg01/Mar15_07-39-34',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.05-simthrd0.60-neg01/Mar15_07-39-10',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-15-21',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.20-simthrd0.60-neg01/Mar14_19-39-27',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.25-simthrd0.60-neg01/Mar14_20-05-33',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.30-simthrd0.60-neg01/Mar14_22-02-10',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.35-simthrd0.60-neg01/Mar14_22-04-19',

    ## Module ablation
    'logs/prw/v41/ablation/hnmX-hpmO-coscale0.15-simthrd0.60-neg01/Mar15_07-52-39',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.15-simthrd0.60-neg01/Mar15_07-45-56',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.15-simthrd0.60-neg01/Mar15_07-45-17',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01-iter2/Mar15_08-04-33',

    ## Sim threshold
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.55-neg01/Mar15_15-13-36',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.65-neg01/Mar15_10-04-51',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.70-neg01/Mar15_10-05-09',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.75-neg01/Mar15_10-06-22',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.80-neg01/Mar15_10-07-14',

    ## Negsize
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg001/Mar15_15-15-36',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg005/Mar15_15-17-03',

]

search_dirs_tmp = [
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-15-21',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',

    # 'logs/prw/v25-coAPPieter3/ablation/coscale0.10-cothrd0.6-neg01/Feb23_09-16-42',
    # 'logs/prw/v25-coAPPieter3/ablation/coscale0.10-cothrd0.6-neg02/Mar01_21-41-28',
    # 'logs/prw/v25-coAPPieter3/ablation/coscale0.10-cothrd0.6-neg03/Mar01_21-41-51',
    # 'logs/prw/v25-coAPPieter3/ablation/coscale0.10-cothrd0.6-neg04/Mar01_21-42-10',
    # 'logs/prw/v25-coAPPieter3/ablation/coscale0.10-cothrd0.6-neg05/Mar01_21-42-27',
    # 'logs/prw/v25-coAPPieter3/ablation/coscale0.10-cothrd0.6-neg10/Mar01_21-42-50',


    'logs/cuhk/v25/coscale0.10-decay20-coapIter3/Mar08_12-03-51',
    'logs/cuhk/v25/coscale0.10-decay22-coapIter3/Mar08_12-04-12',


]

search_dirs_prop = [

    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-15-21',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_00-15-52',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_00-16-05',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_00-16-12',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_00-16-24',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-14-54',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-28-02',

    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_00-16-18',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-16-15',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_21-45-28',
]

# search_dirs_old = [
#     # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-16-32',
#     # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-16-43',
#     # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar14_10-34-01',
#     # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar14_10-34-21',
#     'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-15-21',
#     'logs/prw/v41/ablation/hnmX-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-17-39',
#     'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar13_23-18-01',
#     'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.60-neg01/Mar13_23-18-25',
# ]

search_dirs_beta = [

    ## Beta
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.01-simthrd0.60-neg01/Mar15_07-39-34',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.05-simthrd0.60-neg01/Mar15_07-39-10',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-16-15',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.20-simthrd0.60-neg01/Mar14_19-39-27',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.25-simthrd0.60-neg01/Mar14_20-05-33',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.30-simthrd0.60-neg01/Mar14_22-02-10',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.35-simthrd0.60-neg01/Mar14_22-04-19',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.40-simthrd0.60-neg01/Mar16_16-31-33',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.50-simthrd0.60-neg01/Mar16_19-18-39',

]


search_dirs_modules = [
    ## Module ablation
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-16-15',
    'logs/prw/v41/ablation/hnmX-hpmO-coscale0.10-simthrd0.60-neg01/Mar13_23-17-39',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar13_23-18-01',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.60-neg01/Mar13_23-18-25',

    # 'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar16_18-28-27',
    # 'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.60-neg01/Mar16_18-28-46',
    
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar17_10-18-47',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar17_10-19-01',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar17_10-25-19',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar17_10-39-51',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.10-simthrd0.60-neg01/Mar17_12-43-00',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.10-simthrd0.60-neg01/Mar17_10-42-46',
]

search_dirs_simtrd = [
    ## Sim threshold
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.55-neg01/Mar15_15-13-36',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.65-neg01/Mar15_10-04-51',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.70-neg01/Mar15_10-05-09',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.75-neg01/Mar15_10-06-22',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.80-neg01/Mar15_10-07-14',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.55-neg01/Mar16_00-22-50',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-16-15',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.65-neg01/Mar16_00-23-04',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.70-neg01/Mar16_00-23-22',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.75-neg01/Mar16_00-23-40',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.80-neg01/Mar16_00-24-00',
]

search_dirs_negsize = [
    ## Negsize
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg001/Mar15_15-15-36',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg005/Mar15_15-17-03',
    # 'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar14_10-37-13',

    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg001/Mar16_00-26-52',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg005/Mar16_00-27-25',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_18-16-15',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg02/Mar16_00-27-40',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg03/Mar16_13-52-46',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg04/Mar16_14-12-16',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg05/Mar16_15-46-18',
]

search_dirs_cuhk = [
    # 'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.60-neg01/Mar15_23-46-59',
    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01/Mar16_00-08-25',
    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01-decay20/Mar16_00-30-44',

    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01-decay20/Mar16_20-09-31',
    'logs/cuhk/v41/ablation/hnmO-hpmO-coscale0.10-simthrd0.60-neg01-decay20/Mar16_20-09-43',
]


def draw(search_dirs, figure_name):
    legend=[]    
    fig = plt.figure(num=None, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')        
    ax = plt.subplot(111)
    for root in search_dirs:
        for (path, dirs, files) in os.walk(root):
            do_draw=False
            for file_name in files:
                if '.json' in file_name:
                    do_draw=True  
                    break  
            if do_draw:
                steps = [] 
                mAPs = []
                for step in range(1, 30):
                    checkpoint = os.path.join(path, 'checkpoint_epoch%d.json'%step)
                    if os.path.isfile(checkpoint):
                        steps.append(step)
                        with open(checkpoint, 'r') as f:
                            performance = json.load(f)
                            mAPs.append(performance['mAP'])
                
                ax.plot(steps, mAPs, 'o-')
                legend.append(path)
            

    ax.grid(True)
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

    if 'cuhk' not in figure_name:
        plt.ylim(0.0,0.4)
    else:
        plt.ylim(0.0,0.99)

    plt.xlabel('epoch')
    plt.ylabel('mAP')
    ax.legend(legend, loc="lower right", fontsize=12)
    plt.tight_layout()

    # plt.axhline(y=0.45, color='r', linestyle='-')
    plt.axhline(y=0.95, color='r', linestyle='-')
    plt.axhline(y=0.9, color='r', linestyle='-')
    plt.show()

    os.system('rm -rf %s.png'%figure_name)
    plt.savefig('%s.png'%figure_name)


os.system('rm -rf figures/performance-*.png')
# draw(search_dirs_all, 'figures/performance-all')
# draw(search_dirs_tmp, 'figures/performance-tmp-cuhk')
# draw(search_dirs_prop, 'figures/performance-prop')
draw(search_dirs_modules, 'figures/performance-modules')
draw(search_dirs_beta, 'figures/performance-beta')
draw(search_dirs_simtrd, 'figures/performance-simtrd')
draw(search_dirs_negsize, 'figures/performance-negsize')
draw(search_dirs_cuhk, 'figures/performance-cuhk')