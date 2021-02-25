
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure


search_dirs = [
    
    # # 'logs/prw/v25/coscale0.1/Feb22_15-14-32',
    # # 'logs/prw/v25/coscale0.1/Feb22_15-14-45',
    # # 'logs/prw/v25/coscale0.1/Feb22_15-14-53',
    # # 'logs/prw/v25/coscale0.1/Feb22_15-15-11',
    # # 'logs/prw/v25/coscale0.1/Feb22_15-15-19',

    # # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-28',
    # # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-36',
    'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-42',
    # # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-46',
    # # 'logs/prw/v25-coAPPieter3/coscale0.1/Feb23_09-16-50',

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

    'logs/prw/v32/pathTemp10/Feb24_21-07-13',
    # 'logs/prw/v32/pathTemp1/Feb24_21-08-01',
    # 'logs/prw/v32/pathTemp0.1/Feb24_21-08-49',

    # 'logs/prw/v33/pathTemp10/Feb25_14-01-51',
    # 'logs/prw/v33/pathTemp1/Feb25_14-02-08',
    # 'logs/prw/v33/pathTemp0.1/Feb25_14-02-23',

]   

# search_dirs = [ 
#     'logs/prw/v14/coscale0.4-weight0.1/Feb10_13-23-00',
#     'logs/prw/v14/coscale0.4-weight0.3/Feb10_13-22-13',
#     'logs/prw/v14/coscale0.4/Feb10_13-19-45',
# ]
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

plt.ylim(0.0,0.4)
plt.xlabel('epoch')
plt.ylabel('mAP')
ax.legend(legend, loc="lower right", fontsize=12)
plt.tight_layout()

# plt.axhline(y=0.45, color='r', linestyle='-')
plt.axhline(y=0.95, color='r', linestyle='-')
plt.axhline(y=0.9, color='r', linestyle='-')
plt.show()

os.system('rm -rf performance.png')
plt.savefig('performance.png')
