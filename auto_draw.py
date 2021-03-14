
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure

search_dirs = [
    
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-15-21',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-16-32',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-16-43',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar14_10-34-01',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01/Mar14_10-34-21',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.15-simthrd0.6-neg01/Mar14_10-37-13',
    'logs/prw/v41/ablation/hnmO-hpmO-coscale0.1-simthrd0.6-neg01-iter4/Mar14_10-38-01',

    'logs/prw/v41/ablation/hnmX-hpmO-coscale0.1-simthrd0.6-neg01/Mar13_23-17-39',
    'logs/prw/v41/ablation/hnmO-hpmX-coscale0.1-simthrd0.6-neg01/Mar13_23-18-01',
    'logs/prw/v41/ablation/hnmX-hpmX-coscale0.1-simthrd0.6-neg01/Mar13_23-18-25',

]

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
