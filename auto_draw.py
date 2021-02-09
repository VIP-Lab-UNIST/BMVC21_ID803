
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure

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

    'logs/prw/v14/coscale0.1/Feb09_15-22-08',
    'logs/prw/v14/coscale0.2/Feb09_15-22-26',

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
