
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure

search_dirs = [

                # './logs/prw/v9/no_prior/Jan31_11-54-06', \
                # './logs/prw/v9/base_nocycle/Jan31_11-55-07', \
                # './logs/prw/v9/base/Jan31_11-56-09', \
                # './logs/prw/v9/no_prior_hn0008/Jan31_11-57-35', \
                # './logs/prw/v9/no_prior_hn0005/Jan31_11-58-22', \


                './logs/prw/v10/base/Feb01_11-37-16', \
                './logs/prw/v9/no_prior/Feb01_09-58-41', \
                './logs/prw/v9/base_nocycle/Feb01_09-59-48', \
                './logs/prw/v9/base/Feb01_10-00-14', \
                './logs/prw/v10/base_nocycle/Feb01_11-36-44',\
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
