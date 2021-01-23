
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure

search_dirs = [
                # 'logs/prw/v3/ce/no_prior_hn001/Jan19_13-08-56',\
                # 'logs/prw/v3/ce/base_hn001/Jan19_13-08-06',\
                # 'logs/prw/v3/ce_org/no_prior/Jan19_13-11-06',\
                # 'logs/prw/v3/ce_org/base/Jan19_13-10-31',\
                # 'logs/prw/v3/ce_org/base_new/Jan19_13-12-28',\

                # './logs/prw/v3/bjhan/no_prior/Jan20_10-36-46',\
                # './logs/prw/v3/bjhan/base_hn001/Jan20_10-37-36',\
                # './logs/prw/v3/bjhan/base_hn002/Jan20_10-38-40',\
                # './logs/prw/v3/bjhan/base_hn003/Jan20_10-38-52',\
                # './logs/prw/v3/bjhan/base_hn005/Jan20_10-39-05',\

                # './logs/prw/v3/bjhan/no_prior/Jan21_09-59-18',\
                # './logs/prw/v3/bjhan/base_hn001/Jan21_09-58-46',\
                # './logs/prw/v3/bjhan/base_hn001/Jan21_09-58-50',\
                # './logs/prw/v3/bjhan/change_loss/Jan21_10-00-52',\

                # './logs/prw/v3/bjhan/no_prior/Jan22_10-36-42',\
                './logs/prw/v3/bjhan/no_prior/Jan22_10-36-53',\
                # './logs/prw/v3/bjhan/no_prior/Jan22_10-37-01',\
                './logs/prw/v3/bjhan/base_hn001/Jan22_10-37-22',\
                # './logs/prw/v3/bjhan/base_hn001/Jan22_10-37-32',\
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
