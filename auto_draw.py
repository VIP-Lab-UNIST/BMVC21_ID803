
import json    

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os

import os
import time
from matplotlib.pyplot import figure



search_dirs = [ 

    # 'logs/prw/v5/coscale0.1/Feb07_14-00-10',
    # 'logs/prw/v5/coscale0.15/Feb07_14-00-44',
    # 'logs/prw/v5/coscale0.2/Feb07_14-01-02',

    # 'logs/prw/v9/coscale0.1/Feb07_14-11-32',
    # 'logs/prw/v9/coscale0.15/Feb07_14-12-13',

    # 'logs/prw/v5-th/coscale0.1/Feb08_08-47-26',
    
    # 'logs/prw/v10/coscale0.1-probTemp0.1/Feb08_18-24-07',
    # 'logs/prw/v10/coscale0.1-probTemp0.2/Feb08_18-24-51',
    # 'logs/prw/v10/coscale0.1-probTemp0.5/Feb08_18-25-34',

    # 'logs/prw/v11/coscale0.1-temp0.1/Feb08_18-37-09',

    # 'logs/prw/v12/coscale0.1-probTemp0.1/Feb09_14-41-19',
    # 'logs/prw/v12/coscale0.1-probTemp0.2/Feb09_14-42-03',
    # 'logs/prw/v12/coscale0.1-probTemp0.5/Feb09_14-42-26',

    # 'logs/prw/v14/coscale0.1/Feb09_15-30-53',
    # 'logs/prw/v14/coscale0.2/Feb09_15-29-56',

    # 'logs/prw/v14/coscale0.3/Feb10_13-19-19',
    # # 'logs/prw/v14/coscale0.3-weight0.1/Feb10_13-23-34',

    # 'logs/prw/v14/coscale0.4/Feb10_13-19-45',
    # 'logs/prw/v14/coscale0.4-weight0.3/Feb10_13-22-13',
    # 'logs/prw/v14/coscale0.4-weight0.1/Feb10_13-23-00',

    # 'logs/prw/v14/coscale0.1-weight0.1/Feb11_11-33-16',
    # 'logs/prw/v14/coscale0.1-weight0.5/Feb11_11-33-34',
    # 'logs/prw/v14/coscale0.1-weight1.0/Feb11_11-34-03',
    # 'logs/prw/v14/coscale0.1-weight2.0/Feb11_11-34-28',
    # 'logs/prw/v14/coscale0.1-weight3.0/Feb11_11-34-50',

    # 'logs/prw/v14/coscale0.1-weight4.0/Feb12_02-22-08',
    'logs/prw/v14/coscale0.1-weight5.0/Feb12_02-22-29',
    # 'logs/prw/v14/coscale0.1-weight6.0/Feb12_02-22-55',
    # 'logs/prw/v14/coscale0.1-weight7.0/Feb12_02-23-15',
    # 'logs/prw/v14/coscale0.1-weight8.0/Feb12_02-23-33',
    
    # 'logs/prw/v17/coscale0.1-weight1.0/Feb13_01-43-34',
    # 'logs/prw/v17/coscale0.1-weight2.0/Feb13_01-43-59',
    # 'logs/prw/v17/coscale0.1-weight3.0/Feb13_01-44-21',
    # 'logs/prw/v17/coscale0.1-weight4.0/Feb13_01-44-44',
    # 'logs/prw/v17/coscale0.1-weight5.0/Feb13_01-45-09',
    # 'logs/prw/v17/coscale0.1-weight6.0/Feb13_01-45-28',
    # 'logs/prw/v17/coscale0.1-weight7.0/Feb13_01-45-56',

    # 'logs/prw/v18/coscale0.1-weight0.5/Feb13_13-30-19',
    # 'logs/prw/v18/coscale0.1-weight1.0/Feb13_13-30-47',
    # 'logs/prw/v18/coscale0.1-weight2.0/Feb13_13-31-07',
    # 'logs/prw/v18/coscale0.1-weight3.0/Feb13_13-31-29',
    # 'logs/prw/v18/coscale0.1-weight4.0/Feb13_13-31-52',
    # 'logs/prw/v18/coscale0.1-weight5.0/Feb13_13-32-23',
    # 'logs/prw/v18/coscale0.1-weight0.1/Feb13_13-33-03',
    
    # 'logs/prw/v19/coscale0.1-weight0.1/Feb13_23-32-45',
    # 'logs/prw/v19/coscale0.1-weight1.0/Feb13_23-33-42',
    # 'logs/prw/v19/coscale0.1-weight2.0/Feb13_23-34-02',
    # 'logs/prw/v19/coscale0.1-weight3.0/Feb13_23-34-23',
    # 'logs/prw/v19/coscale0.1-weight4.0/Feb13_23-34-41',
    # 'logs/prw/v19/coscale0.1-weight5.0/Feb13_23-35-03',

    # 'logs/prw/v20/coscale0.1-weight0.1/Feb14_13-15-45',
    # 'logs/prw/v20/coscale0.1-weight0.5/Feb14_13-16-15',
    # 'logs/prw/v20/coscale0.1-weight1.0/Feb14_13-16-38',
    # 'logs/prw/v20/coscale0.1-weight2.0/Feb14_13-16-55',
    # 'logs/prw/v20/coscale0.1-weight3.0/Feb14_13-17-20',
    # 'logs/prw/v20/coscale0.1-weight4.0/Feb14_13-17-39',
    # 'logs/prw/v20/coscale0.1-weight5.0/Feb13_23-55-07',
    # 'logs/prw/v20/coscale0.01-weight1.0/Feb15_05-25-11',
    # 'logs/prw/v20/coscale0.05-weight1.0/Feb15_05-25-47',
    # 'logs/prw/v20/coscale0.15-weight1.0/Feb15_05-26-04',
    # 'logs/prw/v22/coscale0.1-weight1.0-normalized/Feb15_05-57-53',
    # 'logs/prw/v22/coscale0.1-weight1.0-scale0.3/Feb15_05-58-48',
    
    # 'logs/prw/v23/coscale0.1-weight5.0/Feb15_15-04-52',
    # 'logs/prw/v23/coscale0.1-weight3.0/Feb15_16-00-32',
    # 'logs/prw/v23/coscale0.1-weight1.0/Feb15_15-03-37',
    # 'logs/prw/v23/coscale0.1-weight0.5/Feb15_15-07-10',
    # 'logs/prw/v23/coscale0.1-weight0.1/Feb15_15-07-46',
    
    # 'logs/prw/v24/coscale0.1-normbyProbweight/Feb16_03-24-27',
    # 'logs/prw/v24/coscale0.1-normbyProbweight/Feb16_03-28-55',
    # 'logs/prw/v24/coscale0.1-normbysum/Feb16_03-08-42',
    # 'logs/prw/v24/coscale0.1-normbysum/Feb16_03-31-08',
    # 'logs/prw/v20-th/coscale0.1-weight1.0-epoch18th0.8/Feb16_04-02-05',

    # 'logs/prw/v24/coscale0.1-normbysum-nosqrt/Feb16_16-34-05',
    # 'logs/prw/v24/coscale0.1-normbyProbweight-nosqrt/Feb16_16-34-42',
    # 'logs/prw/v24/coscale0.1-normbysum-max/Feb16_16-36-38',
    # 'logs/prw/v24/coscale0.1-normbyProbweight-max/Feb16_16-36-11',    
    # 'logs/prw/v24/coscale0.1-normbysum-min/Feb16_17-56-00',
    # 'logs/prw/v24/coscale0.1-normbyProbweight-min/Feb16_18-17-22',

    # 'logs/prw/v26/coscale0.1-ratio1/Feb17_02-07-59',
    # 'logs/prw/v26/coscale0.1-ratio3/Feb17_02-08-29',
    # 'logs/prw/v26/coscale0.1-ratio5/Feb17_07-54-35',
    # 'logs/prw/v26/coscale0.1-ratio8/Feb17_07-55-03',
    # 'logs/prw/v26/coscale0.1-ratio10/Feb17_07-55-28',

    # 'logs/prw/v26/coscale0.1-ratio20/Feb17_08-50-48',
    # 'logs/prw/v26/coscale0.1-ratio30/Feb17_08-51-16',

    # 'logs/prw/v27/coscale0.1-ratio1-100/Feb17_13-52-27',
    # 'logs/prw/v27/coscale0.1-ratio10-100/Feb17_13-52-55',
    # 'logs/prw/v27/coscale0.1-ratio30-100/Feb17_13-53-18',

    # 'logs/prw/v27/coscale0.1-ratio10-200/Feb17_13-58-46',
    # 'logs/prw/v27/coscale0.1-ratio30-200/Feb17_13-59-19',
    # 'logs/prw/v27/coscale0.1-ratio50-200/Feb17_20-22-59',
    
    # 'logs/prw/v28/num_pos_idx_pow/Feb17_21-33-34',
    # 'logs/prw/v28/num_pos_idx/Feb17_20-49-53',
    # 'logs/prw/v28/num_pos_idx_pow-0.8/Feb18_11-36-43',
    # 'logs/prw/v28/num_pos_idx_pow-0.5/Feb17_20-51-21',
    # 'logs/prw/v28/num_pos_idx_pow-0.3/Feb18_11-36-18',
    # 'logs/prw/v28/num_pos_idx_pow-0.1/Feb18_11-37-08',
    # 'logs/prw/v28/num_pos_idx_pow-0.5-decay16/Feb18_14-05-36',
    
    # 'logs/prw/v29/focal-scale-pow-2/Feb18_14-58-45',
    # 'logs/prw/v29/focal-scale-pow-1/Feb18_14-57-58',
    # 'logs/prw/v29/focal-scale-pow-0.8/Feb18_14-59-26',
    # 'logs/prw/v29/focal-scale-pow-0.5/Feb18_14-59-04',

    # 'logs/prw/v29/focal-scale-pow-2-th0.8/Feb19_10-28-14',
    # 'logs/prw/v29/focal-scale-pow-2-th0.7/Feb19_10-28-42',
    # 'logs/prw/v29/focal-scale-pow-2-th0.5/Feb19_10-29-14',
    # 'logs/prw/v29/focal-scale-pow-2-th0.4/Feb19_10-29-56',
    # 'logs/prw/v29/focal-scale-pow-2-thLinear/Feb19_10-32-45',

    # 'logs/prw/v29/offset10-temp4/Feb19_23-33-45',
    # 'logs/prw/v29/offset10-temp2/Feb19_23-34-22',
    # 'logs/prw/v29/offset12-temp4/Feb19_23-35-58',
    # 'logs/prw/v29/offset12-temp2/Feb19_23-36-28',
    # 'logs/prw/v29/offset12-temp1/Feb19_23-37-33',

    'logs/prw/v30/bin_temp-0.5/Feb20_13-39-51',
    'logs/prw/v30/bin_temp-1.0/Feb20_13-40-11',
    'logs/prw/v30/bin_temp-2.0/Feb20_13-40-29',
    'logs/prw/v30/bin_temp-5.0/Feb20_13-40-51',
    'logs/prw/v30/bin_temp-10.0/Feb20_13-41-08',

    'logs/cuhk/v25/coscale0.1/Feb20_21-27-00',
    'logs/cuhk/v25/coscale0.1-decay18/Feb20_21-28-33',
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
