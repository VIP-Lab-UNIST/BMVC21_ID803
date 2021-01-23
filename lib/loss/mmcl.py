import torch
import numpy as np
from torch import nn
import timeit
import torch.nn.functional as F

class MMCL(nn.Module):
    def __init__(self, delta=5.0, r=0.01):
        super(MMCL, self).__init__()
        self.delta = delta # coefficient for mmcl
        self.r = r         # hard negative mining ratio
      
    def forward(self, logits, targets, neg_idices=None):
        if len(targets.shape)==2:
            multilabels = targets.bool()
        elif len(targets.shape)==1:
            targets = targets.unsqueeze(1)
            multilabels = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, targets, True)
        else: 
            raise ValueError('Incorrect input size')
    
        loss = []
        argidices = torch.argsort(logits.detach().clone(), dim=1, descending=True)
        neg_nums = self.r * (~multilabels).sum(dim=1).float()
        for i, (logit, multilabel, argidx, neg_num) in enumerate(zip(logits, multilabels, argidices, neg_nums)):
            # 1. Sampling postivie pairs
            pos_logit = logit[multilabel]
            num_pos = len(pos_logit)
            # 2. Sampling hard negative pairs
            hard_neg_logit = logit[argidx[~multilabel][:int(neg_num)]]
            # print(argidx[~multilabel][:int(neg_num)].sort(descending=False)[0])
            # print(argidx[~multilabel][:int(neg_num)].shape)
            # print(multilabel.nonzero().squeeze().sort(descending=False)[0])
            # print(multilabel.nonzero().squeeze().shape)
            # print(np.intersect1d(argidx[~multilabel][:int(neg_num)].detach().cpu().numpy(), multilabel.nonzero().squeeze().detach().cpu().numpy()).shape) 
            # print('---------')


            # neg_logit = logit[~multilabel]  
            # _, idx = torch.sort(neg_logit.detach().clone(), descending=True)
            # num = int(self.r * neg_logit.size(0))
            # mask = torch.zeros(neg_logit.size(), dtype=torch.bool).cuda()
            # mask[idx[0:num]] = 1
            # hard_neg_logit = torch.masked_select(neg_logit, mask)
            # print(mask.nonzero().squeeze().shape)
            # print(torch.sort(mask.nonzero().squeeze(), descending=False)[0])
            # print(torch.sort(hard_neg_logit, descending=False)[0])
            # print(neg_logit.sort(descending=True)[0])
            # print(np.intersect1d(argidx[~multilabel][:int(neg_num)].detach().cpu().numpy(), mask.nonzero().squeeze().detach().cpu().numpy()).shape) 
            # raise ValueError

            # if type(neg_idices) == type(None):
            #     hard_neg_logit = logit[argidx[~multilabel][:int(neg_num)]]
            # else:
            #     hard_neg_logit = logit[neg_idices[i].bool()]
            
            
            results = torch.cat([pos_logit.unsqueeze(1), 
                                hard_neg_logit.unsqueeze(0).expand(num_pos, -1)], dim=1)

            l = F.cross_entropy(10*results, torch.zeros(num_pos).long().cuda())                 

            # # 3. Compute classification loss 
            # l = self.delta * torch.mean((1-pos_logit).pow(2)) \
            #                 + torch.mean((1+hard_neg_logit).pow(2))
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
