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

            ## positive index, hard negative index
            pos_idx = multilabel.nonzero().squeeze(1)
            # hn_idx = argidx[~multilabel[argidx]][:int(neg_num)]

            pos_logits = logit[pos_idx]
            pos_logits_sort, pos_logits_sort_idx = torch.sort(pos_logits, descending=True)

            for j, pos_logit_sort in enumerate(pos_logits_sort):

                hn_idx = argidx[~multilabel[argidx]][:(int(neg_num)-j)]
                hard_neg_logits = torch.cat( (pos_logits_sort[(len(pos_logits)-j):], logit[hn_idx]))
                
                results = torch.cat([pos_logit_sort.unsqueeze(0), hard_neg_logits[(j+1):]]).unsqueeze(0)
                l = F.cross_entropy(10*results, torch.zeros([1]).long().cuda())                 
                loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
