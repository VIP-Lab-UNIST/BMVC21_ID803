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
      
    def forward(self, logits, targets, co_cnts, vector=None):
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
        for logit, multilabel, argidx, neg_num, co_cnt in zip(logits, multilabels, argidices, neg_nums, co_cnts):
            # 1. Sampling postivie pairs
            pos_logit = logit[multilabel]
            pos_cnt = co_cnt[multilabel]

            # 2. Sampling hard negative pairs
            hard_neg_logit = logit[argidx[~multilabel][:int(neg_num)]]

            # 3. Compute classification loss 
            l_pos=F.binary_cross_entropy_with_logits(pos_logit, torch.ones(pos_logit.shape).cuda(), weight=pos_cnt)
            l_neg=F.binary_cross_entropy_with_logits(hard_neg_logit, torch.zeros(hard_neg_logit.shape).cuda())
            l = self.delta * l_pos + l_neg            

            # l = self.delta * torch.mean((1-pos_logit).pow(2)) \
                            # + torch.mean((1+hard_neg_logit).pow(2))
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
