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
            # rand_idx = argidx[~multilabel][:int(neg_num)]
            hn_idx = argidx[~multilabel[argidx]][:int(neg_num)]
            pos_idx = multilabel.nonzero().squeeze(1)
            # neg_idx = rand_idx[~(rand_idx.unsqueeze(1)==pos_idx).any(-1)]

            # hard_neg_logit = logit[hn_idx]
            # hard_neg_logit = logit[torch.unique(torch.cat((pos_idx, rand_idx)))]
            hard_neg_logit = logit[torch.cat((pos_idx, hn_idx))]
            # hard_neg_logit = logit[argidx][~multilabel[argidx]][:int(neg_num)]] 
            
            results = torch.cat([pos_logit.unsqueeze(1), 
                                hard_neg_logit.unsqueeze(0).expand(num_pos, -1)], dim=1)

            l = F.cross_entropy(10*results, torch.zeros(num_pos).long().cuda())                 
            # l += torch.log(torch.tensor(1.+num_pos))
            # # 3. Compute classification loss 
            # l = self.delta * torch.mean((1-pos_logit).pow(2)) \
            #                 + torch.mean((1+hard_neg_logit).pow(2))
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
