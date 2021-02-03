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
      
    def forward(self, logits, targets, multi_targets=None, neg_idices=None):
        
        if multi_targets is None:
            targets = targets.unsqueeze(1)
            multilabels = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, targets, True)
        else: 
            multilabels = multi_targets.bool()

        loss = []
        argidices = torch.argsort(logits.detach().clone(), dim=1, descending=True)
        neg_nums = self.r * (~multilabels).sum(dim=1).float()
        
        indices = torch.arange(logits.shape[1]).cuda()

        for i, (logit, y, multilabel, argidx, neg_num) in enumerate(zip(logits, targets, multilabels, argidices, neg_nums)):
            
            ## positive, hard negative index
            pos_idx = multilabel.nonzero().squeeze(1)
            hn_idx = argidx[~multilabel[argidx]][:int(neg_num)]
            

            hard_neg_logit = logit[torch.cat((pos_idx, hn_idx))]

            results = hard_neg_logit.unsqueeze(0).expand(len(pos_idx), -1)
            
            ## calculate the loss
            if results.shape[0] == 1:
                l = F.cross_entropy(10*results, torch.arange(len(pos_idx)).cuda())                 
            else:
                lambda_ = 0.1
                l = F.cross_entropy(10*results, torch.arange(len(pos_idx)).cuda(), reduction='none') 
                weight = torch.zeros(len(results)).cuda().fill_((1 - lambda_)  / (len(results)-1))
                weight[indices[pos_idx] == y] = lambda_
                l = (l * weight).sum()

            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
