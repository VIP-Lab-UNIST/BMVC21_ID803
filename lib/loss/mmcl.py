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
            pos_logit = logit[pos_idx]
            neg_logit = logit[hn_idx]
            pos_logit = torch.mean(pos_logit, dim=0, keepdim=True)
            # neg_logit = torch.mean(neg_logit, dim=0, keepdim=True)
            results_logit = torch.cat((pos_logit, neg_logit), dim=0)
            results_logit = results_logit.unsqueeze(0)
            
            ## calculate the loss
            l = F.cross_entropy(5*results_logit, torch.zeros(1).long().cuda())  
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
