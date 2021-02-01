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
            hn_idx = argidx[~multilabel[argidx]][:int(neg_num)]
            num_pos = len(pos_idx)

            pos_logits = logit[pos_idx]
            pos_logits_sort, pos_logits_sort_idx = torch.sort(pos_logits, descending=True)

            # hard_neg_logits = logit[torch.cat((pos_idx[pos_logits_sort_idx], hn_idx))]
            hard_neg_logit = logit[hn_idx]

            pos_logits_sort_mat = pos_logits_sort.repeat(num_pos, 1)
            pos_mask = torch.tril(torch.ones(num_pos, num_pos)).flip(dims=[0]).bool().detach()
            pos_logits_sort_mat[pos_mask] = pos_logits_sort.unsqueeze(1).repeat(1, num_pos)[pos_mask].detach()

            results = torch.cat([pos_logits_sort_mat, hard_neg_logit.unsqueeze(0).expand(num_pos, -1)], dim=1)
            # results = torch.cat([pos_logits_sort.unsqueeze(1), 
            #                     pos_logits_sort_mat], dim=1)

            l = F.cross_entropy(10*results, torch.zeros(num_pos).long().cuda())                 
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss
