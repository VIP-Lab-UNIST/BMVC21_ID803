import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class MMCL(nn.Module):
    def __init__(self, delta, r):
        super(MMCL, self).__init__()
        self.delta = delta # coefficient for mmcl
        self.r = r         # hard negative mining ratio

    def forward(self, inputs, targets, neg_idices=None):
        m, n = inputs.size()

        if type(neg_idices) == type(None):
            targets = torch.unsqueeze(targets, 1)
            multilabels = torch.zeros(inputs.size()).cuda()
            multilabels.scatter_(1, targets, float(1))
        else:
            multilabels = targets

        loss = []
        for i, (logit, multilabel) in enumerate(zip(inputs, multilabels)):
            pos_logit = logit[multilabel > 0.5]
            num_pos = len(pos_logit)

            if type(neg_idices) == type(None): 
                neg_logit = logit[multilabel < 0.5]  
                _, idx = torch.sort(neg_logit.detach().clone(), descending=True)
                num = int(self.r * neg_logit.size(0))
                mask = torch.zeros(neg_logit.size(), dtype=torch.bool).cuda()
                mask[idx[0:num]] = 1
                hard_neg_logit = torch.masked_select(neg_logit, mask)    
            else: 
                hard_neg_logit = logit[neg_idices[i].bool()]
                
            # l = self.delta * torch.mean((1-pos_logit).pow(2)) + torch.mean((1+hard_neg_logit).pow(2))
            
            results = torch.cat([pos_logit.unsqueeze(1), hard_neg_logit.unsqueeze(0).expand(num_pos, -1)], dim=1)
            l = F.cross_entropy(10*results, torch.zeros(num_pos).long().cuda())
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss