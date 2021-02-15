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
      
    def forward(self, logits, targets, multi_targets=None, coap_weights=None):
        
        if multi_targets is None:
            targets = targets.unsqueeze(1)
            multilabels = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, targets, True)
        else: 
            multilabels = multi_targets>0

        loss = []
        argidices = torch.argsort(logits.detach().clone(), dim=1, descending=True)
        neg_nums = self.r * (~multilabels).sum(dim=1).float()
        
        # indices = torch.arange(logits.shape[1]).cuda()

        for i, (logit, y, multilabel, argidx, neg_num) in enumerate(zip(logits, targets, multilabels, argidices, neg_nums)):
            
            ## positive, hard negative index
            
            if multi_targets is not None:
                multitarget = multi_targets[i]
                ep_idx = (multitarget==1).nonzero().squeeze(1)
                hp_idx = (multitarget==2).nonzero().squeeze(1)
                hn_idx = argidx[~multilabel[argidx]][:int(neg_num)]
                hard_neg_logit1 = logit[torch.cat((ep_idx, hn_idx))]
                results1 = hard_neg_logit1.unsqueeze(0).expand(len(ep_idx), -1)
            
                if len(hp_idx) >0 :
                    hard_neg_logit2 = logit[torch.cat((hp_idx, hn_idx))]
                    results2 = hard_neg_logit2.unsqueeze(0).expand(len(hp_idx), -1)
                    l_easy = F.cross_entropy(10*results1, torch.arange(len(ep_idx)).cuda())   
                    l_hard = F.cross_entropy(10*results2, torch.arange(len(hp_idx)).cuda())  
                    l = (l_easy + 3.0 * l_hard).sum()  
                else:
                    l = F.cross_entropy(10*results1, torch.arange(len(ep_idx)).cuda())
            else:
                pos_idx = multilabel.nonzero().squeeze(1)
                hn_idx = argidx[~multilabel[argidx]][:int(neg_num)]
                hard_neg_logit = logit[torch.cat((pos_idx, hn_idx))]
                results = hard_neg_logit.unsqueeze(0).expand(len(pos_idx), -1)
                l = F.cross_entropy(10*results, torch.arange(len(pos_idx)).cuda(),)   
            
            loss.append(l)
        loss = torch.mean(torch.stack(loss))
        return loss
