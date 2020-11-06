import torch
import numpy as np
from torch import nn

class MMCL(nn.Module):
    def __init__(self, delta=5.0, r=0.01):
        super(MMCL, self).__init__()
        self.delta = delta # coefficient for mmcl
        self.r = r         # hard negative mining ratio

    def forward(self, inputs, targets_, targets ,GT_MC, is_vec=False):
        m, n = inputs.size()

        if is_vec:
            multilabel = targets
        else:
            targets = torch.unsqueeze(targets, 1)
            multilabel = torch.zeros(inputs.size()).cuda()
            multilabel.scatter_(1, targets, float(1))

        # multi-label filtering with PS uniqueness
        # GT_cnt=torch.tensor(GT_MC[0]).cuda()
        # GT_label=torch.tensor(GT_MC[1]).cuda()
        # GT_imname=np.array(GT_MC[2])

        # for i, l_cnt in enumerate(targets_):
        #     imname=GT_imname[(GT_cnt==l_cnt).cpu().detach().numpy()]
        #     multilabel[i, GT_imname==imname]=0
        #     multilabel[i, GT_cnt==l_cnt]=1

        loss = []
        for i in range(m):
            logit = inputs[i]
            label = multilabel[i]

            pos_logit = torch.masked_select(logit, label > 0.5)
            neg_logit = torch.masked_select(logit, label < 0.5)

            _, idx = torch.sort(neg_logit.detach().clone(), descending=True)
            num = int(self.r * neg_logit.size(0))
            mask = torch.zeros(neg_logit.size(), dtype=torch.bool).cuda()
            mask[idx[0:num]] = 1
            hard_neg_logit = torch.masked_select(neg_logit, mask)

            l = self.delta * torch.mean((1-pos_logit).pow(2)) + torch.mean((1+hard_neg_logit).pow(2))
            # l = self.delta * torch.mean((1-pos_logit).pow(2)) + torch.mean((1+neg_logit).pow(2))
            loss.append(l)

        loss = torch.mean(torch.stack(loss))
        return loss