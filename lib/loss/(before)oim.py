from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,  
                lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets,
                              lut, cq, header, momentum)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, \
            lut, cq, header, momentum = ctx.saved_tensors


        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if (0 <= y) and (y < len(lut)):
                lut[y] = momentum * lut[y] + x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets,
                     lut, cq,
                     torch.tensor(header), torch.tensor(momentum))



class OIMLoss(nn.Module):
    """docstring for OIMLoss"""

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer('lut', torch.zeros(
            self.num_pids, self.num_features))
        self.register_buffer('cq',  torch.zeros(
            self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def forward(self, inputs, roi_label, cls_scores):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        # inputs = inputs * cls_scores
        inputs = inputs[label<0]
        label = label[label<0]
        
        projected = oim(inputs, label,
                        self.lut,
                        self.cq, 
                        self.header_cq, momentum=self.momentum)
        projected = projected * self.oim_scalar

        self.header_cq = ((self.header_cq +
                           (label >= self.num_pids).long().sum().item()) %
                          self.num_unlabeled)

        label_ = label.detach().clone()
        label_[label_<0] = 5554

        loss_oim = F.cross_entropy(projected, label_.detach(),
                                   ignore_index=5554)

        return loss_oim



