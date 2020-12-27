from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.autograd import Function

import cv2
import numpy as np
import os

from .mmcl import MMCL
from .mplp import MPLP

class MCLoss(nn.Module):
    """docstring for MCLoss"""

    def __init__(self, num_features):
        super(MCLoss, self).__init__()
        self.num_features = num_features

    def set_scene_vector(self, train_info):
        num_person=len(train_info[3])
        num_scene=list(train_info[3])
        self.num_scene=list(map(lambda x: x-1, num_scene))
        self.memory=Memory(self.num_features, num_person).cuda()
        self.labelpred = MPLP(0.6)
        self.criterion = MMCL(5, 0.01)

    def forward(self, epoch, inputs, cls_scores, roi_labels, scene_nums, GT_roi_labels, scene_names, images, proposals):

        image_tensors=images.tensors
        # merge into one batch, background label = 0
        targets = torch.cat(roi_labels)
        proposals = torch.cat(proposals)
        scene_nums=torch.cat(scene_nums).cuda()

        scene_names= np.concatenate(scene_names)
        label = targets - 1  # background label = -1
        # inputs = inputs * cls_scores

        inputs=inputs[label>0]
        proposals=proposals[label>0]
        scene_nums=scene_nums[label>0]
        scene_names=scene_names[(label>0).clone().detach().cpu().numpy()]
        label=label[label>0]
        
        # draw_proposal(scene_names, proposals, label)
        logits = self.memory(inputs, label, epoch)

        # MC
        if epoch > 5:
            multilabel = self.labelpred.predict(self.memory.mem.detach().clone(), label.detach().clone())
            loss = self.criterion(logits, multilabel, True)
        else:
            loss = self.criterion(logits, label)
        
        return loss

class MemoryLayer(Function):
    def __init__(self, memory, alpha=0.01):
        super(MemoryLayer, self).__init__()
        self.memory = memory
        self.alpha = alpha

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.memory.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.memory)
        for x, y in zip(inputs, targets):
            self.memory[y] = self.alpha * self.memory[y] + (1. - self.alpha) * x
            self.memory[y] /= self.memory[y].norm()
        return grad_inputs, None

class Memory(nn.Module):
    def __init__(self, num_features, num_classes, alpha=0.01):
        super(Memory, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = alpha

        self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)
    
    def forward(self, inputs, targets, epoch=None):
        alpha = 0.5 * epoch / 60
        logits = MemoryLayer(self.mem, alpha=alpha)(inputs, targets)

        return logits


def draw_proposal(scene_names, proposals, labels):
    for i, (scene_name, proposal, label) in enumerate(zip(scene_names, proposals, labels)):
        scene=Image.open('/root/workspace/Personsearch/datasets/PRW-v16.04.20/frames/'+scene_name)
        x,y,x2,y2 =proposal
        roi=scene.crop((x, y, x2, y2))
        roi.save('./roi_%d.jpg'%i)
