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

    def __init__(self, use_hnm, use_hpm, hard_neg, sim_thrd, co_scale, num_features):
        super(MCLoss, self).__init__()
        self.use_hnm = use_hnm
        self.use_hpm = use_hpm
        self.hard_neg = hard_neg
        self.sim_thrd = sim_thrd
        self.co_scale = co_scale
        self.num_features = num_features
        
    def set_scene_vector(self, train_info):
        num_person=len(train_info[3])
        num_scene=list(train_info[3])
        name_scene=train_info[2]

        self.name_scene=np.array(name_scene)
        self.num_scene=torch.tensor(list(map(lambda x: x-1, num_scene))).cuda()
        self.memory=Memory(self.num_features, num_person).cuda()
        
        self.labelpred = MPLP(use_hnm=self.use_hnm, use_hpm=self.use_hpm, \
                                total_scene=self.num_scene, threshold=self.sim_thrd, coapp_scale= self.co_scale)

        self.criterion = MMCL(delta=5.0, r=self.hard_neg)

    def forward(self, epoch, inputs, cls_scores, roi_labels, scene_nums, GT_roi_labels, scene_names, images, proposals):

        image_tensors=images.tensors

        # merge into one batch, background label = 0
        targets = torch.cat(roi_labels)
        proposals = torch.cat(proposals)
        scene_nums=torch.cat(scene_nums).cuda()
        scene_names= np.concatenate(scene_names)
        label = targets - 1  # background label = -1
        scene_nums = scene_nums -1

        mask = (label>=0)
        inputs=inputs[mask]
        cls_scores=cls_scores[mask]
        proposals=proposals[mask]
        scene_nums=scene_nums[mask]
        scene_names=scene_names[mask.clone().detach().cpu().numpy()]
        label=label[mask]

        logits = self.memory(inputs, label, epoch)

        # MC
        # if epoch > -1:
        if epoch > 4:
            multilabels = self.labelpred.predict(self.memory.mem.detach().clone(), label.detach().clone())
            loss = self.criterion(logits, label, multilabels)
            
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

        ## For training
        self.mem = nn.Parameter(torch.zeros(num_classes, num_features), requires_grad=False)

        ## For debuging
        # tmp = torch.randn(num_classes, num_features)/256 + 1.0/16.0
        # tmp /= tmp.norm(dim=1, keepdim=True)
        # self.mem = nn.Parameter(tmp, requires_grad=False)

    def forward(self, inputs, targets, epoch=None):
        # alpha = 0.5 * epoch / 60
        logits = MemoryLayer(self.mem, alpha=0.5)(inputs, targets)

        return logits
