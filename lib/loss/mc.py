from __future__ import absolute_import
import warnings
warnings.filterwarnings("ignore")

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

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar,
                 oim_alpha, oim_beta, GT_MC=None):
        super(MCLoss, self).__init__()
        self.num_features = num_features

        self.memory=Memory(self.num_features, 18048)
        # self.memory=Memory(self.num_features, 17473)
        self.labelpred = MPLP(0.6)
        self.criterion = MMCL(5, 0.01)

        self.GT_MC=GT_MC

    def forward(self, epoch, inputs, roi_label, cls_scores, images, proposals, GT_info):

        image_tensors=images.tensors

        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        inputs = inputs * cls_scores

        inputs=inputs[label>0]
        label=label[label>0]

        logits = self.memory(inputs, label, epoch)

        # MC
        # if epoch > 5:
        #     multilabel = self.labelpred.predict(self.memory.mem.detach().clone(), label.detach().clone())
        #     loss = self.criterion(logits, label, multilabel, self.GT_MC, True)
        # else:
        #     loss = self.criterion(logits, label, label, self.GT_MC)

        # No MC
        # multilabel=torch.zeros(len(label) ,18048).cuda()
        # for i, l_cnt in enumerate(label): multilabel[i, l_cnt]=1
        # loss = self.criterion(logits, label, multilabel, self.GT_MC, True)

        # GT
        multilabel=torch.zeros(len(label) ,18048).cuda()
        GT_cnt=torch.tensor(self.GT_MC[0]).cuda()
        GT_label=torch.tensor(self.GT_MC[1]).cuda()

        for i, l_cnt in enumerate(label):

            l=GT_label[l_cnt==GT_cnt]
            # only label
            if l != -2: multilabel[i, GT_cnt[GT_label==l]]=1
            else: multilabel[i, l_cnt]=1

        loss = self.criterion(logits, label, multilabel, self.GT_MC, True)

        
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

# def imageTensor2Numpy(image_batch):

#     image_numpy = image_batch.detach().cpu().numpy().transpose(1,2,0).copy()
#     image_numpy[:,:,0] = image_numpy[:,:,0]*0.229+0.485
#     image_numpy[:,:,1] = image_numpy[:,:,1]*0.224+0.456
#     image_numpy[:,:,2] = image_numpy[:,:,2]*0.225+0.406
#     image_numpy = image_numpy[:,:,[2,1,0]]
#     image_numpy = (image_numpy*255).astype(np.uint8)

#     return image_numpy

# def draw_bbox(images, proposals, labels, GT_info):
#     image_tensors=images.tensors.detach()
#     labels=labels.reshape(len(proposals), -1).detach()
#     for i, (image_tensors_, proposals_, labels_, GT_info_)  in enumerate(zip(image_tensors, proposals, labels, GT_info)):
#         img_array=imageTensor2Numpy(image_tensors_)
#         img_array=np.ascontiguousarray(img_array, dtype=np.uint8)
#         for proposal, label in zip(proposals_, labels_):
#             if label > 482:
#                 folder_path='./logs/prw/bjhan/draw_05_area12070/'+str(label.item())+'/'
#                 # folder_path='./logs/prw/bjhan/draw_04_cls07_area12070/'+str(label.item())+'/'
#                 if not os.path.isdir(folder_path): os.makedirs(folder_path)
#                 person = img_array[int(proposal[1]):int(proposal[3]), int(proposal[0]):int(proposal[2]), :]
#                 cv2.imwrite(folder_path+'out_%s'%str(GT_info_['im_name']), person)
#                 # print(proposal[3]-proposal[1])
#                 # print(proposal[2]-proposal[0])
#                 # print(str(label.item()))
