from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd

import cv2
import numpy as np
import os


class OIM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets,  
                lut, cq):
        ctx.save_for_backward(inputs, targets,
                              lut, cq)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, \
            lut, cq = ctx.saved_tensors

        momentum = 0.5

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(
                torch.cat([lut, cq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)

        for x, y in zip(inputs, targets):
            if (0 <= y) and (y < len(lut)):
                lut[y] = momentum *  \
                    lut[y] + (1. - momentum)  * x
                lut[y] /= lut[y].norm()
            elif y >= len(lut):
                cq[y - len(lut)] = momentum *  \
                    cq[y - len(lut)] + (1. - momentum)  * x
                cq[y - len(lut)] /= cq[y - len(lut)].norm()
           

        return grad_inputs, None, None, None


def oim(inputs, targets,  lut, cq):
    return OIM.apply(inputs, targets, lut, cq)



class OIMLoss(nn.Module):
    """docstring for OIMLoss"""

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar,
                 oim_alpha, oim_beta):
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

        self.register_buffer('header_cq',  torch.zeros(1).fill_(-1).long())
        self.register_buffer('oim_alpha',  torch.zeros(1).fill_(oim_alpha))
        self.register_buffer('oim_beta',  torch.zeros(1).fill_(oim_beta))
        print(self.oim_alpha)
        print(self.oim_beta)

    def forward(self, inputs, roi_label, cls_scores, images, proposals, GT_info):

        image_tensors=images.tensors
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inputs = inputs * cls_scores
        label = self.pseudo_labeling(inputs.detach().clone(), label, proposals, cls_scores.detach().clone())
        
        projected = oim(inputs, label, self.lut, self.cq)
        projected *= self.oim_scalar

        label_ = label.detach().clone()
        loss_oim = F.cross_entropy(projected, label_,
                                   ignore_index=-1)
        
        # draw
        # draw_bbox(images, proposals, label_, GT_info)

        return loss_oim


    def pseudo_labeling(self, inputs, targets, proposals, cls_scores):
        
        # filter small bboxes out
        # targets=targets.reshape(len(proposals), -1)
        # cls_scores=cls_scores.reshape(len(proposals), -1)
        # for i, (proposals_, targets_, cls_scores_) in enumerate(zip(proposals, targets, cls_scores)):
        #     for j, (proposal, target, cls_score) in enumerate(zip(proposals_, targets_, cls_scores_)):
        #         width=proposal[3]-proposal[1]
        #         height=proposal[2]-proposal[0]
        #         area=width*height
        #         if area < 60*35: targets[i,j]=-1
        #         # if cls_score < 0.7: targets[i,j]=-1
        #         # if cls_score < 0.6: targets[i,j]=-1
        # targets=targets.reshape(-1)

        # unlabeled person list
        unlabels = targets[targets >= len(self.lut)].unique()
        if len(unlabels) !=0:
            if self.header_cq[0] == -1: 
                # when first time, assign new pseudo label
                self.header_cq[0] = 0
                for unlabel in unlabels:
                    if unlabel > 0:
                        # Ignore background
                        targets[targets==unlabel] = len(self.lut)+(self.header_cq[0]%len(self.cq))
                        self.header_cq[0] = self.header_cq[0] + 1 

            else:
                fill_in = min(len(self.cq), self.header_cq[0])
                for unlabel in unlabels:
                    if unlabel > 0:
                        # Ignore background
                        # one unlabeled person
                        feats = inputs[targets==unlabel]
                        unlabel_outputs = feats.mm(self.cq.t())
                        highest_sim, highest_arg = torch.max(unlabel_outputs[:,:fill_in], dim=0)
                        lowest_sim, _ = torch.min(unlabel_outputs[:,:fill_in], dim=0)
                        
                        # pseudo labeling by lowest_sim of one unlabeled person
                        max_lowest_value, max_table_index = torch.max(lowest_sim, dim=0)
                        min_histest_value, min_table_index = torch.min(highest_sim, dim=0)

                        if  max_lowest_value >= self.oim_alpha.item() :
                            targets[targets==unlabel]=len(self.lut)+max_table_index.item()
                            # print('assign %s'%str(len(self.lut)+max_table_index.item()))

                        # assign new pseudo label
                        elif min_histest_value < self.oim_beta.item():    
                            targets[targets==unlabel] = len(self.lut)+(self.header_cq[0]%len(self.cq))
                            self.cq[self.header_cq[0]%len(self.cq)] = feats[highest_arg[min_table_index]]
                            self.header_cq[0] = self.header_cq[0] + 1 
                            # print('new %s'%str(len(self.lut)+self.header_cq[0].item()-1))
                            # if (self.header_cq[0]%len(self.cq)) == len(self.cq)-1: raise ValueError('No assign anymore')
                        else: targets[targets==unlabel] = -1
                    
        return targets.detach().clone()


def imageTensor2Numpy(image_batch):

    image_numpy = image_batch.detach().cpu().numpy().transpose(1,2,0).copy()
    image_numpy[:,:,0] = image_numpy[:,:,0]*0.229+0.485
    image_numpy[:,:,1] = image_numpy[:,:,1]*0.224+0.456
    image_numpy[:,:,2] = image_numpy[:,:,2]*0.225+0.406
    image_numpy = image_numpy[:,:,[2,1,0]]
    image_numpy = (image_numpy*255).astype(np.uint8)

    return image_numpy

def draw_bbox(images, proposals, labels, GT_info):
    image_tensors=images.tensors.detach()
    labels=labels.reshape(len(proposals), -1).detach()
    for i, (image_tensors_, proposals_, labels_, GT_info_)  in enumerate(zip(image_tensors, proposals, labels, GT_info)):
        img_array=imageTensor2Numpy(image_tensors_)
        img_array=np.ascontiguousarray(img_array, dtype=np.uint8)
        for proposal, label in zip(proposals_, labels_):
            if label > 482:
                folder_path='./logs/prw/bjhan/draw_05_area12070/'+str(label.item())+'/'
                # folder_path='./logs/prw/bjhan/draw_04_cls07_area12070/'+str(label.item())+'/'
                if not os.path.isdir(folder_path): os.makedirs(folder_path)
                person = img_array[int(proposal[1]):int(proposal[3]), int(proposal[0]):int(proposal[2]), :]
                cv2.imwrite(folder_path+'out_%s'%str(GT_info_['im_name']), person)
                # print(proposal[3]-proposal[1])
                # print(proposal[2]-proposal[0])
                # print(str(label.item()))
