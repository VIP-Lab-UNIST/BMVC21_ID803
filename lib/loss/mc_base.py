from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd

import cv2
import numpy as np
import os

# multi-class loss
class MCLoss(nn.Module):
    """docstring for OIMLoss"""

    def __init__(self, num_features, num_pids, num_cq_size,
                 oim_momentum, oim_scalar,
                 oim_alpha, oim_beta):
        super(MCLoss, self).__init__()
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

    # For one image loss
    def forward(self, epoch, inputs, roi_label, cls_scores, images, proposals, GT_info):

        image_tensors=images.tensors

        num_feat=roi_label[0].shape[0]
        inputs=inputs * cls_scores
        inputs=[inputs[(num_feat*i):(num_feat*(i+1)),:] for i in range(len(roi_label))]

        label = list(map(lambda x: x-1, roi_label))

        loss=self.oi_forward(inputs, label, GT_info)

        return loss

    ## one image forword
    def oi_forward(self, inputs, roi_labels, GT_info):
        
        for inputs_, roi_labels_, GT_info_ in zip(inputs, roi_labels, GT_info):
            inputs_=inputs_[roi_labels_>=0]
            roi_labels_=roi_labels_[roi_labels_>=0]
            if len(roi_labels_)!=0:
                ## define sorted unique, squeeze roi_labels, images_ids
                uq_roi_labels=torch.unique(roi_labels_).cuda()

                ## define table and label(row: the number of person, column: the number of roi)
                self.mc = torch.zeros(len(uq_roi_labels), self.num_features).cuda()
                self.mc_projected = torch.zeros(len(uq_roi_labels), len(roi_labels_)).cuda()

                ## self.mc: Fill the avg.feature(normalized) into table
                for i, label in enumerate(uq_roi_labels): self.mc[i]=torch.mean(inputs_[label==roi_labels_], 0); self.mc[i] /= self.mc[i].norm(); 
                self.mc=self.mc.detach().clone()

                ## self.mc_projected: Calculate similarity 
                for i, feature in enumerate(inputs_): self.mc_projected[:,i]=feature.unsqueeze(0).mm(self.mc.cuda().t()).squeeze()
            
                # calculate loss of one image
                roi_losses=[]
                for i, (feature, label) in enumerate(zip(inputs_, roi_labels_)):
                    oi_labels=torch.zeros(size=uq_roi_labels.shape)
                    oi_labels[uq_roi_labels==label]=1

                    oi_projected=self.mc_projected[:,i].unsqueeze(0)
                    oi_label=oi_labels.nonzero()[0].cuda()

                    roi_loss=F.cross_entropy(oi_projected, oi_label)
                    roi_losses.append(roi_loss)
            
            else: roi_losses.append(0)

        avg_roi_losses=sum(roi_losses)/len(roi_losses)
    
        return avg_roi_losses

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
