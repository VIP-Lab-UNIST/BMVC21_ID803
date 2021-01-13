import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import average_precision_score, precision_recall_curve

import cv2
import torch
from .misc import ship_data_to_cuda
import json

###
import torch
import torchvision

import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops import boxes as box_ops

from torchvision.ops import roi_align

from torchvision.models.detection import _utils as det_utils
# from lib.model.roi_heads import _utils as det_utils
# from . import _utils as det_utils

from torch.jit.annotations import Optional, List, Dict, Tuple
###

def add_gt_proposals(proposals, gt_boxes, scores):
    # type: (List[Tensor], List[Tensor]) -> List[Tensor]
    proposals_list=[]
    scores_list=[]
    for proposal, score, gt_box in zip(proposals, scores, gt_boxes):
        proposals_list.append(torch.cat((proposal, gt_box)))
        scores_list.append(torch.cat((score, torch.ones(len(gt_box)).cuda())))

    return proposals_list, scores_list

# def add_gt_proposals(proposals, gt_boxes):
#     # type: (List[Tensor], List[Tensor]) -> List[Tensor]
#     proposals = [
#         torch.cat((proposal, gt_box))
#         for proposal, gt_box in zip(proposals, gt_boxes)
#     ]

#     return proposals



def assign_targets_to_proposals(proposals, gt_boxes, gt_labels, proposal_matcher):
    # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    matched_idxs = []
    labels = []
    for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):

        if gt_boxes_in_image.numel() == 0:
            # Background image
            device = proposals_in_image.device
            clamped_matched_idxs_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
            labels_in_image = torch.zeros(
                (proposals_in_image.shape[0],), dtype=torch.int64, device=device
            )
        else:
            #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
            match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

        matched_idxs.append(clamped_matched_idxs_in_image)
        labels.append(labels_in_image)

    return matched_idxs, labels

## Detect
@torch.no_grad()
def draw(model, data_loader, device):

    proposal_matcher = det_utils.Matcher(
            0.5,
            0.1,
            allow_low_quality_matches=False)

    im_names, all_boxes, all_feats = [], [], []
    cnt = 0
    matching={}
    matcing_bbox ={}
    for data in tqdm(data_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        # Target is not used in inference mode.
        outputs = model(0, images)

        # assign label to detected bbox
        gt_boxes = [t["boxes"].cuda() for t in targets]
        gt_labels = [t["labels"] for t in targets]
        proposals = [o['boxes'].cuda() for o in outputs]
        scores = [o['scores'].cuda() for o in outputs]

        proposals, scores = add_gt_proposals(proposals, gt_boxes, scores)
        _, labels = assign_targets_to_proposals(proposals, gt_boxes, gt_labels, proposal_matcher)
        
        for t, bbox, label, score in zip(targets, proposals, labels, scores):
            
            ## for test
            # matcing_bbox[t['im_name']] = [bbox.cpu().numpy().tolist(), label.cpu().numpy().tolist(), score.cpu().numpy().tolist()]

            # ## for training
            scene = cv2.imread('../datasets/PRW-v16.04.20/frames/'+t['im_name'])
            # scene = cv2.imread('../datasets/CUHK-SYSU/Image/SSM/'+t['im_name'])
            camid =  int(t['im_name'].split('_')[0][1])
            
            for b, l in zip(bbox, label):
                
                if l==5555: l=-2
                if (l==-1) or (l==0): continue

                x1, y1, x2, y2 = b
                person = scene[int(y1):int(y2), int(x1):int(x2), :]
                # scene = cv2.rectangle(scene, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

                cv2.imwrite('../datasets/PRW/train/{:03d}_{:02d}_{:06d}.jpg'.format(l, camid, cnt), person)
                # cv2.imwrite('../datasets/CUHK/train/{:03d}_{:02d}_{:06d}.jpg'.format(l, camid, cnt), person)
                matching['{:03d}_{:02d}_{:06d}.jpg'.format(l, camid, cnt)] = t['im_name']
                cnt += 1
        
    # with open('../datasets/PRW/gallery_bbox.json', "w") as fp:
    # with open('../datasets/CUHK/gallery_bbox.json', "w") as fp:
        # json.dump(matcing_bbox, fp)

    with open('../datasets/PRW/train_matching.json', "w") as fp:
    # with open('../datasets/CUHK/train_matching.json', "w") as fp:
        json.dump(matching, fp)
    return None

## GT
# @torch.no_grad()
# def draw(model, data_loader, device):
#     model.eval()

#     im_names, all_boxes, all_feats = [], [], []
#     cnt = 0
#     matching={}
#     for data in tqdm(data_loader, ncols=0):
#         images, targets = ship_data_to_cuda(data, device)
#         # Target is not used in inference mode.
#         outputs = model(0, images)

#         for o, t in zip(outputs, targets):
#             scene = cv2.imread('../datasets/PRW-v16.04.20/frames/'+t['im_name'])
#             for bbox in o['boxes']:
                
#                 x1, y1, x2, y2 = bbox
#                 person = scene[int(y1):int(y2), int(x1):int(x2), :]
#                 # scene = cv2.rectangle(scene, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

#                 # cv2.imwrite('../datasets/PRW_MCformat/gallery_detect/000_00_{:06d}.jpg'.format(cnt), person)
#                 cv2.imwrite('../datasets/PRW_MCformat/gallery_detect/000_00_{:06d}.jpg'.format(cnt), person)
#                 matching['000_00_{:06d}.jpg'.format(cnt)] = t['im_name']
#                 cnt += 1
                
#             # cv2.imwrite('../datasets/PRW_MCformat/gallery_detect/{:s}.jpg'.format(t['im_name']), scene)
#     with open('../datasets/PRW_MCformat/gallery_detect_matching.json', "w") as fp:
#         json.dump(matching, fp)
#     return None


@torch.no_grad()
def inference(model, gallery_loader, probe_loader, device):
    model.eval()
    cpu = torch.device('cpu')

    im_names, all_boxes, all_feats = [], [], []
    for data in tqdm(gallery_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        # Target is not used in inference mode.
        outputs = model(0, images)

        for o, t in zip(outputs, targets):
            im_names.append(t['im_name'])
            box_w_scores = torch.cat([o['boxes'],
                                      o['scores'].unsqueeze(1)],
                                     dim=1)
            all_boxes.append(box_w_scores.to(cpu).numpy())
            all_feats.append(o['embeddings'].to(cpu).numpy())

    probe_feats = []
    for data in tqdm(probe_loader, ncols=0):
        images, targets = ship_data_to_cuda(data, device)
        embeddings = model.ex_feat(images, targets, mode='det')
        for em in embeddings:
            probe_feats.append(em.to(cpu).numpy())

    name_to_boxes = OrderedDict(zip(im_names, all_boxes))

    return name_to_boxes, all_feats, probe_feats

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + \
        (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def detection_performance_calc(dataset, gallery_det, det_thresh=0.5, iou_thresh=0.5,
                               labeled_only=False):
    """
    gallery_det (list of ndarray): n_det x [x1, y1, x2, y2, score] per image

    det_thresh (float): filter out gallery detections whose scores below this
    iou_thresh (float): treat as true positive if IoU is above this threshold
    labeled_only (bool): filter out unlabeled background people
    """
    assert len(dataset) == len(gallery_det)
    gt_roidb = dataset.record

    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for gt, det in zip(gt_roidb, gallery_det):
        gt_boxes = gt['boxes']
        if labeled_only:
            inds = np.where(gt['gt_pids'].ravel() > 0)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        if det != []:
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= det_thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
        else:
            num_det = 0
        if num_det == 0:
            count_gt += num_gt
            continue
        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
        tfmat = (ious >= iou_thresh)
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            if tfmat[:, j].any():
                y_true.append(True)
            else:
                y_true.append(False)
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / count_gt
    ap = average_precision_score(y_true, y_score) * det_rate
    precision, recall, __ = precision_recall_curve(y_true, y_score)
    recall *= det_rate

    print('{} detection:'.format('labeled only' if labeled_only else
                                 'all'))
    print('  recall = {:.2%}'.format(det_rate))
    if not labeled_only:
        print('  ap = {:.2%}'.format(ap))
    return ap, det_rate
    # return precision, recall
