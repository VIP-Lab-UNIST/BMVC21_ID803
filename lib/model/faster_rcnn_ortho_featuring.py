from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from .generalized_rcnn import GeneralizedRCNN
# from .roi_heads import RoIHeads
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .resnet_backbone import resnet_backbone

from ..loss import MCLoss
# from ..loss import OIMLoss
from torch import autograd
import math

class FasterRCNN(GeneralizedRCNN):
    """
    See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L26
    """
    def __init__(self, backbone,
                 num_classes=None, 
                 # transform parameters
                 min_size=900, max_size=1500,
                 image_mean=[0.485, 0.456, 0.406], 
                 image_std=[0.229, 0.224, 0.225],
                 # RPN parameters
                 rpn_anchor_generator=None, 
                 rpn_head=None,
                 rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_roi_pool=None, 
                 feat_head=None, 
                 box_predictor=None,
                 box_score_thresh=0.0, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 embedding_head=None, 
                 reid_regressor=None,
                 part_cls_scalar=1.0,
                 part_num=3):

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                'backbone should contain an attribute out_channels '
                'specifying the number of output channels (assumed to be the '
                'same for all the levels)')

        
        if rpn_anchor_generator is None:
            raise ValueError('rpn_anchor_generator should be specified manually.')
        
        if rpn_head is None:
            raise ValueError('rpn_head should be specified manually.')
        
        if box_roi_pool is None:
            raise ValueError('box_roi_pool should be specified manually.')
        
        if feat_head is None:
            raise ValueError('feat_head should be specified manually.')

        if box_predictor is None:
            raise ValueError('box_predictor should be specified manually.')

        if embedding_head is None:
            raise ValueError('embedding_head should be specified manually.')

        # Construct RPN
        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = self._set_rpn(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        # Construct ROI head 
        roi_heads = self._set_roi_heads(
            embedding_head, reid_regressor, part_cls_scalar, part_num,
            box_roi_pool, feat_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        # Construct image transformer
        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        super(FasterRCNN, self).__init__(
            backbone, rpn, roi_heads, transform)

    def _set_rpn(self, *args):
        return RegionProposalNetwork(*args)

    def _set_roi_heads(self, *args):
        return OrthogonalRoiHeads(*args)

    def ex_feat(self, images, targets, mode='det'):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == 'det':
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == 'reid':
            return self.ex_feat_by_img_crop(images, targets)

    def ex_feat_by_roi_pooling(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals = [x['boxes'] for x in targets]

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
        
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])

        embeddings, _ = self.roi_heads.embedding_head(rcnn_features)
        embeddings = embeddings.squeeze(3).squeeze(2)
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, 'Only support batch_size 1 in this mode'

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        
        embeddings, _ = self.roi_heads.embedding_head(rcnn_features)
        embeddings = embeddings.squeeze(3).squeeze(2)
        return embeddings.split(1, 0)

def part_separation(proposals, num_parts=5):
    part_proposals = []
    
    for k in range(num_parts):
        props_parts = []
        for props in proposals:
            part_height = (props[:,[3]] - props[:,[1]]) / num_parts
            props_part = props.clone()
            props_part[:,[3]] = props_part[:,[3]] - (num_parts-(k+1)) * part_height
            props_part[:,[1]] = props_part[:,[1]] + k * part_height
            props_parts.append(props_part)
        part_proposals.append(props_parts)
    return part_proposals


class OrthogonalRoiHeads(RoIHeads):

    def __init__(self, embedding_head, reid_regressor, part_cls_scalar, num_parts, *args, **kwargs):
        super(OrthogonalRoiHeads, self).__init__(*args, **kwargs)
        self.embedding_head = embedding_head
        self.reid_regressor = reid_regressor
        self.num_parts = int(num_parts)
        part_height = int(math.ceil(24.0 / float(num_parts)))
        self.part_pooling = MultiScaleRoIAlign(
                                featmap_names=['feat_res4'],
                                output_size=[part_height,8],
                                sampling_ratio=2)

        self.part_cls_scalar = float(part_cls_scalar)
        self.part_projectors = nn.ModuleDict()
        for ftname, in_chennel in zip(['feat_res4', 'feat_res5'], [1024, 2048]):
            proj = nn.Conv2d(in_chennel, self.num_parts+1, 1, 1, 0)
            init.normal_(proj.weight, std=0.01)
            init.constant_(proj.bias, 0)
            self.part_projectors[ftname] = proj

    @property
    def feat_head(self):  # re-name
        return self.box_head

    
    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def select_training_samples(self,
                                proposals,  # type: List[Tensor]
                                targets     # type: Optional[List[Dict[str, Tensor]]]
                                ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        assert targets is not None
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        if self.training: 
            gt_labels = [t["cnt"].to(device) for t in targets]
            # gt_labels = [t["labels"].to(device) for t in targets]
        else: gt_labels = [t["labels"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets

    def forward(self, epoch, features, proposals, images, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        image_shapes=images.image_sizes
        # cnt = 483
        # if targets is not None:
        #     for k in range(len(targets)):
        #         for i in range(len(targets[k]['labels'])):
        #             # if(targets[k]['labels'][i] == 5555):
        #             if(targets[k]['labels'][i] == 5555):
        #                 targets[k]['labels'][i] = cnt
        #                 cnt += 1
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        rcnn_features = self.feat_head(
                self.box_roi_pool(features, proposals, image_shapes))

        if self.training:
            result, losses = [], {}
            det_labels = [(y != 0).long() for y in labels]
            box_regression = self.box_predictor(rcnn_features['feat_res5'])
            embeddings_, class_logits = self.embedding_head(rcnn_features, det_labels)
            cls_scores = F.softmax(class_logits, dim=1)[:,[1]]
            
            # #######################################
            # ## ------ Part classification ------ ##
            # #######################################
            # outputs = []
            # for k, v in rcnn_features.items():
            #     outputs.append(self.part_projectors[k](v))
            # predicted = sum(outputs) * cls_scores * self.part_cls_scalar 
            # N, _, d1, d2 = predicted.shape
            # part_labels = torch.zeros([N,d1,d2]).to(predicted.device)
            # fg_labels = torch.cat(det_labels)
            # for part, y in zip(part_labels, fg_labels):
            #     if y > 0:
            #         for k in range(self.num_parts):
            #             part[k].fill_(k+1)
            # loss_parts = F.cross_entropy(predicted, part_labels.long().detach())

            # ########################################

            loss_detection, loss_box_reg = \
                rcnn_loss(class_logits.squeeze(3).squeeze(2), box_regression,
                                     det_labels, regression_targets)
            
            cls_scores = cls_scores.squeeze(3).squeeze(2)
            embeddings_ = embeddings_.squeeze(3).squeeze(2)

            # loss_reid = self.reid_regressor(embeddings_, labels, cls_scores) 
            loss_reid = self.reid_regressor(epoch, embeddings_, labels, cls_scores, images, proposals, targets) 
            losses = dict(loss_detection=loss_detection,
                          loss_box_reg=loss_box_reg,
                          loss_reid=loss_reid)
                          
        else:
            box_regression = self.box_predictor(rcnn_features['feat_res5'])
            embeddings_, class_logits = self.embedding_head(rcnn_features, None)
            embeddings_ = embeddings_.squeeze(3).squeeze(2)
            class_logits = class_logits.squeeze(3).squeeze(2)

            result, losses = [], {}
            boxes, scores, embeddings, labels = \
                self.postprocess_detections(class_logits, box_regression, embeddings_,
                                            proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted
        return result, losses

    def postprocess_detections(self, class_logits, box_regression, embeddings_, proposals, image_shapes):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        class_logits = F.softmax(class_logits, dim=1)
        pred_scores = class_logits[:,1]
        # print('pred_scores.shape', pred_scores.shape)
        embeddings_ = embeddings_ * pred_scores.view(-1, 1)  # CWS

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = boxes[
                inds], scores[inds], labels[inds], embeddings[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class OrthogonalEmbeddingProj(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256,
                 cls_scalar=1.0):
        super(OrthogonalEmbeddingProj, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = list(map(int, in_channels))
        self.dim = int(dim)
        self.cls_scalar = cls_scalar
        
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            proj = nn.Conv2d(in_chennel, 2, 1, 1, 0)
            init.normal_(proj.weight, std=0.01)
            init.constant_(proj.bias, 0)
            self.projectors[ftname] = proj

        self.projectors_reid = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            proj = nn.Sequential(
                nn.Conv2d(in_chennel, indv_dim, 1, 1, 0),
                nn.BatchNorm2d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors_reid[ftname] = proj

        # self.pred_class = nn.Conv2d(self.dim, 2, 1,1,0, bias=False)
        # init.normal_(self.pred_class.weight, std=0.01)

    def forward(self, featmaps, targets=None):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        
        if targets is not None:
            # Train mode
            targets = torch.cat(targets,dim=0)
        
        outputs = []
        for k, v in featmaps.items():
            v = F.adaptive_max_pool2d(v, 1)
            outputs.append(
                self.projectors[k](v)
            )
        projected = sum(outputs) * self.cls_scalar

        outputs_reid = []
        for k, v in featmaps.items():
            v = F.adaptive_max_pool2d(v, 1)
            outputs_reid.append(
                self.projectors_reid[k](v)
            )
        
        embeddings_reid = torch.cat(outputs_reid, dim=1)
        
        
        embeddings_reid = embeddings_reid / \
            embeddings_reid.norm(dim=1, keepdim=True).clamp(min=1e-12)

        return embeddings_reid, projected

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            x = F.adaptive_max_pool2d(x, 1)
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class BboxRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection
    """
    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(BboxRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes),
                nn.BatchNorm1d(4 * num_classes))
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)
        
    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Norm-Aware R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, )
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(
        class_logits, labels.long())

    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

def get_model(args, GT_MC=None, training=True, pretrained_backbone=True):
    phase_args = args.train if training else args.test
    
    # Resnet50
    resnet_part1, resnet_part2 = resnet_backbone('resnet50', 
                                        pretrained_backbone, 
                                        GAP=True, 
                                        num_parts=args.part_num)

    ##### Region Proposal Network ######
    # Anchor generator (Default)
    rpn_anchor_generator = AnchorGenerator(
                            (tuple(args.anchor_scales),), 
                            (tuple(args.anchor_ratios),))
    # 2D embedding head
    backbone = resnet_part1
    # 1D embedding head
    rpn_head = RPNHead(
                resnet_part1.out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])

    ########## Bbox Network #########
    # Bbox regressor
    box_predictor = BboxRegressor(
                    2048, num_classes=2,
                    RCNN_bbox_bn=args.rcnn_bbox_bn)
    # Bbox pooler
    box_roi_pool = MultiScaleRoIAlign(
                    featmap_names=['feat_res4'],
                    output_size=[24,8],
                    sampling_ratio=2)
    # 2D embedding head
    feat_head = resnet_part2
    # 1D embedding head
    embedding_head = OrthogonalEmbeddingProj(
                    featmap_names=['feat_res4', 'feat_res5'],
                    in_channels=[1024, 2048],
                    dim=args.num_features,
                    cls_scalar=args.cls_scalar)
    # ReID regressor
    # reid_regressor = OIMLoss(
    #                     args.num_features, args.num_pids, args.num_cq_size, 
    #                     args.train.oim_momentum, args.oim_scalar)
    reid_regressor = MCLoss(
                        args.num_features, args.num_pids, args.num_cq_size, 
                        args.train.oim_momentum, args.oim_scalar,
                        0, 0, GT_MC)
                        
    model = FasterRCNN( 
                        # Region proposal network
                        backbone=backbone,
                        min_size=phase_args.min_size, max_size=phase_args.max_size,
                        # Anchor generator
                        rpn_anchor_generator=rpn_anchor_generator,
                        # RPN parameters
                        rpn_head=rpn_head,
                        rpn_pre_nms_top_n_train=args.train.rpn_pre_nms_top_n,
                        rpn_post_nms_top_n_train=args.train.rpn_post_nms_top_n,
                        rpn_pre_nms_top_n_test=args.test.rpn_pre_nms_top_n,
                        rpn_post_nms_top_n_test=args.test.rpn_post_nms_top_n,
                        rpn_nms_thresh=phase_args.rpn_nms_thresh,
                        rpn_fg_iou_thresh=args.train.rpn_positive_overlap,
                        rpn_bg_iou_thresh=args.train.rpn_negative_overlap,
                        rpn_batch_size_per_image=args.train.rpn_batch_size,
                        rpn_positive_fraction=args.train.rpn_fg_fraction,
                        # Bbox network
                        box_predictor=box_predictor,
                        box_roi_pool=box_roi_pool,
                        feat_head=feat_head,
                        embedding_head=embedding_head,
                        box_score_thresh=args.train.fg_thresh,
                        box_nms_thresh=args.test.nms,  # inference only
                        box_detections_per_img=phase_args.rpn_post_nms_top_n,  # use all
                        box_fg_iou_thresh=args.train.bg_thresh_hi,
                        box_bg_iou_thresh=args.train.bg_thresh_lo,
                        box_batch_size_per_image=args.train.rcnn_batch_size,
                        box_positive_fraction=args.train.fg_fraction,  # for proposals
                        bbox_reg_weights=args.train.box_regression_weights,
                        reid_regressor=reid_regressor,
                        part_cls_scalar=args.part_cls_scalar,
                        part_num=args.part_num
                        )
    if training:
        model.train()
    else:
        model.eval()
    
    return model
