#self add
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32
from mmdet.core import (anchor_inside_flags, build_anchor_generator,
                        build_assigner, build_bbox_coder, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)

from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead
from .rpn_test_mixin import RPNTestMixin


@HEADS.register_module()
class SEGRPNHead(AnchorHead):

    def __init__(self, num_classes, in_channels, loss_seg, **kwargs):
        super(SEGRPNHead, self).__init__(1,in_channels, **kwargs)
        self.loss_seg = build_loss(loss_seg)

    def _init_layers(self):
        # cls reg branch
        self.rpn_conv1 = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        # seg branch
        self.rpn_conv2 = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(self.feat_channels,
                                 self.num_anchors * self.cls_out_channels, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        self.rpn_seg = nn.Conv2d(self.feat_channels, 1, 1)
    
    def init_weights(self):
        """Initialize weights of the head."""
        normal_init(self.rpn_cls, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.rpn_conv1, std=0.01)
        normal_init(self.rpn_conv2, std=0.01)
        normal_init(self.rpn_seg, std=0.01)

    def forward_single(self, x):
        x1 = self.rpn_conv1(x)
        x1 = F.relu(x1, inplace=True)
        x2 = self.rpn_conv2(x)
        x2 = F.relu(x2, inplace=True)
        rpn_cls_score = self.rpn_cls(x1)
        rpn_bbox_pred = self.rpn_reg(x1)
        rpn_seg_score = self.rpn_seg(x2)
        return rpn_cls_score, rpn_bbox_pred, rpn_seg_score

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses= self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list= self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            seg_scores_sig = [each.sigmoid() for each in outs[2]]
            return losses, proposal_list, seg_scores_sig

    # def loss_single(self, 
    #                 cls_score, 
    #                 bbox_pred, 
    #                 seg_scores, 
    #                 anchors, 
    #                 cls_labels, 
    #                 cls_label_weights,
    #                 seg_labels,
    #                 seg_label_weights,
    #                 bbox_targets, 
    #                 bbox_weights, 
    #                 num_total_samples):

    #     # classification loss
    #     cls_labels = cls_labels.reshape(-1)
    #     cls_label_weights = cls_label_weights.reshape(-1)
    #     cls_score = cls_score.permute(0, 2, 3,
    #                                   1).reshape(-1, self.cls_out_channels)
    #     loss_cls = self.loss_cls(
    #         cls_score, cls_labels, cls_label_weights, avg_factor=num_total_samples)
    #     # regression loss
    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     bbox_weights = bbox_weights.reshape(-1, 4)
    #     bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
    #     if self.reg_decoded_bbox:
    #         anchors = anchors.reshape(-1, 4)
    #         bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
    #     loss_bbox = self.loss_bbox(
    #         bbox_pred,
    #         bbox_targets,
    #         bbox_weights,
    #         avg_factor=num_total_samples)

    #     seg_scores = seg_scores.permute(0, 2, 3, 1).reshape(-1, 1)
    #     seg_labels = seg_labels.reshape(-1)
    #     seg_label_weights = seg_label_weights.reshape(-1)
    #     #loss_seg = self.loss_seg(
    #     #    seg_scores,
    #     #    seg_labels,
    #     #    seg_label_weights,
    #     #    avg_factor = torch.sum(seg_label_weights))
    #     return loss_cls, loss_bbox

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             seg_scores,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        return super(SEGRPNHead, self).loss(cls_scores = cls_scores,
                                     bbox_preds = bbox_preds,
                                     gt_bboxes = gt_bboxes,
                                     gt_labels = gt_labels,
                                     img_metas = img_metas,
                                     gt_bboxes_ignore = gt_bboxes_ignore)
        # featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # assert len(featmap_sizes) == self.anchor_generator.num_levels

        # device = cls_scores[0].device

        # anchor_list, valid_flag_list = self.get_anchors(
        #     featmap_sizes, img_metas, device=device)
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        # cls_reg_targets = self.get_targets(
        #     anchor_list,
        #     valid_flag_list,
        #     gt_bboxes,
        #     img_metas,
        #     gt_bboxes_ignore_list=gt_bboxes_ignore,
        #     gt_labels_list=gt_labels,
        #     label_channels=label_channels)
        # if cls_reg_targets is None:
        #     return None
        # (cls_labels_list, cls_label_weights_list, bbox_targets_list, bbox_weights_list,
        #  num_total_pos, num_total_neg) = cls_reg_targets
        # num_total_samples = (
        #     num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # # anchor number of multi levels
        # num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # # concat all level anchors and flags to a single tensor
        # concat_anchor_list = []
        # for i in range(len(anchor_list)):
        #     concat_anchor_list.append(torch.cat(anchor_list[i]))
        # all_anchor_list = images_to_levels(concat_anchor_list,
        #                                    num_level_anchors)

        # # seg_labels_list, seg_label_weights_list = self.get_seg_target(seg_scores, gt_bboxes, device)

        # losses_cls, losses_bbox = multi_apply(
        #     self.loss_single,
        #     cls_scores,
        #     bbox_preds,
        #     seg_scores,
        #     all_anchor_list,
        #     cls_labels_list,
        #     cls_label_weights_list,
        #     seg_labels_list,
        #     seg_label_weights_list,
        #     bbox_targets_list,
        #     bbox_weights_list,
        #     num_total_samples=num_total_samples)
        # return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)


    # def get_seg_target(self, seg_scores, gt_bboxes, device):
    #     strides=[4, 8, 16, 32, 64]
    #     batch_size = len(gt_bboxes)
    #     feat_sizes = [each.size()[-2:] for each in seg_scores]
    #     seg_labels_list = []
    #     seg_weights_list = []
    #     for si, stride in enumerate(strides):
    #         seg_labels = torch.ones((batch_size, feat_sizes[si][0], feat_sizes[si][1]), dtype=torch.int64, device=device)
    #         seg_weights = torch.ones((batch_size, feat_sizes[si][0], feat_sizes[si][1]), dtype=torch.float32, device=device)
    #         for pi, bbox_per in enumerate(gt_bboxes):
    #             bbox_per_down = (bbox_per / stride).type(torch.int32)
    #             for bi in range(bbox_per.size()[0]):
    #                 x_min, y_min, x_max, y_max = bbox_per_down[bi]
    #                 x_mid = torch.floor_divide((x_max + x_min), 2).type(torch.int32)
    #                 y_mid = torch.floor_divide((y_max + y_min), 2).type(torch.int32)
    #                 choice = random.randint(1, 8)
    #                 if choice == 1:
    #                     seg_weights[pi, y_mid:y_max, x_min:x_max] = 0
    #                 elif choice == 2:
    #                     seg_weights[pi, y_min:y_mid, x_min:x_max] = 0
    #                 elif choice == 3:
    #                     seg_weights[pi, y_min:y_max, x_mid:x_max] = 0
    #                 elif choice == 4:
    #                     seg_weights[pi, y_min:y_max, x_min:x_mid] = 0
    #                 elif choice == 5:
    #                     x_mid_l = torch.floor_divide((x_min + x_mid), 2).type(torch.int32)
    #                     x_mid_r = torch.floor_divide((x_mid + x_max), 2).type(torch.int32)
    #                     y_mid_t = torch.floor_divide((y_min + y_mid), 2).type(torch.int32)
    #                     y_mid_b = torch.floor_divide((y_mid + y_max), 2).type(torch.int32)
    #                     seg_weights[pi, y_min:y_max, x_min:x_max] = 0
    #                     seg_weights[pi, y_mid_t:y_mid_b,x_mid_l:x_mid_r] = 1
    #                 elif choice == 6:
    #                     x_mid_l = torch.floor_divide((x_min + x_mid), 2).type(torch.int32)
    #                     x_mid_r = torch.floor_divide((x_mid + x_max), 2).type(torch.int32)
    #                     y_mid_t = torch.floor_divide((y_min + y_mid), 2).type(torch.int32)
    #                     y_mid_b = torch.floor_divide((y_mid + y_max), 2).type(torch.int32)
    #                     seg_weights[pi, y_mid_t:y_mid_b,x_mid_l:x_mid_r] = 0
    #                 seg_labels[pi, y_min:y_max, x_min:x_max] = 0
    #         seg_labels_list.append(seg_labels.reshape(batch_size, -1))
    #         seg_weights_list.append(seg_weights.reshape(batch_size, -1))

    #     return seg_labels_list, seg_weights_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'seg_scores'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   seg_scores,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        results = super(SEGRPNHead, self).get_bboxes(cls_scores = cls_scores,
                                            bbox_preds = bbox_preds,
                                            img_metas = img_metas,
                                            cfg = cfg,
                                            rescale = rescale,
                                            with_nms = with_nms)
        return results
    
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (num_anchors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (num_anchors * 4, H, W).
            mlvl_anchors (list[Tensor]): Box reference for each scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        # bboxes from different level should be independent during NMS,
        # level_ids are used as labels for batched NMS to separate them
        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                # We set FG labels to [0, num_class-1] and BG label to
                # num_class in RPN head since mmdet v2.5, which is unified to
                # be consistent with other head since mmdet v2.0. In mmdet v2.0
                # to v2.4 we keep BG label as 0 and FG label as 1 in rpn head.
                scores = rpn_cls_score.softmax(dim=1)[:, 0]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            anchors = mlvl_anchors[idx]
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:cfg.nms_pre]
                scores = ranked_scores[:cfg.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(
                scores.new_full((scores.size(0), ), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(
            anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if cfg.min_bbox_size > 0:
            w = proposals[:, 2] - proposals[:, 0]
            h = proposals[:, 3] - proposals[:, 1]
            valid_inds = torch.nonzero(
                (w >= cfg.min_bbox_size)
                & (h >= cfg.min_bbox_size),
                as_tuple=False).squeeze()
            if valid_inds.sum().item() != len(proposals):
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
                ids = ids[valid_inds]

        # TODO: remove the hard coded nms type
        nms_cfg = dict(type='nms', iou_threshold=cfg.nms_thr)
        dets, keep = batched_nms(proposals, scores, ids, nms_cfg)
        return dets[:cfg.nms_post]


    def simple_test_rpn(self, x, img_metas):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Proposals of each image.
        """
        rpn_outs = self(x)
        proposal_list = self.get_bboxes(*rpn_outs, img_metas)
        seg_scores_sig = [each.sigmoid() for each in rpn_outs[2]]
        return proposal_list, seg_scores_sig
