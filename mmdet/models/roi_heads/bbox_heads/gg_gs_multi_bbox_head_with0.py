#self add
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from mmdet.core import (force_fp32, multiclass_nms)
from .convfc_bbox_head import SharedFCBBoxHead
from mmdet.models.builder import HEADS, build_loss

class Propogator(nn.Module):
    """
    Gated Propogator for GGNN
    Using LSTM gating mechanism
    """
    def __init__(self, state_dim):
        super(Propogator, self).__init__()

        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Sigmoid()
        )
        self.tansform = nn.Sequential(
            nn.Linear(state_dim*2, state_dim),
            nn.Tanh()
        )

    def forward(self, state, A):
        a0 = torch.mm(state, A)

        a = torch.cat((a0, state), 1)

        r = self.reset_gate(a)
        z = self.update_gate(a)
        joined_input = torch.cat((a0, r * state), 1)
        h_hat = self.tansform(joined_input)

        output = (1 - z) * state + z * h_hat

        return output
class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, A, fc_out_channels, n_steps=5):
        super(GGNN, self).__init__()
        self.n_steps = n_steps
        self.A = A
        state_dim = A[0].size()[0]
        #convert input node dimension to state_dim
        self.feature2state_dim = nn.Sequential(
                                    nn.Linear(fc_out_channels, state_dim),
                                    nn.Tanh())
        self.last_fc = nn.Linear(state_dim, state_dim)
        # Propogation Model
        self.propogator_list = nn.ModuleList()
        self.feat_weight = nn.Parameter(torch.ones(len(A), dtype=torch.float32), requires_grad=True)
        for _ in A:
            self.propogator_list.append(Propogator(state_dim))

    def forward(self, feat):
        state = self.feature2state_dim(feat)
        gg_feats = 0
        for index, a in enumerate(self.A):
            state_each = state
            for _ in range(self.n_steps):
                state_each = self.propogator_list[index](state_each, a)
            gg_feats += state_each * self.feat_weight[index]
        return self.last_fc(gg_feats / (torch.sum(self.feat_weight)+1e-4))

class CLASS_HEAD(nn.Module):
    def __init__(self, gggs_config, cls_last_dim, fc_out_channels):
        super(CLASS_HEAD, self).__init__()

        self.fc_bins = nn.ModuleList()
        #background bin
        self.fc_bins.append(nn.Linear(cls_last_dim, 2))
        self.num_bins = gggs_config.num_bins
        state_dim    = gggs_config.state_dim
        self.num_bins = gggs_config.num_bins
        #forceground bin
        for i in range(self.num_bins-1):
            adjecent = torch.load(gggs_config.adjecent_path[i])
            adjecent = [each.cuda() for each in adjecent]
            self.fc_bins.append(GGNN(A=adjecent,
                                     fc_out_channels=fc_out_channels,
                                     n_steps=gggs_config.n_steps))
    def forward(self, feat):
        class_preds = []
        for i in range(self.num_bins):
            class_preds.append(self.fc_bins[i](feat))
        return torch.cat(class_preds, dim=-1).contiguous()

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
 
@HEADS.register_module
class GGGSMultiBBoxHeadWith0(SharedFCBBoxHead):

    def __init__(self,
                 num_fcs=2,
                 fc_out_channels=1024,
                 gggs_config=None,
                 *args,
                 **kwargs):
        super(GGGSMultiBBoxHeadWith0, self).__init__(num_fcs=num_fcs,
                                         fc_out_channels=fc_out_channels,
                                         *args,
                                         **kwargs)
        
        self.fc_cls = CLASS_HEAD(gggs_config=gggs_config,
                                 cls_last_dim=self.cls_last_dim,
                                 fc_out_channels=fc_out_channels)
        self.loss_bins = []
        for i in range(gggs_config.num_bins):
            self.loss_bins.append(build_loss(gggs_config.loss_bin))

        self.label2binlabel = torch.load(gggs_config.label2binlabel).cuda()
        self.pred_slice = torch.load(gggs_config.pred_slice).cuda()

        # TODO: update this ugly implementation. Save fg_split to a list and
        #  load groups by gs_config.num_bins
        with open(gggs_config.fg_split, 'rb') as fin:
            fg_split = pickle.load(fin)

        self.fg_splits = []
        self.fg_splits.append(torch.from_numpy(fg_split['(10000,~)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['(2000,10000)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['(500,2000)']).cuda())
        self.fg_splits.append(torch.from_numpy(fg_split['(0,500)']).cuda())

        self.others_sample_ratio = gggs_config.others_sample_ratio

    def init_weights(self):
        self.fc_cls._initialization()
        # 重写权重初始化函数
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
            
    def _sample_others(self, label):

        # only works for non bg-fg bins

        fg = torch.where(label > 0, torch.ones_like(label),
                         torch.zeros_like(label))
        fg_idx = fg.nonzero(as_tuple=True)[0]
        fg_num = fg_idx.shape[0]
        if fg_num == 0:
            return torch.zeros_like(label)

        bg = 1 - fg
        bg_idx = bg.nonzero(as_tuple=True)[0]
        bg_num = bg_idx.shape[0]

        bg_sample_num = int(fg_num * self.others_sample_ratio)

        if bg_sample_num >= bg_num:
            weight = torch.ones_like(label)
        else:
            sample_idx = np.random.choice(bg_idx.cpu().numpy(),
                                          (bg_sample_num, ), replace=False)
            sample_idx = torch.from_numpy(sample_idx).cuda()
            fg[sample_idx] = 1
            weight = fg

        return weight

    def _remap_labels(self, labels):

        num_bins = self.label2binlabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i]
            new_bin_label = mapping[labels]

            if i < 1:
                weight = torch.ones_like(new_bin_label)
                # weight = torch.zeros_like(new_bin_label)
            else:
                weight = self._sample_others(new_bin_label)
            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _remap_labels1(self, labels):

        num_bins = self.label2binlabel.shape[0]
        new_labels = []
        new_weights = []
        new_avg = []
        for i in range(num_bins):
            mapping = self.label2binlabel[i]
            new_bin_label = mapping[labels]

            weight = torch.ones_like(new_bin_label)

            new_labels.append(new_bin_label)
            new_weights.append(weight)

            avg_factor = max(torch.sum(weight).float().item(), 1.)
            new_avg.append(avg_factor)

        return new_labels, new_weights, new_avg

    def _slice_preds(self, cls_score):

        new_preds = []

        num_bins = self.pred_slice.shape[0]
        for i in range(num_bins):
            start = self.pred_slice[i, 0]
            length = self.pred_slice[i, 1]
            sliced_pred = cls_score.narrow(1, start, length)
            new_preds.append(sliced_pred)

        return new_preds

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None):
        losses = dict()

        if cls_score is not None:
            # Original label_weights is 1 for each roi.
            new_labels, new_weights, new_avgfactors = self._remap_labels(labels)
            new_preds = self._slice_preds(cls_score)

            num_bins = len(new_labels)
            for i in range(num_bins):
                losses['loss_cls_bin{}'.format(i)] = self.loss_bins[i](
                    new_preds[i],
                    new_labels[i],
                    new_weights[i],
                    avg_factor=new_avgfactors[i],
                    reduction_override=reduction_override
                )

        if bbox_pred is not None:
            pos_inds = labels != self.num_classes
            if torch.sum(pos_inds) == 0:
                print("called")
                losses["loss_bbox"] = torch.tensor(2, dtype=torch.float32)
                return losses
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to=('cls_score'))
    def _merge_score1(self, cls_score):
        '''
        Do softmax in each bin. Merge the scores directly.
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i]

        merge[:, 0] = bg_score[:, 0]
        fg_idx = (bg_score[:,1] > 0.5).nonzero(as_tuple=True)[0]
        merge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score2(self, cls_score):
        '''
        Do softmax in each bin. Softmax again after merge.
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i]

        merge[:, 0] = bg_score[:, 0]
        fg_idx = (bg_score[:,1] > 0.5).nonzero(as_tuple=True)[0]
        merge[fg_idx] = fg_merge[fg_idx]
        merge = F.softmax(merge)

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score(self, cls_score):
        '''
        Do softmax in each bin. Decay the score of normal classes
        with the score of fg.
        From v1.
        '''

        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, self.num_classes)).cuda()
        #!!!! add background label=num_classes
        merge = torch.zeros((num_proposals, self.num_classes+1)).cuda()

        # import pdb
        # pdb.set_trace()
        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i][:, 1:]

        weight = bg_score.narrow(1, 0, 1)

        # Whether we should add this? Test
        fg_merge = weight * fg_merge

        merge[:, -1] = bg_score[:, 1]
        merge[:, :-1] = fg_merge[:, :]
        # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        # erge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score4(self, cls_score):
        '''
        Do softmax in each bin.
        Do softmax on merged fg classes.
        Decay the score of normal classes with the score of fg.
        From v2 and v3
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            fg_merge[:, split] = fg_score[i]

        fg_merge = F.softmax(fg_merge, dim=1)
        weight = bg_score.narrow(1, 1, 1)
        fg_merge = weight * fg_merge

        merge[:, 0] = bg_score[:, 0]
        merge[:, 1:] = fg_merge[:, 1:]
        # fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        # erge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score'))
    def _merge_score5(self, cls_score):
        '''
        Do softmax in each bin.
        Pick the bin with the max score for each box.
        '''
        num_proposals = cls_score.shape[0]

        new_preds = self._slice_preds(cls_score)
        new_scores = [F.softmax(pred, dim=1) for pred in new_preds]

        bg_score = new_scores[0]
        fg_score = new_scores[1:]
        max_scores = [s.max(dim=1, keepdim=True)[0] for s in fg_score]
        max_scores = torch.cat(max_scores, 1)
        max_idx = max_scores.argmax(dim=1)

        fg_merge = torch.zeros((num_proposals, 1231)).cuda()
        merge = torch.zeros((num_proposals, 1231)).cuda()

        for i, split in enumerate(self.fg_splits):
            tmp_merge = torch.zeros((num_proposals, 1231)).cuda()
            tmp_merge[:, split] = fg_score[i]
            roi_idx = torch.where(max_idx == i,
                                  torch.ones_like(max_idx),
                                  torch.zeros_like(max_idx)).nonzero(
                as_tuple=True)[0]
            fg_merge[roi_idx] = tmp_merge[roi_idx]

        merge[:, 0] = bg_score[:, 0]
        fg_idx = (bg_score[:, 1] > 0.5).nonzero(as_tuple=True)[0]
        merge[fg_idx] = fg_merge[fg_idx]

        return merge

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def get_bboxes(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    scale_factor,
                    rescale=False,
                    cfg=None):
        if isinstance(cls_score, list):
            cls_score = sum(cls_score) / float(len(cls_score))

        scores = self._merge_score(cls_score)
        # scores = F.softmax(cls_score, dim=1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
