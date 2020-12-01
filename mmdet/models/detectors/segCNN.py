from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class SEGRCNN(TwoStageDetector):
    
    def __init__(self,
                 stage,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        
        super(SEGRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector."""
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(SEGRCNN, self).show_result(data, result, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list, seg_scores = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals
        if self.with_roi_head:
            # add attention to the feature maps
            x = list(x)
            for i in range(len(x)):
                x[i] = x[i] * seg_scores[i]
            x = tuple(x)
            roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                    gt_bboxes, gt_labels,
                                                    gt_bboxes_ignore, gt_masks,
                                                    **kwargs)
            losses.update(roi_losses)

        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        # assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        if proposals is None:
            proposal_list, seg_scores_sig= self.rpn_head.simple_test_rpn(x, img_metas)
            self.test_seg_scores(seg_scores_sig)
        else:
            proposal_list = proposals
        if self.with_roi_head:
            x = list(x)
            for i in range(len(x)):
                x[i] = x[i] * seg_scores_sig[i]
            x = tuple(x)
            return self.roi_head.simple_test(
                x, proposal_list, img_metas, rescale=rescale)
        else:
            results = []
            for each_proposal, each_meta in zip(proposal_list, img_metas):
                each_proposal = each_proposal.cpu().numpy()
                scale_factor = each_meta["scale_factor"]
                each_proposal[:, :-1] /= scale_factor.reshape(1,-1)
                results.append(each_proposal)
            return results

    def test_seg_scores(self, seg_scores_sig):
        import numpy as np
        import cv2
        strides=[4,8,16,32,64]
        seg_scores_sig = [each.cpu().numpy() for each in seg_scores_sig]
        size = [each.shape[-2:] for each in seg_scores_sig]
        for i, stride in enumerate(strides):
            img = np.ones((size[i][0] * stride, size[i][1] * stride, 3))*255
            for h in range(size[i][0]):
                for w in range(size[i][1]):
                    img[h*stride:h*stride+stride, w*stride:w*stride+stride, :] = seg_scores_sig[i][0,0,h,w] * 255
            img = img.astype(np.uint8)
            cv2.imwrite("seg_{}.jpg".format(stride), img)
