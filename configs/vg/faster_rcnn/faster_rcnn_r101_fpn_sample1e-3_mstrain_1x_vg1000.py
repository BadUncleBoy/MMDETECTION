_base_ = './faster_rcnn_r50_fpn_sample1e-3_mstrain_1x_vg1000.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
data = dict(samples_per_gpu=2)
work_dir = 'exps/vg/r101'
load_from = 'checkpoints/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth'
resume_from = None
