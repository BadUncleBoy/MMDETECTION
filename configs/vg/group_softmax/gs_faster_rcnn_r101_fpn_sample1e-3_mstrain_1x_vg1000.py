_base_ = './gs_faster_rcnn_r50_samplele-3_mstrain_1x_vg1000.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
work_dir = 'exps/vg/gsr101'
load_from = 'exps/vg/r101/latest.pth'
resume_from = None
