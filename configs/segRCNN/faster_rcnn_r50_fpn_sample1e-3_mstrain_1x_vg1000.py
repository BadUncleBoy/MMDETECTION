_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/selfvoc.py',
    '../_base_/schedules/schedule_2x.py', 
    '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))
data = dict(samples_per_gpu=3, workers_per_gpu=0)

load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume_from = None
work_dir = './exps/segRCNN/r50/'
seed = 0
gpu_ids = range(0,1)


