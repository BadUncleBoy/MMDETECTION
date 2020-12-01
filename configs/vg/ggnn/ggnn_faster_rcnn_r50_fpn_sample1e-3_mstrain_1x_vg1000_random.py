_base_ = [
    '../../_base_/models/ggnn_faster_rcnn_r50_fpn.py',
    '../../_base_/datasets/vg1000_detection.py',
    '../../_base_/schedules/schedule_1x.py', 
    '../../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(num_classes=1000,
        ggnn_config=dict(initweight_path="./data/vg/init_weights_random.pt"))))
test_cfg = dict(
    rcnn=dict(
        score_thr=0.0001,
        max_per_img=300))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
data = dict(samples_per_gpu=1, train=dict(dataset=dict(pipeline=train_pipeline)))
work_dir = "exps/vg/ggnnr50_random"
seed = 0
gpu_ids = range(0, 1)
load_from = "exps/vg/r50/latest.pth"
resume_from = None
