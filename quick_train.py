import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmcv import Config
import mmcv
import os.path as osp
cfg = Config.fromfile('./configs/segRCNN/seg_config.py')
# # Build dataset
datasets = [build_dataset(cfg.data.train)]

# # Build the detector
model = build_detector(
     cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# # Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# # Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
