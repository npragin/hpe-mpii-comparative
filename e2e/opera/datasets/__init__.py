# Copyright (c) Hikvision Research Institute. All rights reserved.
from .builder import DATASETS, PIPELINES, build_dataset, build_dataloader
from .coco_pose import CocoPoseDataset
from .mpii_pose import MPIIPoseDataset
from .crowd_pose import CrowdPoseDataset
from .objects365 import Objects365
from .pipelines import *
from .utils import replace_ImageToTensor

__all__ = [
    'DATASETS', 'PIPELINES', 'build_dataset', 'build_dataloader',
    'CocoPoseDataset', 'MPIIPoseDataset', 'CrowdPoseDataset', 'Objects365',
    'replace_ImageToTensor'
]
