# dataset settings
dataset_type = 'opera.MPIIPoseDataset'
data_root = '../datasets/mpii/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True) # Image net norm

# train_pipeline
train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', to_float32=True),
    dict(type='opera.LoadAnnotations',
         with_bbox=True,
         with_keypoint=True,
         with_area=True),
    dict(
        type='mmdet.PhotoMetricDistortion',
        brightness_delta=76.5, # 0.3 
        contrast_range=(0.7, 1.3), # 0.3 
        saturation_range=(0.7, 1.3), # 0.3 
        hue_delta=72), # 0.2
    dict(type='opera.RandomFlip', flip_ratio=0.0),
    dict(
        type='mmdet.AutoAugment',
        policies=[
            [
                dict(
                    type='opera.Resize',
                    img_scale=[(256, 256)],  # Updated to 256x256
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
        ]),
    dict(type='mmdet.Normalize', **img_norm_cfg),
    dict(type='mmdet.Pad', size_divisor=1),
    dict(type='opera.DefaultFormatBundle',
         extra_keys=['gt_keypoints', 'gt_areas']),
    dict(type='mmdet.Collect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints', 'gt_areas']),
]

# test_pipeline
test_pipeline = [
    dict(type='mmdet.LoadImageFromFile'),
    dict(
        type='mmdet.MultiScaleFlipAug',
        img_scale=(256, 256),  # Updated to 256x256
        flip=False,
        transforms=[
            dict(type='mmdet.Resize', keep_ratio=True),
            dict(type='mmdet.Normalize', **img_norm_cfg),
            dict(type='mmdet.Pad', size_divisor=1),
            dict(type='mmdet.ImageToTensor', keys=['img']),
            dict(type='mmdet.Collect', keys=['img']),
        ])
]

# dataset configuration
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + './',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + './',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val.json',
        img_prefix=data_root + './',
        pipeline=test_pipeline)
)
evaluation = dict(interval=2, metric='keypoints')