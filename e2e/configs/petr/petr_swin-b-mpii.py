_base_ = './petr_r50_16x2_100e_mpii.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[256, 512, 1024]))

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=12)
# optimizer
optimizer = dict(lr=1e-4)

# learning policy # Didn't end up using this as it was worse
# lr_config = dict(
#     policy='CosineAnnealing',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=0.1,
#     by_epoch=False,
#     min_lr=1e-6
# )

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=2, max_keep_ckpts=5)