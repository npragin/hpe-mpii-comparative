_base_ = './petr_r50_16x2_100e_mpii.py'
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'  # noqa
model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        in_channels=[192, 384, 768]))

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=12)
# optimizer
optimizer = dict(lr=1e-4)

# learning policy Not used
# lr_config = dict(
#     policy='CosineAnnealing',  # Using cosine annealing
#     warmup='linear',           # Linear warm-up
#     warmup_iters=500,          # Number of iterations for warm-up
#     warmup_ratio=0.1,          # Starting ratio of the base lr
#     by_epoch=False,            # If True, `warmup_iters` is in epochs, otherwise in iterations
#     min_lr=1e-6                # Minimum learning rate during annealing)
# )

runner = dict(type='EpochBasedRunner', max_epochs=50)
checkpoint_config = dict(interval=2, max_keep_ckpts=5)