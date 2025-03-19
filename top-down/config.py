config = {
    # Task specific
    "num_keypoints":    16,

    # Model architecture
    "swin_variant":     "swin_b",
    "head_hidden_dim":  1024,
    "pretrained":       True,
    "target_size":      (256, 256),

    ### Training
    "max_epoch":        12,
    "warmup_epochs":    2,

    # Dataloader
    "batch_size":       32,
    "num_workers":      14, 

    # Learning rate
    "warmup_lr_factor": 0.2,
    "swin_lr":          1e-5,
    "mlp_lr":           1e-3,

    # Loss weighting
    "coords_loss_weight":1,
    "vis_loss_weight":  1,

}
