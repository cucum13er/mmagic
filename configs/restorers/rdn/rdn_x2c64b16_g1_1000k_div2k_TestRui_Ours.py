exp_name = 'rdn_x2c64b16_g1_1000k_div2k'

scale = 2
# model settings
model = dict(
    type='BasicRestorer',
    generator=dict(
        type='RDN',
        in_channels=3,
        out_channels=3,
        mid_channels=64,
        num_blocks=16,
        upscale_factor=scale),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'))
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
train_dataset_type = 'SROurDataset'
val_dataset_type = 'SROurDataset_val'
#val_dataset_type = 'SROurDataset'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=64),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='lq',
        flag='color',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFile',
        io_backend='disk',
        key='gt',
        flag='color',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            gt_folder='data/Ours/X2/',
            pipeline=train_pipeline,
            scale=scale,
            filename_tmpl = '{}'
            )),
    val=dict(
        type=val_dataset_type,
        gt_folder= 'data/Ours/X2/',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        gt_folder= 'data/Ours/X2/',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
total_iters = 1000000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[200000, 400000, 600000, 800000],
    gamma=0.5)

checkpoint_config = dict(interval=5000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=5000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]