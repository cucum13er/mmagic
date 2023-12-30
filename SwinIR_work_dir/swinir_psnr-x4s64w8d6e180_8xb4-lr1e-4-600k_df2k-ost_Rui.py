exp_name = 'dasr_x4c64b16_g1_100k_div2k'
scale = 4
img_size = 48
model = dict(
    type='BlindSR_MoCo',
    train_contrastive=False,
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    generator=dict(
        type='SwinIRNet_HA',
        upscale=4,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv',
        fuse_rstb=True,
        fuse_basic=False),
    contrastive_part=dict(
        type='MoCo_label',
        queue_len=8192,
        feat_dim=64,
        momentum=0.999,
        backbone=dict(
            type='EasyRes',
            in_channels=3,
            pretrained=
            '/work/pi_xiandu_umass_edu/ruima/pretrain/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth'
        ),
        neck=dict(
            type='MoCoV2Neck',
            in_channels=512,
            hid_channels=2048,
            out_channels=64,
            with_avg_pool=True),
        head=dict(type='SNNLossHead', temperature=0.07)),
    contrastive_loss_factor=0.1)
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=4)
train_dataset_type = 'SRMultiFolderLabeledDataset'
val_dataset_type = 'SRMultiFolderDataset'
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
    dict(
        type='CopyValues',
        src_keys=['lq', 'gt'],
        dst_keys=['lq_tmp', 'gt_tmp']),
    dict(type='PairedRandomCrop', gt_patch_size=192),
    dict(
        type='CopyValues',
        src_keys=['lq', 'gt'],
        dst_keys=['lq_view', 'gt_view']),
    dict(
        type='CopyValues',
        src_keys=['lq_tmp', 'gt_tmp'],
        dst_keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=192),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip',
        keys=['lq_view', 'gt_view'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(
        type='Flip',
        keys=['lq_view', 'gt_view'],
        flip_ratio=0.5,
        direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(
        type='RandomTransposeHW',
        keys=['lq_view', 'gt_view'],
        transpose_ratio=0.5),
    dict(
        type='Collect',
        keys=['lq', 'gt', 'lq_view', 'gt_view', 'gt_label'],
        meta_keys=['lq_path', 'gt_path', 'gt_label']),
    dict(type='ImageToTensor', keys=['lq', 'gt', 'lq_view', 'gt_view'])
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
    dict(type='PairedRandomCrop', gt_patch_size=192),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]
data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=2, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type='SRMultiFolderLabeledDataset',
            lq_root='data/DIV2K_Flickr2K/lq/X4',
            gt_folder='data/DIV2K_Flickr2K/gt',
            pipeline=[
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
                dict(
                    type='CopyValues',
                    src_keys=['lq', 'gt'],
                    dst_keys=['lq_tmp', 'gt_tmp']),
                dict(type='PairedRandomCrop', gt_patch_size=192),
                dict(
                    type='CopyValues',
                    src_keys=['lq', 'gt'],
                    dst_keys=['lq_view', 'gt_view']),
                dict(
                    type='CopyValues',
                    src_keys=['lq_tmp', 'gt_tmp'],
                    dst_keys=['lq', 'gt']),
                dict(type='PairedRandomCrop', gt_patch_size=192),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['lq_view', 'gt_view'],
                    flip_ratio=0.5,
                    direction='horizontal'),
                dict(
                    type='Flip',
                    keys=['lq', 'gt'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(
                    type='Flip',
                    keys=['lq_view', 'gt_view'],
                    flip_ratio=0.5,
                    direction='vertical'),
                dict(
                    type='RandomTransposeHW',
                    keys=['lq', 'gt'],
                    transpose_ratio=0.5),
                dict(
                    type='RandomTransposeHW',
                    keys=['lq_view', 'gt_view'],
                    transpose_ratio=0.5),
                dict(
                    type='Collect',
                    keys=['lq', 'gt', 'lq_view', 'gt_view', 'gt_label'],
                    meta_keys=['lq_path', 'gt_path', 'gt_label']),
                dict(
                    type='ImageToTensor',
                    keys=['lq', 'gt', 'lq_view', 'gt_view'])
            ],
            scale=4,
            filename_tmpl='{}')),
    val=dict(
        type='SRMultiFolderDataset',
        lq_folders=['data/Set5/X4/lq/sig_1.0'],
        gt_folder='data/Set5/X4/gt/',
        pipeline=[
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
            dict(type='PairedRandomCrop', gt_patch_size=192),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=4,
        filename_tmpl='{}'),
    test=dict(
        type='SRMultiFolderDataset',
        lq_folders=['data/BSD100/X4/lq/sig_3.0'],
        gt_folder='data/BSD100/X4/gt/',
        pipeline=[
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
            dict(type='PairedRandomCrop', gt_patch_size=192),
            dict(
                type='Collect',
                keys=['lq', 'gt'],
                meta_keys=['lq_path', 'gt_path']),
            dict(type='ImageToTensor', keys=['lq', 'gt'])
        ],
        scale=4,
        filename_tmpl='{}'))
optimizers = dict(
    contrastive_part=dict(type='Adam', lr=1e-09, betas=(0.9, 0.999)),
    generator=dict(type='Adam', lr=0.0001, betas=(0.9, 0.999)))
total_iters = 80000
lr_config = dict(
    policy='Step', by_epoch=False, step=[20000, 40000, 60000], gamma=0.5)
checkpoint_config = dict(interval=20000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=40000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '--work-dir'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpus = 1
