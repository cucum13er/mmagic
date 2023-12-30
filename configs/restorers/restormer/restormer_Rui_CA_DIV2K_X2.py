# model settings
exp_name = 'restormer_CA_DIV2K'
scale = 2
embed_dim = 48
lq_size = 96
# model settings
model = dict(
    	type='BlindSR_MoCo',
        train_contrastive=False,
        pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
        # pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
        generator=dict(
            type='Restormer_HA',
            inp_channels=3,
            out_channels=3,
            dim=embed_dim,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',
            scale = scale,
            dual_pixel_task=False,
            fuse_type = 'CA',
        ),

        contrastive_part = dict(
                    type='MoCo_label',
                    queue_len=8192,
                    feat_dim=64,
                    momentum=0.999,
                    backbone=dict(
                        type='EasyRes',
                        in_channels=3,
                        # pretrained = '/home/rui/Rui_SR/mmselfsup/work_dirs/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth',
                        pretrained = '/work/pi_xiandu_umass_edu/ruima/pretrain/selfsup/moco/moco_easyres_epoch2000_temp0_07_DIV2K_supcon/weights_2000.pth',
                        ),
                    neck=dict(
                        type='MoCoV2Neck',
                        in_channels=512,
                        hid_channels=2048,
                        out_channels=64,
                        with_avg_pool=True),
                    head=dict(type='SNNLossHead', temperature=0.07)
                    ),
        contrastive_loss_factor = 0.1,
        )
# model training and testing settings
train_cfg = None
test_cfg = dict(metrics = ['PSNR','SSIM'], crop_border=scale)

# model training and testing settings
train_cfg = None
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=scale)

# dataset settings
# dataset settings
train_dataset_type = 'SRMultiFolderLabeledDataset'
#train_dataset_type = 'SRMultiFolderDataset'
# val_dataset_type = 'SRMultiFolderLabeledDataset'
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
    #### add another view ########################
    dict(type='CopyValues', src_keys=['lq','gt'], dst_keys=['lq_tmp','gt_tmp']),
    dict(type='PairedRandomCrop', gt_patch_size=lq_size*scale), # only crop lq and gt
    dict(type='CopyValues', src_keys=['lq','gt'], dst_keys=['lq_view','gt_view']), # another view
    dict(type='CopyValues', src_keys=['lq_tmp','gt_tmp'], dst_keys=['lq','gt']), # the whole image back
    dict(type='PairedRandomCrop', gt_patch_size=192), # crop the new lq and gt
    # random flip and transpose for both views
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['lq_view', 'gt_view'], flip_ratio=0.5,
        direction='horizontal'),    
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='Flip', keys=['lq_view', 'gt_view'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='RandomTransposeHW', keys=['lq_view', 'gt_view'], transpose_ratio=0.5),
    # collect all views and label
    dict(type='Collect', keys=['lq', 'gt' ,'lq_view', 'gt_view', 'gt_label'], meta_keys=['lq_path', 'gt_path', 'gt_label']),
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
    # dict(type='PairedRandomCrop', gt_patch_size=192), # crop the new lq and gt
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path']),
    dict(type='ImageToTensor', keys=['lq', 'gt'])
]

data = dict(
    workers_per_gpu=2,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_root = 'data/DIV2K_Flickr2K/lq/X2',
            # lq_folders=['data/MultiDegrade/DIV2K/X4/train/sig_0.5',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_01',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_02',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_03',
            # 		 'data/MultiDegrade/DIV2K/X4/train/sig_04',
            		# ],
            gt_folder='data/DIV2K_Flickr2K/gt',
            pipeline=train_pipeline,
            scale=scale,
            filename_tmpl = '{}'
            )),
    val=dict(
        type=val_dataset_type,
        lq_folders=[
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.5',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_01',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_02',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_03',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_04',
                    # 'data/Set5/X4/lq/sig_0.5',            		 
                    # 'data/Set5/X4/lq/sig_1.0',            		 
                    # 'data/Set5/X4/lq/sig_2.0',
                    # 'data/Set5/X4/lq/sig_3.0',
                    'data/Set5/X2/lq/sig_4.0',
                    # 'data/Set5/X4/lq/sig_1.0',
                    # 'data/BSD100/X4/lq/sig_1.0',
                    # 'data/Urban100/X4/lq/sig_4.0',
                   ],
        gt_folder= 'data/Set5/X2/gt/',
        # gt_folder= 'data/Urban100/X2/gt/',    
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'),
    test=dict(
        type=val_dataset_type,
        lq_folders=[
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_0.5',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_01',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_02',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_03',
                    # 'data/MultiDegrade/DIV2K_aniso/X4/test/sig_04',
                    # 'data/Set5/X4/lq/sig_0.5',            		 
                    #'data/Set5/X4/lq/sig_1.0',            		 
                    # 'data/Set5/X4/lq/sig_4.0',
                    'data/Set14/X2/lq/sig_4.0',
                    # 'data/Set5/X4/lq/sig_4.0',
                    # 'data/Set5/X2/lq/sig_4.0',
                    #'data/Set5/X4/lq/sig_1.0',
                    #'data/BSD100/X4/lq/sig_4.0',
                    # 'data/Urban100/X2/lq/sig_4.0',
                    # 'data/Urban100/X2/lq_aniso/sig_0.2_4.0theta_0.0',
                   ],
        # gt_folder= 'data/Set5/X2/gt/',
        gt_folder= 'data/Set14/X2/gt/',
        # gt_folder= 'data/Urban100/X2/gt/',   
        #gt_folder= 'data/BSD100/X4/gt/',
        pipeline=test_pipeline,
        scale=scale,
        filename_tmpl='{}'))

# optimizer
optimizers = dict(
                 # type='Adam', lr=1e-2, betas=(0.9, 0.999),
                 # contrastive_part=dict(type='LARS', lr=4.8, weight_decay=1e-6, momentum=0.9),
                 contrastive_part=dict(type='Adam', lr=1e-9, betas=(0.9, 0.999) ),
                 generator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)),


        )

# learning policy
total_iters = 160000
lr_config = dict(
    policy='Step',
    by_epoch=False,
    step=[40000, 80000, 120000],
    gamma=0.5)

checkpoint_config = dict(interval=20000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=40000, save_image=True, gpu_collect=True)
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None #'/work/pi_xiandu_umass_edu/ruima/pretrain/Restormer/restormer_Rui_pretrainFromOfficial.pth'
resume_from = None# '/home/rui/Rui_SR/mmediting/work_dirs/restorers/hasr/X4/hasr_0808_fromX2pretrain/iter_60000.pth'
workflow = [('train', 1)]