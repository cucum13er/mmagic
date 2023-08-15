_base_ = [
    '../_base_/default_runtime.py',    
]

experiment_name = 'reformer_temp'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = f'./work_dirs/{experiment_name}/'
scale = 2
# model settings
model = dict(
    type='BaseEditModel',
    generator=dict(
        type='Reformer',
        inp_channels=3,
        out_channels=3,
        num_blocks=[8, 8],
        heads = [8, 8],
        dim=768,
        image_size=224,
        patch_size=16,
        ffn_expansion_factor=4,
        bias=False,
        scale=scale,
        # keys=['imgLR', 'imgRef', 'prior_regis'],
        ),
    pixel_loss=dict(type='L1Loss', loss_weight=1.0, reduction='mean'),
    regis_loss=dict(type='Registration_loss', loss_weight = 1.0, padding_size = 16, patch_size = 16),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        )
    )
    
# dataset settings
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='lr',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='ref',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt_img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),   
    # dict(
    #     type='PairedRandomCrop', 
    #     gt_patch_size=224, 
    #     lq_key='lr', 
    #     gt_key='hr',
    #     scale = scale,
    #     ),
    # dict(
    #     type='Flip',
    #     keys=['lr', 'hr', 'ref'],
    #     flip_ratio=0.5,
    #     direction='horizontal'),
    # dict(
    #     type='Flip', 
    #     keys=['lr', 'hr', 'ref'], 
    #     flip_ratio=0.5, 
    #     direction='vertical'),
    # dict(
    #     type='RandomTransposeHW', 
    #     keys=['lr', 'hr', 'ref'], 
    #     transpose_ratio=0.5),
    dict(type='PackInputs',
          keys=['lr','ref','gt_img','regis'],
          meta_keys=['lr_path','ref_path','gt_img_path'],
          data_keys= ['lr','ref','gt_img','regis'],
          )
]
val_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='lr',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt_img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='ref',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackInputs',
          keys=['lr','ref','gt_img','regis'],
          meta_keys=['lr_path','ref_path','gt_img_path'],
          data_keys= ['lr','ref','gt_img','regis'],
          )
]
test_pipeline = val_pipeline 
# dataset settings
dataset_type = 'RefImageDataset'
data_root = '/media/rui/Samsung4TB/Datasets/testdata2_choose/'

train_dataloader = dict(
    num_workers=4,
    batch_size=64,
    drop_last=True,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline),
        )

val_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline,
          )
        )

val_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ])

test_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=val_pipeline,
          )
        )

test_evaluator = dict(
    type='Evaluator',
    metrics=[
        dict(type='MAE'),
        dict(type='PSNR', crop_border=scale),
        dict(type='SSIM', crop_border=scale),
    ])

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=3000, val_interval=500)
val_cfg = dict(type='MultiValLoop')
test_cfg = dict(type='MultiTestLoop')

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))

# learning policy
param_scheduler = dict(
    type='MultiStepLR', by_epoch=False, milestones=[200000], gamma=0.5)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=500,
        save_optimizer=True,
        by_epoch=False,
        out_dir=save_dir,
    ),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='Reformer_PreRegis',
         lq_key='lr', 
         gt_key='gt_img', 
         ref_key='ref',  
         )
]
