_base_ = [
     '../../configs/_base_/datasets/voc0712.py',
    '../../configs/_base_/schedules/schedule_2x.py', '../../configs/_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

runner = dict(type='FreeRobustRunner', max_epochs=30)

# model settings
input_size = 300
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='ResNet',
        depth=50,  # 选取Resnet50
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        stage_with_dcn=(False, False, False),
        num_stages=3,  # ResNet 系列包括 stem+ 4个 stage 输出
        out_indices=(1, 2),  # 输出的特征图索引,(3,)表示仅使用了C5的输出
        frozen_stages=-1,  # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        norm_cfg=dict(type='BN', requires_grad=True),  # BN 层的beta和gamma不更新
        norm_eval=True,  # BN 层的均值和方差都直接采用全局预训练值，不进行更新
        style='pytorch',  # caffe 和 PyTorch 是指 Bottleneck 模块的区别，默认pytorch
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=20,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
        ])
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(300, 300),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


# optimizer
# optimizer = dict(type='SGD', lr=1e-4, momentum=0.9, weight_decay=1e-4)

# optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0001,
#                  paramwise_cfg = dict(custom_keys={'backbone':dict(lr_mult=0.2,decay_mult=1.0)}))

# optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='SGD', lr=5e-3, momentum=0.9, weight_decay=0.0001,
                 paramwise_cfg = dict(custom_keys={'backbone':dict(lr_mult=0.2,decay_mult=1.0)}))

# optimizer = dict(type='SGD', lr=5e-4, momentum=0.9, weight_decay=5e-4)
# optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001,
#                  paramwise_cfg = dict(custom_keys={'backbone':dict(lr_mult=0.2,decay_mult=1.0)}))
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(_delete_=True)
optimizer_config = dict(type=('FreeRobustOptimizerHook'))

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=[
                data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            ],
            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
