_base_ = [
    '../../configs/_base_/models/ssd300.py', '../../configs/_base_/datasets/voc0712.py',
    '../../configs/_base_/schedules/schedule_2x.py', '../../configs/_base_/default_runtime.py'
]
# dataset settings
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'
model = dict(bbox_head = dict(num_classes = 20))
runner = dict(type='AdvClockRunner', max_epochs=100)
# find_unused_parameters = True

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
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

test_pipeline = [
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

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(300, 300),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=False),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='DefaultFormatBundle'),
#             dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
#         ])
# ]

# optimizer
optimizer = dict(type='SGD', lr=2e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(type=('AdvClockOptimizerHook'))
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    shuffle = False,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            # ann_file=[
            #     data_root + 'VOC2007/ImageSets/Main/trainval.txt',
            #     data_root + 'VOC2012/ImageSets/Main/trainval.txt'
            # ],

            ann_file=[
                "robustdetector/utils/person_only_anno_2007.txt",
                "robustdetector/utils/person_only_anno_2012.txt"
            ],

            img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file= "robustdetector/utils/person_only_anno_2007.txt",
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file= "robustdetector/utils/person_only_anno_2007_test.txt",
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline))

# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=4,
#     shuffle = False,
#     train=dict(
#         type='RepeatDataset',
#         times=3,
#         dataset=dict(
#             type=dataset_type,
#             # ann_file=[
#             #     data_root + 'VOC2007/ImageSets/Main/trainval.txt',
#             #     data_root + 'VOC2012/ImageSets/Main/trainval.txt'
#             # ],
#
#             ann_file=[
#                 "robustdetector/utils/person_only_anno_2007.txt",
#                 "robustdetector/utils/person_only_anno_2012.txt"
#             ],
#
#             img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
#             pipeline=train_pipeline)),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'VOC2007/',
#         pipeline=val_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
#         img_prefix=data_root + 'VOC2007/',
#         pipeline=test_pipeline))


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=32*2)
