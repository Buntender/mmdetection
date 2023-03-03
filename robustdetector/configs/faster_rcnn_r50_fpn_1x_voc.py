_base_ = [
    '../../configs/_base_/models/faster_rcnn_r50_fpn.py',
    '../../configs/_base_/datasets/voc0712.py',
    '../../configs/_base_/schedules/schedule_1x.py', '../../configs/_base_/default_runtime.py'
]

model = dict(roi_head = dict(bbox_head = dict(num_classes = 20)))
