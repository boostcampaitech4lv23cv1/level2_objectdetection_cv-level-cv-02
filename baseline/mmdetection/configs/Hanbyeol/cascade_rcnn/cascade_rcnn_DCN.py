_base_ = [
    # './_base_/datasets/custom_dataset_detection.py',
    './_base_/datasets/coco_detection.py',
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/schedules/schedule_1x.py',
    './_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
