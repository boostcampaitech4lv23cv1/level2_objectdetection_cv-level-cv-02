# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/' 

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=30, 
        interpolation=1,
        p=0.3),
    dict(
        type="HorizontalFlip",
        p=0.5
    ),
    dict(
        type="GaussianBlur",
        p=0.3
    ),
    dict(
        type="Emboss",
        p=0.3
    ),
    dict(
        type="Sharpen",
        p=0.3
    ),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0), 
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.2),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.08),
    dict(type='CLAHE', clip_limit=4, p=0.3),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=[(1024, 1024), (800, 800)], multiscale_mode='range', keep_ratio=True), # ì´ë¯¸ì§€ size!
    dict(type='Pad', size_divisor=32),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        # keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024), # ìš°ë¦¬ ë°ì´í„°ì— ë§žì¶°ì„œ
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=3, # batch size 2 -> 4ë¡œ ë³€ê²½ 
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes, # ìš°ë¦¬ ë°ì´í„°ëŒ€ë¡œ ì¶”ê°€
        ann_file=data_root + 'train01.json', # ì•„ì§ CV ì•ˆ ë‚˜ëˆ´ìœ¼ë¯€ë¡œ ì „ì²´ json ë„˜ê¹€
        img_prefix=data_root, # ìˆ˜ì •
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes, # ìš°ë¦¬ ë°ì´í„°ëŒ€ë¡œ ì¶”ê°€
        ann_file=data_root + 'val01.json', # ìˆ˜ì •
        img_prefix=data_root, # ìˆ˜ì •
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes, # ìš°ë¦¬ ë°ì´í„°ëŒ€ë¡œ ì¶”ê°€
        ann_file=data_root + 'test.json', # ìˆ˜ì •
        img_prefix=data_root, # ìˆ˜ì •
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', classwise=True, save_best='bbox_mAP_50')