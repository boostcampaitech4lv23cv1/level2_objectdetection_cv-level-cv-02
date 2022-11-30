_base_ = './retinanet_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=152,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet152')))
