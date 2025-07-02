_base_ = [
    'D:/project/Deeplabv3/base/deeplabv3.py',
    'D:/project/Deeplabv3/base/dataset.py',
    'D:/project/Deeplabv3/base/runtime.py',
    'D:/project/Deeplabv3/base/1000.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet18_v1c',
    backbone=dict(depth=18),
    decode_head=dict(in_channels=512,
    channels=128,
    num_classes=3),
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=3))

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')