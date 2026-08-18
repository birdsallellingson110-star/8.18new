_base_ = ['/home/lixiaob/cjy/rumpl_venv310/lib/python3.10/site-packages/mmpose/.mim/configs/_base_/default_runtime.py']

# Square crops/heatmaps are the AdaFuse H36M protocol.  The ResNet-152
# checkpoint is still the public COCO detector; only the sampling geometry is
# changed here so that the 2D front-end can be tested without changing the
# 3D evaluator.
train_cfg = dict(max_epochs=210, val_interval=10)
optim_wrapper = dict(optimizer=dict(type='Adam', lr=5e-4))
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001,
         by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=210,
         milestones=[170, 200], gamma=0.1, by_epoch=True),
]
auto_scale_lr = dict(base_batch_size=512)
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

codec = dict(type='MSRAHeatmap', input_size=(384, 384),
             heatmap_size=(96, 96), sigma=3)

model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=152,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet152')),
    head=dict(
        type='HeatmapHead',
        in_channels=2048,
        out_channels=17,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True))

dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]
train_dataloader = dict(batch_size=32, num_workers=2,
    persistent_workers=True, sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type, data_root=data_root,
        data_mode=data_mode, ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'), pipeline=train_pipeline))
val_dataloader = dict(batch_size=32, num_workers=2,
    persistent_workers=True, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type, data_root=data_root,
        data_mode=data_mode, ann_file='annotations/person_keypoints_val2017.json',
        bbox_file='data/coco/person_detection_results/'
                  'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'), test_mode=True, pipeline=val_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file=data_root +
                     'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator
