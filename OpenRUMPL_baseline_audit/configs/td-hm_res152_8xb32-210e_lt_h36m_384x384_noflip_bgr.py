"""MMPose adapter for the public Learnable-Triangulation H36M frontend.

The LT checkpoint is trained by reading OpenCV images (BGR) and applying the
ImageNet normalization without a BGR->RGB conversion.  Its 17 output channels
are in the LT/H36M label order, not COCO order, so test-time horizontal flip
augmentation must be disabled as well.
"""

_base_ = ['./td-hm_res152_8xb32-210e_coco-384x384.py']

model = dict(
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    head=dict(out_channels=20, loss=dict(type='KeypointMSELoss', use_target_weight=True)),
    test_cfg=dict(flip_test=False, shift_heatmap=False),
)
