"""Control variant: public LT checkpoint with RGB preprocessing.

This is retained only as an input-protocol ablation.  The official LT
pipeline uses BGR, so the BGR config is the protocol-matched candidate.
"""

_base_ = ['./td-hm_res152_8xb32-210e_coco-384x384.py']

model = dict(
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    test_cfg=dict(flip_test=False, shift_heatmap=False),
)
