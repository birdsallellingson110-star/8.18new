"""17-joint evaluation adapter for a checkpoint trained with AdaFuse's
20-joint union head.

The training config uses 20 channels to match the released AdaFuse H36M/MPII
union interface.  ``convert_ada_union20_h36m17_20260814.py`` extracts the 17
valid H36M channels for the existing heatmap exporter and triangulation audit.
"""

_base_ = ['./td-hm_res152_8xb32-210e_lt_h36m_384x384_noflip_bgr.py']

model = dict(
    head=dict(out_channels=17),
    test_cfg=dict(flip_test=False, shift_heatmap=False),
)
