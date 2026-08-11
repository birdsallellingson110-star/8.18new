"""第四查: 逐元素对比两条 ray 构造路径。
  A = build_rays_one_view (clip dataloader 用)
  B = baseline JointsDataset_RUMPL.create_3d_ray_coords + generate_direction (真 baseline 代码)
同一份原始 (joints_2d, camera), 看 direction / intersection 逐元素差。
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import pickle
import numpy as np
from dataset.stvft_dataset import build_rays_one_view, normalize_screen, IMAGE_W, IMAGE_H
from dataset.joints_dataset_rumpl import JointsDataset_RUMPL

PKL = "/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/split_0_chunk_0000.pkl"
d = pickle.load(open(PKL, 'rb'))

def baseline_rays(j2d_raw, cam):
    """复刻 baseline get_rays 的归一化 + create_3d_ray_coords + generate_direction (真 baseline 方法)。"""
    o = JointsDataset_RUMPL.__new__(JointsDataset_RUMPL)
    o.intersection_ray_with = 'Closest'
    o.ray_as_intersection_with_donut = False
    o.use_t = False
    o.use_grid = False
    o.downsample = 1
    o.bug_test = False
    o.num_joints = 17
    W, H = IMAGE_W, IMAGE_H   # baseline image_size = NETWORK.IMAGE_SIZE = [1920,1080]
    cam2 = {k: (np.array(v).copy() if not np.isscalar(v) else v) for k, v in cam.items()}
    # NORMALIZE_CAMERAS (get_rays 671-688)
    cc = o.normalize_screen_coordinates(np.array([float(cam['cx']), float(cam['cy'])]), W, H)
    cam2['cx'] = cc[0]; cam2['cy'] = cc[1]
    cam2['fx'] = float(cam['fx']) / W * 2
    cam2['fy'] = float(cam['fy']) / W * 2
    cam2['R'] = np.array(cam['R'])
    cam2['T'] = np.array(cam['T']).reshape(3, 1)
    # INPUTS_NORMALIZED (get_rays 760)
    j_n = o.normalize_screen_coordinates(j2d_raw.astype(np.float64).copy(), W, H)
    trans_inv = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # no_augmentation
    coords_world = o.create_3d_ray_coords(cam2, trans_inv, j_n.copy())
    cam_center = cam2['T'].T   # (1,3)
    dirs, inters = o.generate_direction_vectors_and_intersection_points(coords_world, cam_center)
    return np.asarray(dirs, np.float64), np.asarray(inters, np.float64)

print("=== 逐元素对比 build_rays_one_view (A) vs baseline 真函数 (B) ===")
for (ci, fi, vi) in [(0, 2, 0), (0, 2, 1), (3, 10, 5), (7, 0, 12)]:
    cam = d['camera_parameters_all'][ci][vi]
    j2d = d['joints_2d_mmpose'][ci, fi, vi].astype(np.float64)  # (17,2)
    dA, iA = build_rays_one_view(j2d, cam)
    dB, iB = baseline_rays(j2d, cam)
    dd = np.abs(dA.astype(np.float64) - dB)
    di = np.abs(iA.astype(np.float64) - iB)
    print(f"\nclip{ci} frame{fi} view{vi}:")
    print(f"  direction   max|A-B|={dd.max():.3e}  mean={dd.mean():.3e}  |A|~{np.abs(dA).mean():.3f}")
    print(f"  intersection max|A-B|={di.max():.3e}  mean={di.mean():.3e}  |A|~{np.abs(iA).mean():.3f}")
    # 若差大, 打印第0关节两边的值
    if dd.max() > 1e-3 or di.max() > 1e-3:
        print(f"  [j0] dirA={np.round(dA[0],4)} dirB={np.round(dB[0],4)}")
        print(f"  [j0] intA={np.round(iA[0],4)} intB={np.round(iB[0],4)}")
