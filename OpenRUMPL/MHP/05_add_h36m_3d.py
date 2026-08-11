"""纯 CPU: 从 clip 数据存的 SMPL+H 参数算 h36m 3D GT 关节 (不重渲染、不碰 GPU)。

为后面"三数据集对标(H36M)"备料。复用 02_clip_run.py 的 SMPL+H forward 逻辑,
唯一区别 = 用 J_regressor_h36m 而非 coco。

自带验证: 同时用 J_regressor_coco 重算 coco 3D, 与 pkl 里存的 coco 3D 对比。
  - clip pkl 没存手部 pose (forward 需 156 维, 缺 90 维手部 → 置 0)。
  - 若重算 coco 3D ≈ 存的 coco 3D → 手部置0 无影响 + forward 正确 → h36m 3D 可信。

用法: python MHP/05_add_h36m_3d.py [--limit-files N] [--validate-only]
输出: 每个 train pkl 旁生成 <name>_h36m3d.pkl, 含 {'joints_3d_h36m': (N,27,Jh,3), 'fnames': ...}
"""
import argparse, glob, os, pickle
import numpy as np
import torch

device = torch.device('cpu')   # 纯 CPU
TRAIN_GLOB = "/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl"
SUPPORT = "/mnt/data/dataset/c2i/rumpl_support"
OUT_DIR = "/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train_h36m3d"

from human_body_prior.body_model.body_model import BodyModel


def load_bms():
    bm = {}
    for gid, gname in [(0, 'male'), (1, 'female'), (2, 'neutral')]:
        p = os.path.join(SUPPORT, f'body_models/smplh/{gname}/model.npz')
        bm[gid] = BodyModel(bm_fname=p, num_betas=16).to(device)
    return bm


def smplh_vertices(bm, gender, global_orient, body_pose, transl, betas):
    """复刻 02 的 smplh_forward_batch, 但手部=0 (clip 未存手部). 返回 (L,6890,3)."""
    L = global_orient.shape[0]
    root = torch.from_numpy(global_orient).float()           # (L,3)
    pb = torch.from_numpy(body_pose).float()                 # (L,63)
    hand = torch.zeros(L, 90)                                 # 手部置0 (未存)
    t = torch.from_numpy(transl).float()
    b = torch.from_numpy(betas).float().unsqueeze(0).expand(L, -1)
    out = bm[int(gender)].forward(root_orient=root, pose_body=pb, pose_hand=hand, trans=t, betas=b)
    return out.v.detach().numpy()                            # (L,6890,3)


def kabsch(A, B):
    """求刚体变换 (R,t) 使 R@A^T+t ≈ B. A,B: (N,3). 返回 R(3,3),t(3,),残差rmse."""
    cA, cB = A.mean(0), B.mean(0)
    H = (A - cA).T @ (B - cB)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cB - R @ cA
    res = np.sqrt(((R @ A.T).T + t - B) ** 2).sum(1).mean()
    return R, t, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit-files', type=int, default=None)
    ap.add_argument('--validate-only', action='store_true', help='只跑首个文件验证, 不写盘')
    args = ap.parse_args()

    Jc = np.load(os.path.join(SUPPORT, 'J_regressor_coco.npy'))   # (Jc,6890)
    Jh = np.load(os.path.join(SUPPORT, 'J_regressor_h36m.npy'))   # (Jh,6890)
    print(f"[regressor] coco {Jc.shape}  h36m {Jh.shape}")
    bm = load_bms()

    files = sorted(glob.glob(TRAIN_GLOB))
    if args.limit_files:
        files = files[:args.limit_files]
    os.makedirs(OUT_DIR, exist_ok=True)

    max_coco_diff = 0.0
    for fi, f in enumerate(files):
        d = pickle.load(open(f, 'rb'))
        N = d['joints_3d'].shape[0]
        h36m_all = []
        for i in range(N):
            v = smplh_vertices(bm, d['smplh_gender'][i], d['smplh_global_orient'][i],
                               d['smplh_body_pose'][i], d['smplh_transl'][i], d['smplh_betas'][i])  # (27,6890,3)
            coco_raw = np.einsum('jv,lvc->ljc', Jc, v)   # (27,17,3) 原始世界系
            h36m_raw = np.einsum('jv,lvc->ljc', Jh, v)   # (27,17,3) 原始世界系
            stored = d['joints_3d'][i]                   # (27,17,3) room系(目标)
            # 用 (原始coco → 存coco) 估每clip刚体变换, 应用到 h36m
            R, t, res = kabsch(coco_raw.reshape(-1, 3), stored.reshape(-1, 3))
            h36m_room = (R @ h36m_raw.reshape(-1, 3).T).T + t  # 对齐到room系
            h36m_all.append(h36m_room.reshape(27, -1, 3).astype(np.float32))
            max_coco_diff = max(max_coco_diff, res)      # Kabsch残差=验证量
        if args.validate_only or fi == 0:
            print(f"[validate] file{fi} {os.path.basename(f)}: N={N} "
                  f"Kabsch残差(原始coco对齐存coco) max={max_coco_diff:.4f} m "
                  f"(应~mm级; 小=forward正确+变换刚体, h36m可信)")
            print(f"           h36m 3D shape per-clip = (27,{Jh.shape[0]},3)  示例关节0(room系)={h36m_all[0][0,0]}")
            if args.validate_only:
                return
        out = os.path.join(OUT_DIR, os.path.basename(f).replace('.pkl', '_h36m3d.pkl'))
        with open(out, 'wb') as fo:
            pickle.dump({'joints_3d_h36m': np.stack(h36m_all)}, fo)
        if fi % 20 == 0:
            print(f"  [{fi+1}/{len(files)}] saved {os.path.basename(out)}")
    print(f"[done] {len(files)} files → {OUT_DIR}; coco重算vs存 全局 max|diff|={max_coco_diff:.4f} m")


if __name__ == '__main__':
    main()
