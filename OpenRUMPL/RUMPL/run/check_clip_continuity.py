"""查 clip 连续帧是否真连续 (gate↑loss↑ 的最便宜最致命假设)。

打印某关节在 27 帧的 3D 坐标 + 逐帧步长。真连续=步长平滑小; 假连续=跳变。
joints_3d: (N_clip, 27, 17, 3) 米制。
"""
import argparse
import glob
import pickle
import numpy as np

COCO = ['nose','leye','reye','lear','rear','lsho','rsho','lelb','relb',
        'lwri','rwri','lhip','rhip','lkne','rkne','lank','rank']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--glob', default="/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl")
    ap.add_argument('--joint', default='rwri', help='关节名')
    ap.add_argument('--clip', type=int, default=0, help='打印第几个 clip 的完整轨迹')
    ap.add_argument('--n-clips-stats', type=int, default=20, help='统计步长用的 clip 数')
    args = ap.parse_args()

    j = COCO.index(args.joint)
    f = sorted(glob.glob(args.glob))[0]
    d = pickle.load(open(f, 'rb'))
    j3d = d['joints_3d']                     # (N,27,17,3)
    print(f"file: {f}")
    print(f"joints_3d shape: {j3d.shape}  (N_clip, T, J, 3)  joint={args.joint}(idx{j})")

    traj = j3d[args.clip, :, j, :]           # (27,3) 米
    print(f"\n=== clip {args.clip} 关节 {args.joint} 27 帧 3D 坐标 (米) ===")
    for t in range(traj.shape[0]):
        print(f"  t={t:2d}: [{traj[t,0]:+.4f}, {traj[t,1]:+.4f}, {traj[t,2]:+.4f}]")

    steps = np.linalg.norm(np.diff(traj, axis=0), axis=1)   # (26,) 逐帧步长 米
    print(f"\n=== 逐帧步长 ||x[t+1]-x[t]|| (米) ===")
    print("  " + " ".join(f"{s:.4f}" for s in steps))
    print(f"  step: mean={steps.mean():.4f} max={steps.max():.4f} min={steps.min():.4f} "
          f"std={steps.std():.4f}")
    # 跳变指标: 最大步长 / 中位步长。平滑运动该接近 1~几; 跳变会很大。
    med = np.median(steps)
    print(f"  jumpiness max/median = {steps.max()/(med+1e-9):.2f} (平滑~几; 跳变会很大)")

    # 跨多个 clip 的步长统计 (排除单 clip 偶然)
    N = min(args.n_clips_stats, j3d.shape[0])
    allsteps = np.linalg.norm(np.diff(j3d[:N, :, j, :], axis=1), axis=2)  # (N,26)
    print(f"\n=== {N} 个 clip 的 {args.joint} 步长汇总 (米) ===")
    print(f"  mean={allsteps.mean():.4f} max={allsteps.max():.4f} "
          f"p99={np.percentile(allsteps,99):.4f} median={np.median(allsteps):.4f}")


if __name__ == '__main__':
    main()
