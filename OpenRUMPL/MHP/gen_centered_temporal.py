"""生成"中心对齐"的连续时序 CMU 数据: 中心帧=官方测试帧(baseline=40.40干净), 邻帧=±4密集视频帧(HRNet匹配)。
pose5+pose6, 5相机, L=9。中心用官方entry, 邻帧从hdVideos抽+HRNet+匹配center, 全部swap(06_swap约定)。
输出 cmu_panoptic_validation.pkl(win_tensors/windows_for_seq可读)。
"""
import pickle, json, cv2, numpy as np, os, argparse
from collections import defaultdict

P = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]); Pt = P.T
def swap(s):
    s = dict(s); j = np.asarray(s['joints_3d'], float); s['joints_3d'] = (P @ j.T).T.astype(np.float32)
    cam = dict(s['camera']); R = np.asarray(cam['R'], float); T = np.asarray(cam['T'], float).reshape(3, 1)
    Rn = R @ Pt; Tn = P @ T; cam['R'] = Rn; cam['T'] = Tn; cam['t'] = -Rn @ Tn; s['camera'] = cam; return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--off', default="/mnt/data/cjydata/cmu_singleperson/MPL_data/datasets_mmpose/annot_pose56_5cams_coco_filtered_1_1_mmpose_hrnet_coco_matched/cmu_panoptic_validation.pkl")
    ap.add_argument('--vid-root', default="/mnt/data/cjydata/cmu_singleperson")
    ap.add_argument('--out', default="/mnt/data/cjydata/cmu_temporal_centered/full")
    ap.add_argument('--seqs', nargs='+', default=['171204_pose5', '171204_pose6'])
    ap.add_argument('--L', type=int, default=9)
    args = ap.parse_args()
    CAMS = {3: '03', 6: '06', 12: '12', 13: '13', 23: '23'}; H = args.L // 2
    os.makedirs(args.out, exist_ok=True)

    d = pickle.load(open(args.off, 'rb'))
    off = defaultdict(dict)
    for e in d:
        off[e['pose_id']][(e['image_id'], e['camera_id'])] = e
    centers = {pid: sorted(set(i for (i, c) in off[pid] if c == 3)) for pid in args.seqs}
    print(f"中心帧: {[(p, len(centers[p])) for p in args.seqs]}")

    from mmpose.apis import MMPoseInferencer
    inf = MMPoseInferencer(pose2d='td-hm_hrnet-w32_8xb64-210e_coco-384x288', device='cuda:0')

    entries = []
    for pid in args.seqs:
        for cid, cs in CAMS.items():
            cap = cv2.VideoCapture(os.path.join(args.vid_root, pid, 'hdVideos', f'hd_00_{cs}.mp4'))
            nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for c in centers[pid]:
                ce = off[pid].get((c, cid))
                if ce is None:
                    continue
                c2d = np.asarray(ce['joints_2d'])[:, :2]
                cap.set(cv2.CAP_PROP_POS_FRAMES, c - H)               # 一次seek读L帧连续
                for o in range(-H, H + 1):
                    iid = c + o
                    if o == 0:
                        entries.append(swap(ce)); cap.grab(); continue   # 中心=官方, 推进读指针
                    ok, fr = cap.read()
                    if not ok or iid < 0 or iid >= nframes:
                        continue
                    res = next(inf(fr, return_vis=False)); dets = res['predictions'][0]
                    if not dets:
                        continue
                    best = min(dets, key=lambda p: np.linalg.norm(np.asarray(p['keypoints'])[:, :2] - c2d))
                    e = dict(ce)
                    e['joints_2d'] = np.asarray(best['keypoints'])[:, :2].astype(np.float32)
                    e['joints_2d_conf'] = np.asarray(best['keypoint_scores']).reshape(-1, 1).astype(np.float32)
                    e['image_id'] = iid
                    entries.append(swap(e))
            cap.release()
            print(f"[{pid}/00_{cs}] 完成, 累计 {len(entries)} entries", flush=True)
    outp = os.path.join(args.out, 'cmu_panoptic_validation.pkl')
    pickle.dump(entries, open(outp, 'wb'))
    print(f"=== 完成: {len(entries)} entries → {outp} ===")


if __name__ == '__main__':
    main()
