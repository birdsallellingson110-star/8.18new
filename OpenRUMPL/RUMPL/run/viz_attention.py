"""注意力可视化(对标 PoseFormer Fig5/6), 诊断 STVFTv2 学到了什么。
图2 Temporal: 中心帧(t=L//2) query 在 attend 哪些帧 — 理想: 中心崩→借邻帧, 中心好→信自己。
图1 VFT: fusion token 跨视角的选择性 — V=2 一视角崩时是否偏好好视角。
用 CMU 真实 clip: 一个"中心帧好"、一个"中心帧崩(邻帧好)"对比。
"""
import argparse, os, sys, pickle
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.stvft.stvft_v2 import STVFTv2
from run.cmu_eval_v2 import windows_for_seq, win_tensors, ALLV

OUT = '/home/lixiaob/cjy/viz_attn'


def vft_attn_hook(store):
    def hook(module, inp, out):
        x = inp[0]                                          # (N, V+1, C) normed
        N, T, C = x.shape
        qkv = module.qkv(x).reshape(N, T, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        attn = ((q @ k.transpose(-2, -1)) * module.scale).softmax(-1)   # (N, nh, T, T)
        store.append(attn.detach())
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--pkl', required=True)
    ap.add_argument('--cfg', default='configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml')
    ap.add_argument('--L', type=int, default=9)
    ap.add_argument('--views', nargs='+', type=int, default=[3, 6])     # V=2 研究口径
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    dev = args.device; ct = args.L // 2; os.makedirs(OUT, exist_ok=True)

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    tl = ck.get('args', {}).get('temporal_layers', 2)
    model = STVFTv2(args.cfg, ck['args']['pretrained_ckpt'], temporal_layers=tl).to(dev).eval()
    model.load_state_dict(ck['model'])
    for blk in model.temporal.blocks:
        blk.capture = True
    print(f"[load] ep={ck.get('epoch','?')}, L={args.L}, layers={tl}, views={args.views}")

    d = pickle.load(open(args.pkl, 'rb'))
    seqs = defaultdict(lambda: defaultdict(dict))
    for e in d:
        seqs[e['pose_id']][e['image_id']][e['camera_id']] = e

    # 收集所有窗口 + 中心帧/邻帧 conf, 选 good-center / bad-center
    cand = []
    for pid, fr in seqs.items():
        for w in windows_for_seq(fr, args.views, args.L):
            rays, confs, dts, gt = win_tensors(fr, w, args.views, args.L, ct)
            cf = confs[:, :, :, 0].numpy()                 # (J,V,L)
            c_center = cf[:, :, ct].mean()                 # 中心帧平均 conf
            c_neigh = np.delete(cf, ct, axis=2).mean()     # 邻帧平均 conf
            cand.append((pid, w, c_center, c_neigh, rays, confs, dts, gt))
    c_center_arr = np.array([c[2] for c in cand])
    gap = np.array([c[2] - c[3] for c in cand])            # 中心-邻帧 conf 差
    good_i = int(np.argmax(c_center_arr))                  # 中心帧最好
    bad_i = int(np.argmin(gap))                            # 中心比邻帧差最多(中心崩,邻帧好)
    print(f"[select] good-center: 中心conf={cand[good_i][2]:.2f} 邻帧={cand[good_i][3]:.2f}")
    print(f"[select] bad-center : 中心conf={cand[bad_i][2]:.2f} 邻帧={cand[bad_i][3]:.2f}")

    for tag, idx in [('good_center', good_i), ('bad_center', bad_i)]:
        _, w, cc, cn, rays, confs, dts, gt = cand[idx]
        r1 = rays[None].to(dev); c1 = confs[None].to(dev); d1 = dts[None].to(dev)
        J, V = 17, len(args.views)

        # 图2 Temporal: full forward → 末层 attn 中心行
        _ = model(r1, c1, d1, t_target=ct)
        tattn = model.temporal.blocks[-1].attn_cache       # (B*J, nh, L, L), B=1
        nh = tattn.shape[1]
        center_row = tattn[:, :, ct, :].reshape(J, nh, args.L).mean(0).cpu().numpy()  # (nh,L) 关节平均
        per_frame_conf = confs[:, :, :, 0].mean((0, 1)).cpu().numpy()                 # (L,) 每帧平均conf

        # 图1 VFT: 中心帧 token 过 vft_forward (带hook)
        store = []
        h = model.backbone.blocks_view_fusion[-1].attn.register_forward_hook(vft_attn_hook(store))
        token = model.encode(r1, c1)                       # (1,J,V,L,768)
        _ = model.vft_forward(token[:, :, :, ct, :])       # (1,J,V,768)
        h.remove()
        vattn = store[-1]                                  # (J, nh, V+1, V+1)
        fusion_to_view = vattn[:, :, 0, 1:].mean(1).cpu().numpy()   # (J,V) fusion→各视角, head平均
        center_view_conf = confs[:, :, ct, 0].mean(0).cpu().numpy() # (V,) 中心帧各视角conf

        # ---- 画 ----
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        im0 = axes[0].imshow(center_row, aspect='auto', cmap='viridis')
        axes[0].set_title(f'[{tag}] Temporal: 中心帧(t={ct})→各帧 attention\n每帧conf: ' +
                          ' '.join(f'{x:.2f}' for x in per_frame_conf))
        axes[0].set_xlabel(f'Key 帧 (0..{args.L-1}, t={ct}=中心)'); axes[0].set_ylabel('head')
        axes[0].axvline(ct, color='r', ls='--', lw=1); plt.colorbar(im0, ax=axes[0])
        im1 = axes[1].imshow(fusion_to_view, aspect='auto', cmap='viridis')
        axes[1].set_title(f'[{tag}] VFT: fusion→各视角 (中心帧)\n各视角conf: ' +
                          ' '.join(f'{x:.2f}' for x in center_view_conf))
        axes[1].set_xlabel(f'视角 {args.views}'); axes[1].set_ylabel('关节 (0..16)')
        plt.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        fp = os.path.join(OUT, f'attn_{tag}.png'); plt.savefig(fp, dpi=110); plt.close()
        print(f"[saved] {fp}")
        # 文字摘要(便于不看图也能判断)
        print(f"  {tag}: 中心帧attend自己={center_row[:,ct].mean():.3f}, "
              f"attend邻帧均值={np.delete(center_row,ct,1).mean():.3f}")


if __name__ == '__main__':
    main()
