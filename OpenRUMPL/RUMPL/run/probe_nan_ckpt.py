"""从临界态 ckpt 快速触发 nan 梯度 + anomaly 定位算子(不训练, 几个 backward 就触发)。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import torch
from torch.utils.data import DataLoader
from dataset.stvft_dataset import STVFTClipDataset, make_collate_random_views
from models.stvft.stvft_pretrained import STVFTPretrained

torch.autograd.set_detect_anomaly(True)
CFG = "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"
RUMPL = "/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar"
STVFT = "/mnt/data/cjyoutput/stvft/full_coldstart_diag/stvft_final.pth"
GLOB = "/mnt/data/cjyoutput/mhp_workspace_unused"  # placeholder
GLOB = "/mnt/data/cjydata/mhp_workspace/clip_full/stage_V_room/train/*.pkl"
dev = 'cuda'


def mpjpe(p, g):
    return (((p - g) ** 2).sum(-1) + 1e-8).sqrt().mean()


def per_t(model, rays, confs, dts, L):
    preds = []
    for t in range(L):
        nd = dts - dts[:, :, t:t + 1]
        preds.append(model(rays, confs, nd, t_target=t))
    return torch.stack(preds, 1)


def main():
    model = STVFTPretrained(CFG, RUMPL, freeze_backbone=True, gate_init=0.1).to(dev)
    sd = torch.load(STVFT, map_location='cpu')
    sd = sd['model'] if 'model' in sd else sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.train()
    print(f"[load] gate={model.gate.item():.4f} missing={len(missing)} unexpected={len(unexpected)}")

    ds = STVFTClipDataset(GLOB, L_window=5)
    dl = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=make_collate_random_views(2, 5, seed=1))
    for it, b in enumerate(dl):
        rays = b['rays'].to(dev); confs = b['confs'].to(dev); dts = b['delta_ts'].to(dev); gt = b['gt_3d'].to(dev)
        preds = per_t(model, rays, confs, dts, 5)
        pos = mpjpe(preds, gt)
        vel = mpjpe(preds[:, 1:] - preds[:, :-1], gt[:, 1:] - gt[:, :-1])
        loss = pos + 0.1 * vel
        model.zero_grad()
        loss.backward()   # anomaly 会在产 nan 的 backward 算子处抛 RuntimeError
        bad = [nm for nm, p in model.named_parameters()
               if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()]
        print(f"it{it} loss={loss.item():.4f} pos={pos.item()*1000:.1f}mm bad_grads={len(bad)}")
        if bad:
            print("FIRST BAD GRADS:", bad[:5])
            break
        if it >= 80:
            print("80 个 batch 无 nan — 临界态不够近, 需换 ckpt")
            break


if __name__ == '__main__':
    main()
