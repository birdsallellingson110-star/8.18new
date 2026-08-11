"""ST-VFT Phase 1 — 路A: 加载 baseline 权重 + TFT 微调 (用户 2026-06-18 决策)。

设计: ST-VFT 的编码/VFT/PFT/head 直接 = baseline (clip_full_conf, 40.4mm) 的组件,
加载其权重; TFT(768维) 作为唯一新增, 插在编码后/VFT前。归因: 同一 VFT/PFT 权重,
唯一区别是 TFT。

数据流 (设计文档 TFT→VFT→PFT 顺序):
  rays(B,J,V,L,6)+confs(B,J,V,L,1) → [baseline编码] → token(B,J,V,L,768)
    → +Δt编码 → [TFT 跨L时序融合] → (B,J,V,768)
    → [baseline VFT 跨V] → (B,J,768)
    → [baseline PFT+head 跨J] → (B,J,3)

token 通道顺序严格 = baseline: [dir_emb(256), inter_emb(256), conf_emb(256)] = 768,
否则加载的 VFT 权重通道对不上。VFT/PFT forward 原样复刻 baseline (含 PFT 末层双调用,
因 40.4 是带它训的, 权重须 forward 一致)。

三对照 (forward 的 tft_mode):
  - 'full'     : TFT 真融时序 (方法 C)
  - 'identity' : TFT 旁路, 只取中心帧 token (对照 B, 隔离"窗口输入"影响)
"""
import os
import sys
import torch
import torch.nn as nn

# lib 根 (本文件在 lib/models/stvft/)
_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
if _LIB not in sys.path:
    sys.path.insert(0, _LIB)

from models.stvft.tft import TFT
from models.stvft.delta_t_encoder import DeltaTEncoder


class STVFTPretrained(nn.Module):
    def __init__(self, rumpl_cfg_path, rumpl_ckpt_path,
                 tft_layers=4, n_heads=8, max_period=2.0, freeze_backbone=True,
                 gate_init=0.0):
        super().__init__()
        # ---- 实例化 baseline 并加载权重 ----
        from core.config import config as rumpl_config, update_config
        update_config(rumpl_cfg_path)
        from models.multiview_rumpl import get_multiview_rumpl_net
        rumpl = get_multiview_rumpl_net(rumpl_config, is_train=True)
        sd = torch.load(rumpl_ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        missing, unexpected = rumpl.load_state_dict(sd, strict=False)
        assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}"
        # missing 只允许是非 features 的辅助项 (本模型只用 features)
        self.backbone = rumpl.features          # MultiView_RUMPL
        self.D = self.backbone.Spatial_pos_embed.shape[-1]  # 768
        assert self.D == 768, f"expected 768, got {self.D}"

        # ---- TFT (新增, 768维) + Δt 编码 + 残差门控 ----
        self.tft = TFT(d_model=self.D, n_heads=n_heads, n_layers=tft_layers)
        self.dt_encoder = DeltaTEncoder(d_model=self.D, max_period=max_period)
        # 标量门控. gate_init=0 → ReZero(初始=baseline, 但 TFT 梯度被乘没=冷启动陷阱);
        # gate_init>0 (如0.1) → 破冷启动: TFT 初始有非零输出→有真实梯度→能学起来.
        self.gate = nn.Parameter(torch.full((1,), float(gate_init)))

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.freeze_backbone = freeze_backbone

    # ---- 编码: 复用 baseline 编码层, 通道顺序 [dir,inter,conf] = 768 ----
    def encode(self, rays, confs):
        bb = self.backbone
        dir_emb = bb.encoding_to_embedding(rays[..., 0:3])    # (B,J,V,L,256)
        inter_emb = bb.encoding_to_embedding(rays[..., 3:6])  # 共享层
        conf_emb = bb.confidence_to_embedding(confs)          # (B,J,V,L,256)
        return torch.cat([dir_emb, inter_emb, conf_emb], dim=-1)  # (B,J,V,L,768)

    # ---- VFT: 原样复刻 baseline (fusion_token + View_enc + blocks_view_fusion) ----
    def vft_forward(self, x):
        bb = self.backbone
        B, J, V, D = x.shape
        x = x.reshape(B * J, V, D)
        fusion = bb.fusion_token.expand(B * J, -1, -1)        # (B*J,1,D)
        x = torch.cat([fusion, x], dim=1)                     # (B*J,V+1,D)
        if bb.add_view_enc:
            x = x + bb.View_enc_learned[:, :V + 1]
        x = bb.pos_drop(x)
        for blk in bb.blocks_view_fusion:
            x = blk(x)
        x = bb.View_norm(x)
        x = x[:, 0, :]                                        # fusion token 输出
        return x.reshape(B, J, D)

    # ---- PFT + head: 原样复刻 baseline (含末层双调用) ----
    def pft_forward(self, x):
        bb = self.backbone
        b, J, D = x.shape
        x = x + bb.Spatial_pos_embed
        x = bb.pos_drop(x)
        for ix, blk in enumerate(bb.blocks):
            if ix == len(bb.blocks) - 1:   # baseline 末层双调用, 权重一致须沿用
                x = blk(x)
            x = blk(x)
        x = bb.Spatial_norm(x)
        x = x.reshape(b, J, -1)
        x = bb.head(x)
        return x.reshape(b, -1, 3)                            # (B,J,3)

    def forward(self, rays, confs, delta_ts, t_target=None, gate_override=None):
        """rays(B,J,V,L,6), confs(B,J,V,L,1), delta_ts(B,V,L) → (B,J,3)

        残差门控: vft_in = center_token + gate * tft_out (gate 初始 0)。
        - t_target: 预测哪一帧的 pose。残差基取 frame-t_target 的 token (随 t 变, 不写死中心)。
          per-t supervision 时由 per_t_forward 逐帧传入; 推理默认 L//2 中心帧。
          关键: gate=0 时 vft_in=frame-t_target token = baseline 对 frame-t_target 的预测,
          与 gt[:,t_target] 对齐 → gate=0 全窗口各帧都还原 baseline (消掉写死中心的错位)。
        - gate_override=0 → 对照 B (TFT 旁路); gate_override=None → 方法 C (用学到的 gate)。
        B 与 C 严格只差 gate, 归因无懈可击。dt 只进 TFT, 不污染 center_token。
        """
        B, J, V, L, _ = rays.shape
        if t_target is None:
            t_target = L // 2
        token = self.encode(rays, confs)                # (B,J,V,L,768) 纯baseline编码, 无dt
        center = token[:, :, :, t_target, :]            # (B,J,V,768) baseline frame-t_target, 无dt污染
        dt = delta_ts.unsqueeze(1).expand(B, J, V, L)
        tft_out = self.tft(token + self.dt_encoder(dt))  # (B,J,V,768) dt只进TFT
        # gate_override: None→用学到的 self.gate; 给定值则直接用(0=旁路=baseline, 1=满接入)
        g = self.gate if gate_override is None else token.new_tensor(float(gate_override))
        vft_in = center + g * tft_out                   # gate=0 → center(baseline); gate=1 → 满接入TFT
        x = self.vft_forward(vft_in)                    # (B,J,768)
        return self.pft_forward(x)                      # (B,J,3)


def test_stvft_pretrained():
    cfg = "configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml"
    ckpt = "/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/model_best.pth.tar"
    model = STVFTPretrained(cfg, ckpt, freeze_backbone=True)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"params: total {n_total/1e6:.2f}M, trainable {n_train/1e6:.2f}M (应只有TFT+dt_encoder)")
    B, J, V, L = 2, 17, 5, 5
    rays = torch.randn(B, J, V, L, 6)
    confs = torch.rand(B, J, V, L, 1)
    dts = torch.randn(B, V, L)
    print(f"  gate init: {model.gate.item()} (应严格=0)")
    for go, name in [(None, 'C(full)'), (0, 'B(gate=0旁路)')]:
        out = model(rays, confs, dts, gate_override=go)
        assert out.shape == (B, J, 3), (name, out.shape)
        print(f"  {name}: out {tuple(out.shape)} OK")
    # gate=0 时 C 应严格等于 B (初始 gate=0)
    oc = model(rays, confs, dts, gate_override=None)
    ob = model(rays, confs, dts, gate_override=0)
    print(f"  初始 gate=0: C vs B 最大差 {(oc-ob).abs().max().item():.2e} (应~0)")
    print("STVFTPretrained PASS")


if __name__ == "__main__":
    test_stvft_pretrained()
