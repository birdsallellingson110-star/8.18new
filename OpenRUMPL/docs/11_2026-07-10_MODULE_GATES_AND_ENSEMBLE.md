# 2026-07-10 模块 gate 与最终可用结果

## 1. 已验证主结果

CMU pose5/6, V=2 全10相机配置, abs KP* mm, 相对 baseline 的 Δ(负数=改善)。

- general seed1: 配置平均 -2.42mm, 最差配置 -0.26mm, 10/10 全降。继续作为单模型主结果。
- lw0.7 3-seed mean: 配置平均 -1.85mm, 最差配置 -0.12mm, 10/10 全降。作为 leg-weight 鲁棒性补充。
- prediction ensemble(g1 + lw0.7 seeds 0/1/2): 配置平均 -2.64mm, 最差配置 -0.47mm, 10/10 全降。当前最均衡的多模型结果。
- prediction ensemble(g1 + lw0.7 seed1): 配置平均 -2.66mm, 最差配置 -0.24mm, 10/10 全降。平均略强, 但最差配置不如 g1+lw3。

## 2. 否掉的模块/方向

### DeProPose 方案A: pairwise ray residual

V=2 下两条射线的 pairwise 最近距离天然对称, 两个视角得到相同几何残差, 归一化后权重≈1。实测 DEPRO_LAMBDA=0.5 first-batch max diff 仅 9.5e-7, 对 V=2 没有效信号。

### DeProPose 方案B: 2-pass 点到射线残差

零训练 sweep 结果:

```
λ0.25 平均 +0.15mm, 最差 +0.56mm, 改善 3/10
λ0.5  平均 +0.84mm, 最差 +1.89mm, 改善 1/10
λ1.0  平均 +3.52mm, 最差 +5.91mm, 改善 0/10
```

结论: 当前 RUMPL/CMU 射线残差作为可靠度会放大 2D 噪声, 不适合作为主模块。

### S2DHand-style test-time self-supervision

严格无GT信号检查:

- ray refine 仅贴回两条2D射线: seed1 [3,6] 从 39.00mm 退到 71.63mm。
- 单视角 prediction consensus/平均: 比 V2 融合平均退 +73mm 量级, 0/10 改善。

结论: RUMPL 的单视角输出和 CMU 2D ray 自监督信号不足以做可靠 TTA; 不能照搬 S2DHand。

### Weight soup

权重平均不在同一 basin, 直接炸:

```
gen5soup 平均 +748.54mm
lw7soup 平均 +679.82mm
g1+lw7s1 soup 平均 +7.18mm
```

结论: 不做 checkpoint weight soup。

## 3. 当前建议

论文/汇报层次:

1. 单模型主结果: general seed1, -2.42mm, 10/10全降。
2. 稳健补充: lw0.7 3-seed mean, -1.85mm, 10/10全降。说明 leg-weight 能把多seed均值也推到全配置改善。
3. 多模型上界/部署可选: prediction ensemble(g1+lw0.7三seed), -2.64mm, 最差 -0.47mm, 10/10全降。

下一步如果还要继续创新模块, 不建议再从纯几何/TTA入手。更靠谱的是围绕已有成功机制做轻量化: teacher/student 蒸馏的配置条件化、seed1 结果复现性增强、或者训练时显式优化 worst-config risk。
