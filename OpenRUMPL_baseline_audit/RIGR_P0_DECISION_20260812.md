# RIGR P0 结果与决策（2026-08-12）

## 协议

- 数据：真实 H36M S9/S11，验证缓存 `H76_validation_all_subsets.npz`。
- 评估：absolute MPJPE，action-equal All-17，V2/V3/V4 全相机组合。
- 当前 H76：V2 **34.816 mm**，V3 **30.489 mm**，V4 **29.691 mm**。
- 图像证据：已有冻结 HRNet-W32 dense heatmap，按 MMPose 的 crop 变换映射回原图。
- P0 只做上限诊断，不把 GT 送入推理；2D oracle 在每个局部窗口中选择最接近标注的候选，再用标定射线三角化。

实现文件：`diagnose_rigr_heatmap_oracle_20260812.py`。

## 全量结果

top-4 局部模式、半径 16 heatmap pixel 的全量结果：

| 方法 | V2 | V3 | V4 |
|---|---:|---:|---:|
| H76 baseline | 34.816 | 30.489 | 29.691 |
| HRNet local 2D oracle | 99.095 | 67.034 | 63.074 |
| 局部候选平均像素误差 | 11.963 px | 11.963 px | 11.964 px |

P0 输出：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/RIGR_P0_20260812/full_r16_top4_2d.json`

半径 2/4/8 的小样本结果与半径 16 同趋势；半径 16、top-8 的 8 组 smoke
也没有显示超过 H76 的趋势（仅作为实现检查，不进入主表）。

## 闭环检查

1. pkl 2D 标注按 RUMPL lower-body swap 后，与相机投影的 3D target 平均约
   1.15 px，证明相机、单位、关节顺序和投影公式一致。
2. 直接用标注 2D 做射线三角化，在第一组 V2 的误差约 5.56 mm，说明射线求解器
   本身没有造成几十毫米误差。
3. 批量射线求解与原逐关节求解在 smoke 上逐项一致；加速版本只改变计算组织。

## 决策

P0 **不满足进入 heatmap-only P1 的门槛**：局部 HRNet 末端峰的 oracle 远差于
H76，不能据此训练一个“局部峰修正头”并声称会提升精度。原因不是几何管线错误，
而是冻结 2D detector 的终端 heatmap 峰包含约 12 px 的跨视角/关节误差，H76 已经
通过 ray/VFT/PFT 学到了比直接峰修正更强的统计先验。

因此停止以下路线：只用 top-k 峰、局部 heatmap scalar 或其简单 Transformer
替换 PFT。下一步转到 **P2 feature-level geometry loop**：按 MVGFormer 的
“3D query → 相机投影 → feature sampling → cross-view update → triangulation”
实现，读取 HRNet 中间 feature map（而不是只读取 17 个末端 heatmap 通道）。首版
仍冻结 HRNet，只训练跨视角采样/更新和几何求解；若没有至少 0.5 mm 的稳定改善，
停止 RIGR 主线并回到已验证的 H76+E2 组合。

