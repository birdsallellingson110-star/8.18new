# E2-V234 与 HRNet 输入协议实现记录（2026-08-12）

## 本轮目标

把只覆盖 V3/V4 的 E2 扩展为同一套候选效用模型处理 V2/V3/V4，并在开始更大
的图像特征结构前固定 HRNet 输入协议。新增代码不覆盖旧实验目录。

## 已实现文件

- `build_e2_v234_candidate_cache_20260812.py`
  - 读取已完成的 RIGR→E2 22 候选缓存；
  - 追加每个 2/3/4-view 子集的 confidence-weighted DLT 与 IRLS 候选；
  - 输出 44 候选缓存和候选顺序 manifest；
  - 候选生成只读取射线，GT 仅随缓存保留供后续监督，不参与候选生成。
- `diagnose_e2_v234_candidate_pool_20260812.py`
  - 零训练 oracle/候选互补性诊断。
- `train_e2_v234_universal_20260812.py`
  - 统一 V2/V3/V4 task；
  - 内部 modulo-10 holdout；
  - checkpoint 选择指标为 V2/V3/V4 action-equal soft MPJPE 均值；
  - 保留 E2 的 direct rank + GHT expected-risk 两阶段损失。
- `audit_hrnet_input_protocol_20260812.py`
  - 记录 detector config/checkpoint SHA、box 来源、MMPose input metadata；
  - 对照当前 token 坐标公式与 MMPose MSRA codec；
  - 只读，不修改任何数据。
- `launch_e2_v234_universal_20260812.sh`
  - GPU0/seed0、GPU1/seed1 并行启动。

## Smoke 与诊断

### HRNet 坐标

当前 HRNet 配置为非 UDP MSRA codec，输入尺寸 `288×384`，heatmap/stage-4
高分辨率尺寸 `72×96`，flip-test 开启。早期审计脚本把 H36M 源文件的归一化
`scale≈1.956` 错当作 MMPose 像素 scale，造成几千像素的假报警，已经修正。
使用 MMPose metadata 后，当前公式

```text
(xy - center + 0.5*scale) / scale * [72, 96]
```

与官方 MSRA affine+codec 坐标最大差异约 `1e-5` feature pixel。现有图像特征
缓存无需因为该审计重导出。闭区间 `align_corners` 差异是采样 convention 诊断，
不应与 MSRA codec 坐标混用。

审计报告：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/HRNet_Input_Audit/report.json`

### 候选池

输入：固定 RIGR→E2 22 候选。

| 阶段 | 现有22候选 oracle | 加 confidence | 加 IRLS | 加两者 |
|---|---:|---:|---:|---:|
| V2 | 29.142 | 29.128 | 29.124 | 29.124 |
| V3 | 20.939 | 20.890 | 20.887 | 20.887 |
| V4 | 17.830 | 17.764 | 17.762 | 17.762 |

新增候选互补性较小但非零，足以进行受控 E2 训练；若训练不提升，应归因于
utility scorer 学习能力，而不是候选完全重复。

诊断报告：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/E2_V234_CandidateDiagnostic/result.md`

### V234 smoke

已使用 1 个训练 batch、1 个 validation batch、attention depth=1 跑通：

- V2/V3/V4 所有 task mask 正确；
- baseline、hard、soft、oracle 四种输出正常；
- 没有 test GT 参与训练或 checkpoint 选择；
- 44 候选组合长度不一致只保存在 manifest，不强行写入规则 ndarray。

## 正式运行

目录：
`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/E2_V234_Universal_20260812/`

- `seed0`：GPU0；
- `seed1`：GPU1；
- attention depth=2；
- 10 epoch direct + 5 epoch GHT；
- batch size=256；
- 训练数据为现有 20k balanced RIGR→E2 cache；
- 最终 S9/S11 只在 holdout 最优 checkpoint 上评估一次。

## 下一步判定

1. 若 V2 相对 RIGR `32.880 mm` 至少下降 `0.3 mm`，且 V3/V4 不回退超过
   `0.1 mm`，保留 V234 universal 主线；
2. 若 V2 无提升但 V3/V4 保持，则 E2 仍可作为 V3/V4 候选融合消融，不声称
   任意视角数统一；
3. 若 universal 破坏 V3/V4，则保留 V3/V4 specialist，并为 V2 单独增加更
   有互补性的候选（不重复训练普通 residual）；
4. 只有通过该门槛，才继续 MVGFormer-style 2D offset/confidence→可微三角化
   迭代，或 full-heatmap epipolar fusion。
