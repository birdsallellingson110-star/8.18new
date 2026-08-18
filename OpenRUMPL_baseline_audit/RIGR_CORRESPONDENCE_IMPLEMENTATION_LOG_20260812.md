# RIGR HRNet correspondence implementation and E2 cascade log

日期：2026-08-12  
目的：在保持冻结 HRNet 2D 检测输入、相机射线和 RUMPL/H76 几何管线不变的前提下，验证论文启发的跨视角特征对应是否能补足 RUMPL 的视角融合瓶颈，并严格区分 correspondence、几何偏置和 E2 候选评分的作用。

## 1. 口径与基线

- 数据：正式 Human3.6M S9/S11 验证/测试管线。
- 输入：冻结的 HRNet 中间特征、2D 关键点/置信度、已标定相机射线；没有使用 GT 2D。
- 评估：absolute MPJPE，action-equal All-17；V2/V3/V4 表示视角数。
- 参考基线：H76/RUMPL 真实 H36M 基线 V2/V3/V4 = 34.816/30.489/29.691 mm。
- 原有 RIGR HRNet feature + geometry bias = V2/V3/V4 32.880/29.269/28.387 mm（seed0）。
- 原有 RIGR→E2 专项重训（固定旧 RIGR 候选，E2 只针对 V3/V4）均值：V3 28.5553，V4 27.8078 mm。

## 2. 实现内容

文件：

- `train_rigr_hrnet_feature_20260812.py`
- `evaluate_rigr_e2_cascade_20260812.py`
- `export_rigr_refined_e2_candidates_20260812.py`
- `train_e2_v234_universal_20260812.py`

新增 `--correspondence-attention`：

1. 保留 HRNet 5×5 patch，而不是立即把每个视角压成一个 token；
2. 当前视角的中心特征作为 query；
3. 所有有效视角的 patch token 作为 key/value；
4. padding 视角在 key 侧屏蔽；
5. correspondence residual 用零初始化 gate，epoch 0 与原 RIGR 特征流严格一致；
6. 可选 geometry-biased attention 在视角 token attention logit 处加入射线夹角、射线距离和置信度 pair bias，同样零初始化。

该设计是受 Epipolar Transformer 的跨视角对应、MVGFormer 的几何/特征交替细化思路启发的轻量实现，不声称是论文官方代码的逐行复现。它仍然只使用稀疏 HRNet patch 和相机射线，未引入图像级 epipolar line sampling。

## 3. 实现正确性检查

- synthetic forward smoke：V2/V3/V4 形状正确；零初始化时 `(output - prediction).abs().max() = 0`。
- 16 train/16 validation smoke：成功完成前向、反向和 checkpoint 保存。
- HRNet 输入审计已确认当前 crop/resize 公式与官方非 UDP MSRA codec 的 MMPose 像素坐标口径一致，未发现输入缩放错误。
- E2 导出器补充读取 `correspondence_attention` checkpoint 标志，避免导出时静默退回旧 RIGR。

## 4. correspondence 单模块 2×2 种子对照

训练协议：balanced20k group list，8 epochs；同一 RIGR 训练/验证管线。

| 模型 | seed | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| correspondence，无几何偏置 | 0 | 33.3679 | 29.7429 | 28.9007 |
| correspondence，无几何偏置 | 1 | 33.2969 | 29.8325 | 29.0249 |
| correspondence + geometry bias | 0 | 33.4217 | 29.8037 | 28.9645 |
| correspondence + geometry bias | 1 | 32.6996 | 29.1525 | 28.2486 |

单模块结果说明 correspondence 相比原 H76/RIGR 输入流有稳定但有限的收益；几何偏置的单模块结果存在 seed 交互，不能用第一组不同 seed 的结果直接宣称偏置收益。

## 5. 旧 E2 直接套新候选：失败但有诊断价值

把 correspondence 候选直接替换到旧 E2 checkpoint，而不重训 E2：

- 无偏置 correspondence：V3/V4 = 29.006/28.286 mm；
- 几何偏置 correspondence：V3/V4 = 29.018/28.282 mm；
- 旧 RIGR→E2：V3/V4 = 28.555/27.808 mm。

这证明候选分布发生变化后，旧 E2 scorer 存在明显 distribution mismatch；该结果不能被解读为 correspondence 无效。因此后续对新候选重新训练 E2。

## 6. correspondence→E2 专项级联：严格 2×2 配对

E2 训练协议保持原专项 E2：22 候选、10 direct + 5 GHT epochs、同一 holdout 和温度。

| correspondence | E2 seed0 | E2 seed1 | 均值 |
|---|---:|---:|---:|
| 无偏置，V3 | 28.8790 | 28.8776 | **28.8783** |
| 几何偏置，V3 | 28.4160 | 28.4135 | **28.4148** |
| 无偏置，V4 | 28.1370 | 28.1561 | **28.1465** |
| 几何偏置，V4 | 27.6405 | 27.6545 | **27.6475** |

配对增益（同一 E2 seed 的 bias - no-bias）：

- V3：0.4630 / 0.4641 mm，均值 **0.4635 mm**；
- V4：0.4965 / 0.5016 mm，均值 **0.4990 mm**。

这组结果支持“geometry bias 在 correspondence 后提供稳定约 0.46–0.50 mm 级联收益”，随机种子差异约 0.01 mm，不能解释该收益。

## 7. 通用 V234 E2

为避免论文中只报告 V3/V4 被质疑，使用同一 44 候选池、同一训练器同时训练 V2/V3/V4：

| 模型 | V2 | V3 | V4 |
|---|---:|---:|---:|
| correspondence，无偏置 + V234 E2 | 33.3191 | 29.1226 | 28.4227 |
| correspondence + geometry bias + V234 E2 | **32.7769** | **28.6621** | **27.9243** |

补充反向 seed 后的 V234 均值（seed0/seed1）：

- 无偏置：V2 **33.3333±0.0201**，V3 **29.1050±0.0249**，V4 **28.4034±0.0273**；
- 几何偏置：V2 **32.7701±0.0096**，V3 **28.6679±0.0083**，V4 **27.9273±0.0042**。

同 seed 配对增益为 V2 **0.5632**、V3 **0.4371**、V4 **0.4761** mm，四个结果方向一致。

V234 共享训练不会损坏 V2（仍为 32.78 mm），但会稀释 V3/V4 专项 E2 的性能。因此：

- 需要一个统一模型时，报告 V234 版本；
- 追求各视角阶段最低 MPJPE 时，使用 V3/V4 专项 E2，并明确这是 stage-specific scorer；
- 不能把专项 V3/V4 结果伪装成一个同时训练 V2/V3/V4 的单模型结果。

## 8. stage-specific output heads 对照

只将 E2 最后一层 utility calibration 拆成 V2/V3/V4 三个头，共享其余编码器和候选池：

| 模型 | V2 | V3 | V4 |
|---|---:|---:|---:|
| 无偏置 + stage heads | 33.3237 | 29.1466 | 28.5019 |
| 几何偏置 + stage heads | 32.7829 | 28.6617 | 27.9296 |

与共享头相比没有实质改善，故不作为主线创新，不再继续堆叠校准头。

## 9. 当前决策

主线暂定为：

> 冻结 HRNet 2D 输入 → HRNet patch-level cross-view correspondence → ray-geometry biased view attention → RUMPL/H76 3D residual →（V3/V4）E2 multi-hypothesis utility scorer。

论文表格必须同时给出：

1. RUMPL/H76 baseline；
2. correspondence only；
3. correspondence + geometry bias；
4. correspondence + geometry bias + E2（V3/V4 专项）；
5. correspondence + geometry bias + V234 E2（统一模型）；
6. seed mean/std、参数量和运行时间。

不要再做无依据的大模块堆叠；下一步只值得做：

- 对主线偏置 correspondence checkpoint 做第三个 seed 或完整训练数据复核；
- 输出 Negative View Rate、2→3→4 单调性和未见视角组合泛化；
- 若需要进一步降 MPJPE，优先调 E2 候选池/训练校准，而不是继续加更多 Transformer 层；
- 时序放在单帧主线稳定后，作为单独 T=9 消融，不与当前 correspondence 结果混报。

## 10. 输出位置

- correspondence：`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_20260812/`
- correspondence E2 专项：`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Retrain_20260812/`
- correspondence V234 缓存：`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Export_20260812/` 和 `RIGR_Correspondence_V234_Cache_20260812/`
- correspondence V234 共享头：`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_V234_Retrain_20260812/`
- correspondence V234 stage heads：`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_V234_StageHeads_20260812/`
- 旧 E2 直接套新候选诊断：`/mnt/data/cjyoutput/open_source_fusion_audit_20260812/RIGR_Correspondence_E2_Diagnostic_20260812/`
