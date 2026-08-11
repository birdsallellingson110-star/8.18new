# ST-VFT 项目：实验进度、关键决策与未来计划

> 更新: 2026-06-20
> 结果数字见: [training_results/RESULTS_SUMMARY.md](training_results/RESULTS_SUMMARY.md)

---

## 0. 项目目标

**研究命题**: "RUMPL + 时序融合"——在已复现的 RUMPL(多视角 2D→3D 提升)基础上，加入时序模块(ST-VFT)，验证时序信息能否在**少视角(V=2)**下进一步降低误差（少视角几何欠定，靠相邻帧时序补几何）。

- **对标论文**: RUMPL (arXiv 2512.15488)，3 个评估数据集: CMU Panoptic、Human3.6M、RICH。
- **核心思路**: 加载训练好的 RUMPL 权重 + 新增时序模块(TFT)微调，归因干净(同权重，唯一变量=时序)。

---

## 1. 实验历程与关键决策（时间线）

### Phase 0 — RUMPL Baseline 复现
- 用 MHP 生成 AMASS 合成数据(128109 单帧样本, 18 子集)，训练 `multiview_rumpl`。
- 修了 3 个关键 config bug(USE_T/RANDOM_NUM_VIEWS/APPLY_VIEW_FUSION)。
- **w/o conf 结果**: 单人 CMU V=2 KP* **43.4mm ≈ 论文 w/o conf 41.1mm**(忠实复现)。

### 关键转折 1 — conf 的发现
- 论文 Table 3 证明 **conf 贡献 ~6mm**(w/o conf 41.1 → 带 conf 35.0)。我们 Phase 0 的 43.4 对应的是论文 **w/o conf** 那一行(`clip_full` 默认没开 conf)。
- **带 conf 重训**(`clip_full_conf`, 单变量只开 conf): 控制变量审计逐字段通过 → CMU V=2 Abs 全17 **42.1→37.9(↓4.2mm)**, 方向与论文一致。

### 关键转折 2 — 论文主结果代码未开源
- 论文主结果 config `crf_4925` 用的 model 类 `multiview_pose_3d_fuser` **在开源仓库不存在**(只有 `multiview_rumpl`)，直接跑会 ImportError。
- **决策**: 不逐字节复刻论文(它含 5 处死配置/bug)，用 `multiview_rumpl` 实现论文方法。baseline = `clip_full + conf`(40.4mm KP*)，**放弃追 35.0**(残余 gap 来自未开源代码+数据子集+域差，已知系统性)，**进入时序**。

### 关键转折 3 — 路线 A（用户洞察）
- ST-VFT 不从头训，而是**加载 baseline 权重 + TFT 微调**。归因从"两个从头训的模型比"提升到"同一权重加不加 TFT 比"，干净得多。
- 实现 `STVFTPretrained`: 复用 baseline 的 VFT/PFT/编码权重 + 新增 TFT + **残差门控**(`vft_in = center + gate·tft_out`, gate 初始严格 0 → 初始 ≈ baseline)。

### 中等规模验证（160clip）— 路线 A 成立
- **前置门槛(float64 铁证)**: gate=0 逐元素还原 baseline(差 4.6e-14)→ 门控+权重+通路 100% 正确。
- **gate=0 初始 loss 33.6mm** = baseline 量级。
- **root cause 排查(教科书级)**: 初始 loss 202→132→33.6，逐步定位到 **04_fix_data_to_room 的 room 范围用错**(训练 config ROOM `[-2,0.3,-0.8,0.8]` vs official 渲染 room-size `[-0.5,-0.1,-0.2,0.2]`)，修复后 mean_abs 0.77→0.49 对齐 baseline。
- **gate 机制(ReZero 陷阱)**: gate=0 时 TFT 梯度被乘没(∂L/∂tft_out=gate·...=0)，gate 必须先长大 TFT 才解冻；gate.grad=1.7e-2 健康(时序有信号)，但 160clip 太少看不到增益 → **需全量定论**。

### 全量数据生成（进行中）
- **01 运动采样**: 加 `--motion-sampling`(优先 pose 帧间变化大的 clip，ST-VFT 学帧间变化，静止 clip 无用)，采出 286262 clip 池(18 子集，CMU 不在=防泄漏)。
- **哨兵批验 room**: 20clip 走 02→04，mean_abs=0.492 ✅。
- **worker 调优**: 8worker OOM 边缘(GPU0 EGL 集中 24189/24564)→ 拆 8chunk 降 6worker(19.8G 安全)；6worker 吞吐 ≈ 4worker(GPU0 是硬瓶颈)。
- **健壮性**: setsid(断联续跑) + chunk-size=10(中断丢<50min) + resume(补数据复用，不重头)。进程曾被 harness 连带 kill，chunk10+resume 救场(600clip 没丢)。

---

## 2. 当前进度（2026-06-20）

| 项 | 状态 |
|---|---|
| RUMPL baseline (w/o conf + 带 conf) | ✅ 完成, 见 RESULTS_SUMMARY |
| ST-VFT 代码(模块/dataloader/训练 loop/路线 A/门控) | ✅ 完成, 单测通过 |
| 中等规模验证(路线 A 成立 + root cause 修复) | ✅ 通过 |
| **全量 3000clip coco render** | 🔄 进行中(6worker, ~2 天, 当前 600/3000) |
| h36m 格式数据 | ⏳ 待转换(从 coco 离线转, ~1h, 不重 render) |
| 全量训练 | ⏳ 待 render 完 |
| CMU 时序评估(B vs C) | ⏳ 待准备(连续帧+Δt 匹配) |

---

## 3. 未来计划

### 近期(数据生成 → 训练)
1. **coco render 完**(~2 天) → 04 修复 room(`[-0.5,-0.1,-0.2,0.2]`) → `Counter(sources)` 验 18 子集分布。
2. **h36m 转换**(~1h, 离线): 从 coco 数据的 SMPL+H 参数重算 h36m 3D(J_regressor_h36m) + coco 2D 转 h36m 2D(mmpose2h36m)。**不重 render**——h36m/coco 只差关键点定义，render 图和 HRNet 检测关键点无关。
3. **全量训练 ST-VFT**(路线 A): gate lr=1e-3, TFT lr=3e-5(不动), epoch 30-50。**一次只动一个旋钮**。
   - 成功判据: 训练 loss < 33.6 + gate 长起来。

### 评估(金标准)
4. **CMU 时序评估准备**: CMU 测试集现为 skip64 单帧, ST-VFT 需连续 L=5 帧 + Δt 尺度匹配(CMU 30fps vs 训练 120fps)。
5. **B vs C 对照**(论文核心数字):
   - A = 原始单帧 baseline(40.4, 论文水平锚点)
   - B = 完整模型但 TFT 旁路(gate=0, 隔离"窗口输入"影响)
   - C = 完整 ST-VFT
   - **时序净增益 = B − C**(不是 A−C，避免混入"多喂几帧"的影响)
6. **微调进阶**: (a) 冻结 backbone 只训 TFT(纯时序归因) → (b) 全微调(TFT 大 lr / VFT-PFT 小 lr，差 10 倍)。两个都报。

### 三数据集(论文对标)
7. **CMU + RICH**(都是 coco): 用 coco 数据训练。
8. **Human3.6M**(h36m): 用转换的 h36m 数据训练(单独模型，RUMPL 是 per-dataset 训练范式)。

---

## 4. 预期

- **核心假设**: ST-VFT 在 V=2(少视角)下，靠时序把 MPJPE 压到 baseline(40.4mm KP*)以下。
- **最强论点**: 若 ST-VFT 用**更浅的层 + 时序**就超过更深的纯空间 RUMPL，证明"时序 > 单纯堆空间深度"——呼应研究动机(少视角几何欠定，靠时间补比靠更深网络补更对路)。
- **诚实预期**: 训练 loss 降是早期信号(必要非充分)；最终结论看 CMU B vs C。160clip 看不到信号是数据量问题，全量(10000 池采 3000)接近 baseline 数据量级，能下定论。
- **论文叙事**: "我们忠实复现 RUMPL(w/o conf 对齐 41.1)，开源缺主结果模块故 conf 版达 40.4mm，以此为 baseline，时序增益 = 相对它的提升"——诚实且可归因，比硬凑 35.0 更可信。

---

## 5. 关键文件索引

```
EXPERIMENT_PROGRESS_AND_PLAN.md       本文件
training_results/RESULTS_SUMMARY.md   结果汇总(精度对比)
training_results/training_summary.md      w/o conf 训练
training_results/training_summary_conf.md 带 conf + 控制变量审计 + CMU V=2
training_results/cmu_eval_summary.md      单人 CMU 评估(w/o conf V=2/V=5)

RUMPL/lib/models/stvft/                ST-VFT 模块(tft/vft/pft/ray_embed/delta_t_encoder/stvft_pretrained)
RUMPL/lib/dataset/stvft_dataset.py     clip 级 dataloader(ray 复刻 RUMPL + 窗口 + Δt + collate)
RUMPL/run/train_stvft.py               ST-VFT 训练(per-t supervision + MPJVE + gate)
RUMPL/configs/.../clip_full_conf.yaml  带 conf baseline config
MHP/01_clip_create_dataset.py          clip 采样(--motion-sampling)
MHP/02_clip_run.py                     clip render + HRNet
MHP/04_fix_data_to_room.py             room placement(注意 --room-min/max 用 official 范围!)
```
