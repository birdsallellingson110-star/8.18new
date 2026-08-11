# 官方代码对照笔记（2026-07-25）

路径：`/mnt/data/cjydata/reference_code/`

## 已克隆

| 仓库 | 状态 | 用途 |
|---|---|---|
| KTPFormer | OK | KPA SemGCN |
| MHFormer | OK | 多假设 |
| MotionBERT | OK | 2D 噪声预训练 |
| D3DP | OK | JPMA 关节级选择 |
| EFMK | OK | 骨架先验 / 重投影聚合 |
| PoseFormerV2 | OK | 频域抗噪 |
| PCT | OK | VQ 姿态 codebook（UniCodebook 最接近参考） |
| FusionFormer | **空壳**（仅 README） | 不可用 |
| UniCodebook | **无公开代码** | 只能跟论文 + PCT |
| UPose3D | 无代码 | — |

旧有：AdaFuse / LT / PlaneSweep / MvP / MVGFormer

---

## 关键纠正（相对我们草稿）

### 1. KTPFormer-KPA ≠ softmax 图消息

官方：`LearnableGraphConv` = 固定骨架 `adj`（row-norm + diag=1）+ 可学习残差 `adj2@1e-6` + 双路 W（self/neighbor）+ 逐关节 M。
**禁止**对 adj 做 softmax。
插入点：PFT **之前一次** frontend，不是每层；单帧跳过 TPA。

### 2. UniCodebook / PCT：软 DCSA  alone 不够

PCT 流程：Stage-I **硬 VQ + EMA codebook（buffer，无 grad）+ commitment×15** → **冻结** → Stage-II CE 逼离散。
我们当前 `PoseCodebookDCSA` 更像 continuous memory bank（软注意力 + Parameter codebook）。
要成功：warmup 硬 VQ → 冻 codebook → 加 CE/ST 离散压力，再软注入。

### 3. MHFormer：H=3 硬编码，最终只对合并结果算 loss

不是 3 个独立 3D head。正确移植：VFT 放 **H 个 fusion token** → Conv1d/Linear 压成 1 → 再进 PFT；loss 只看最终 MPJPE。

### 4. D3DP JPMA：测时关节级重投影选假设

可与 per-view ray-depth 假设拼接；不必上整套 diffusion。

### 5. MotionBERT：训时噪声+mask+压 conf

是 **训练配方**，不是换主干；对齐 A2/struct_occ 故事。

---

## 修订后的实验优先级

1. **KPA-faithful**：照抄 SemGCN KPA，挂 A2  
2. **MH3 fusion tokens**：H=3，挂 A2  
3. **D3-PCT**：两阶段 codebook（冻 + CE），挂 A2  
4. Conf-FiLM：仍可作廉价对照，但优先级低于上面三个

---

## 2026-07-25 并行实验已开

实现（env）：
- `RUMPL_KPA=1`：FaithfulKPA（SemGCN，无 softmax）
- `RUMPL_MULTI_HYP=3`：VFT 三 fusion token + Conv1d 合并
- `RUMPL_POSE_CODEBOOK=1`：PCT buffer codebook + EMA freeze@8000 + CE + soft inject

在跑：
- GPU0：`KPA_a2_seed0_20260725`（A2）
- GPU1：`MH3_a2_seed0_20260725`（A2）
- 排队：两者 END 后自动开 `D3PCT_a2_seed0_20260725`，各带 `chain_module_waitlog` 做 V2–V5×occ 评测

日志：`/mnt/data/cjyoutput/baseline_reaudit_20260722/{KPA,MH3,D3PCT}_*.log`
