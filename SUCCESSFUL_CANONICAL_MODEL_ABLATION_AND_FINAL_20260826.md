# 成功模型过程：canonical 双前端消融与最终结果

> 冻结记录，2026-08-26。本文只记录本项目成功保留的模型路线、累积消融和最终结果；
> 不收录外部论文比较，也不把失败分支混入主表。指标均为 Human3.6M S9/S11、
> action-equal All-17 absolute MPJPE（mm），越低越好。

## 1. 最终保留的两条模型链

### HRNet-W32 输入

`canonical RUMPL（token10） → 22c canonical E2 → continuous-nowarp H18`

- body-canonical ray frame；`body canonical reg=1e-2`；pelvis prior 开启；
- Global Joint-Query 关闭；训练早期使用 10% token dropout，测试时关闭；
- E2 使用 11 个 generator 候选 + 11 个 confidence-triangulation 候选；
- H18 为 camera-independent、continuous-time、centered T=9、stride=5；
- clean S8 预选择 `continuous_nowarp`，best epoch 1；
- robust-torso 后续分支没有通过完整 V2/V3/V4 判据，未进入最终模型。

### ResNet-152 输入

`canonical Global Joint-Query RUMPL → 22c canonical E2 → uncertainty-seq025 H18`

- body-canonical ray frame；`body canonical reg=1e-4`；pelvis prior 关闭；
- Global Joint-Query residual 开启，depth=2，max delta=0.5；
- E2 候选组成与 HRNet 相同；
- H18 使用 7 维 label-free uncertainty gate、stage-balanced loss、
  sequence loss weight=0.25；
- clean S8 预选择 `uncertainty_seq025`，best epoch 2。

共同设置：E2 V2 temperature=0.4，V3/V4 temperature=1.8；H18 checkpoint 均只由
clean S8 holdout 选择，S9/S11 与遮挡集不参与模型选择。

## 2. Stage-1 clean 累积消融

### 2.1 ResNet-152

| 累积模型 | V2 | V3 | V4 | 相对上一行 |
|---|---:|---:|---:|---:|
| canonical Global Joint-Query generator | 30.617 | 23.712 | 22.434 | — |
| + canonical 22c E2（两 seed 均值） | 30.643 | 21.528 | 19.566 | +0.027 / **−2.184 / −2.868** |
| + selected uncertainty-seq025 H18 | **29.416** | **21.020** | **19.281** | **−1.228 / −0.508 / −0.285** |

解释：E2 的主要成功点是利用冗余候选显著降低 V3/V4；V2 仅有 0.027 mm 的可忽略
波动。H18 随后将三列全部降低，形成最终 ResNet baseline。

### 2.2 HRNet-W32

| 累积模型 | V2 | V3 | V4 | 相对上一行 |
|---|---:|---:|---:|---:|
| canonical token10 generator | 38.412 | 31.376 | 28.900 | — |
| + canonical 22c E2（两 seed 均值） | 38.423 | 29.776 | 27.708 | +0.011 / **−1.600 / −1.192** |
| + selected continuous-nowarp H18 | **37.392** | **29.501** | **27.713** | **−1.031 / −0.275 / +0.005** |

解释：E2 同样主要改善 V3/V4；H18 明显改善 V2/V3，V4 相对两-seed E2 均值仅
0.005 mm 波动。按 selected-path 的 matched H18 baseline，时序净变化为
`−1.137/−0.392/−0.089 mm`，因此最终 H18 在其严格匹配输入上三列均改善。

## 3. clean H18 候选选择消融

模型先按 clean S8 holdout 选择，再只对 S9/S11 做一次最终报告。

### ResNet-152

| H18 变体 | S8 holdout mean | best epoch | S9/S11 V2/V3/V4 |
|---|---:|---:|---:|
| canonical H18 | 10.0016 | 9 | 29.533/21.044/19.315 |
| uncertainty-stagebalanced | 9.9988 | 3 | 29.336/20.996/**19.268** |
| **uncertainty-seq025（最终）** | **9.9959** | **2** | **29.416/21.020/19.281** |

最终选择依据是 S8 holdout，不是 S9/S11 单列最小值，所以保留 `uncertainty-seq025`。

### HRNet-W32

`continuous_nowarp` 的 S8 holdout mean 为 `14.3929 mm`、best epoch 1，最终为
`37.392/29.501/27.713 mm`。两个 uncertainty 扩展没有获得更低的 S8 选择指标，
最终保持 continuous-nowarp，不增加无效复杂度。

## 4. Stage-1 clean 最终冻结值

| 输入 | V2 | V3 | V4 |
|---|---:|---:|---:|
| **ResNet-152 完整 T=9** | **29.416** | **21.020** | **19.281** |
| **HRNet-W32 完整 T=9** | **37.392** | **29.501** | **27.713** |

## 5. Stage-2 VOC Occ-2/Occ-3：当前 H18 消融

协议：26,269 个 dense 时序中心组提供上下文，2,021 个中心组评分；Occ-2/Occ-3
分别遮挡 2/3 个源视角，每个被选视角两个 VOC 物体，scale 0.2--0.5，seed 42；
所有 V2/V3/V4 相机组合；模型只在 clean H36M 上训练。

| 输入/设置 | matched T=1 V2/V3/V4 | 完整 T=9 V2/V3/V4 | H18 降低误差 |
|---|---:|---:|---:|
| ResNet Occ-2 | 49.739/27.886/22.672 | **45.278/25.652/21.349** | **4.461/2.235/1.323** |
| ResNet Occ-3 | 56.699/30.897/24.607 | **51.111/27.862/22.653** | **5.588/3.035/1.954** |
| HRNet Occ-2 | 57.996/34.088/29.667 | **53.966/32.204/28.705** | **4.030/1.884/0.962** |
| HRNet Occ-3 | 63.893/36.498/31.153 | **58.852/33.970/29.695** | **5.041/2.528/1.458** |

H18 在两种输入、两种遮挡强度和全部视角数上均有效，而且遮挡收益明显大于 clean。

## 6. Stage-2 最终冻结值

| 输入 | Occ-2 V2 | Occ-2 V3 | Occ-2 V4 | Occ-3 V2 | Occ-3 V3 | Occ-3 V4 |
|---|---:|---:|---:|---:|---:|---:|
| **ResNet-152 完整 T=9** | **45.278** | **25.652** | **21.349** | **51.111** | **27.862** | **22.653** |
| **HRNet-W32 完整 T=9** | **53.966** | **32.204** | **28.705** | **58.852** | **33.970** | **29.695** |

## 7. 与旧完整模型的同协议最终对比

旧 checkpoint 已在相同 VOC 图像、二维前端、中心帧和 T=9 窗口上重放。负数代表
当前模型更好。

| 输入/设置 | 旧完整 T=9 | 当前完整 T=9 | 当前−旧 |
|---|---:|---:|---:|
| ResNet Occ-2 | 48.092/27.529/22.123 | **45.278/25.652/21.349** | **−2.814/−1.877/−0.774** |
| ResNet Occ-3 | 54.351/30.527/23.734 | **51.111/27.862/22.653** | **−3.239/−2.665/−1.081** |
| HRNet Occ-2 | **52.561**/32.479/**28.270** | 53.966/**32.204**/28.705 | +1.405/**−0.275**/+0.435 |
| HRNet Occ-3 | **56.937/33.820/29.341** | 58.852/33.970/29.695 | +1.914/+0.150/+0.354 |

必须保留的限制：当前 ResNet 相对旧完整模型六列全优；当前 HRNet 只有 Occ-2 V3
更好，其余五列退化。因此“新泛化模型全面超过旧模型”只对 ResNet 成立。

## 8. 冻结 checkpoint 与结果入口

### H18 checkpoints

- HRNet：
  `/mnt/data/cjyoutput/camera_generalization_20260824/hrnet_token10_generalization_20260825/canonical_h18/model_continuous_nowarp/model_best.pth.tar`
  SHA256 `d0796f9820cb272590878db245e2c9e817f28df1e992045eaff1a4a39a0e3b1b`
- ResNet-152：
  `/mnt/data/cjyoutput/camera_generalization_20260824/stage1_h36m_dual_frontend/resnet152/canonical_h18/model_uncertainty_seq025/model_best.pth.tar`
  SHA256 `b7aaa26ffbef24c9d965e441aa6ac3eaebef471f777c2235090cef6bbdddfaf9`

### 结果入口

- clean 选择：
  `/mnt/data/cjyoutput/camera_generalization_20260824/final_temporal_selection_20260825.json`
- VOC 最终表：
  `/mnt/data/cjyoutput/h36m_occ_voc_dense_20260824/final_occ23_table.json`
- 旧/新完整模型复测：
  `/home/lixiaob/cjy/OpenRUMPL_baseline_audit/VOC_OCC23_OLD_VS_CANONICAL_T9_20260826.md`
- 本记录机器可读版：
  `/home/lixiaob/cjy/SUCCESSFUL_CANONICAL_MODEL_ABLATION_AND_FINAL_20260826.json`
