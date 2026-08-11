# RUMPL clip_full + conf 训练总结 (对标论文带 conf, Table 3 CMU 35.0)

## 背景
Phase0 的 clip_full 是 **w/o conf** 配置 (`POSEFORMER_CONCAT_CONFIDENCE_EMB` 默认 false),
对应论文 Table 3 的 `RUMPL (w/o Conf)` 行 (KP* 41.1mm)。论文主结果 35.0mm 是**带 conf** 的。
本次单变量实验: 在 clip_full 基础上**只开 conf**,其余不动,验证 conf 增益。

- config: `configs/cmu_panoptic/rumpl_amass/clip_full_conf.yaml`
  (= clip_full + `POSEFORMER_CONCAT_CONFIDENCE_EMB: true`,单变量)
- 模型: `multiview_rumpl`,`confidence_to_embedding: Linear(1→256)` 激活
- VFT/PFT 工作维度 **768** (dir 256 + inter 256 + conf 256;clip_full 是 512,无 conf)
- 注:本配置 anchor 仍是 clip_full 的 Closest + 归一化 (非论文 crf_4925 的光心/不归一化),
  若 CMU 不达标再逐项对齐 crf_4925 清单

## 训练 (GPU0, RTX 4090)
- 20 epoch, lr 1e-4 → @10/@15 ×0.1
- Training 12063s (~3.4h) + Testing 2531s,Total ~4h
- 显存 ~4.6GB,Speed ~2000-6000 samples/s
- model: `output/multiview_amass_rumpl/multiview_rumpl_999/run_conf_2026-06-17_15-38-54/`
  `{model_best, final_state}.pth.tar`

## Synth Val V=2 (训练内验证, 同分布) — 单位 cm

| 指标 | w/o conf (Phase0 训练末) | **带 conf** | 改善 |
|---|---|---|---|
| Relative | 2.087 | **1.859** | ↓0.23 |
| Absolute | 1.634 | **1.375** | ↓0.26 |

- val 单调下降 (1.877→1.867→1.863→1.861→1.859),无过拟合
- ✅ **conf 在 synth 上确认有效**

## 单人 CMU 真实评估 V=2 (对标论文 35.0) — 已完成

config: `cmu_eval_sp_v2_conf.yaml` (= cmu_eval_sp_v2 + 开 conf 一行)
**严格控制变量**: NETWORK 段 + 所有 DATASET 输入字段与训练 clip_full_conf **完全一致**
(diff 仅差 conf 一行), 只有数据路径/TEST_VIEWS 不同。model_best (epoch20), 697 组。

| V=2 Abs (mm) | w/o conf (Phase0) | 带 conf | 改善 | 论文 |
|---|---|---|---|---|
| 全17 (All-KP) | 42.1 | **37.9** | ↓4.2 | 30.8 |
| KP* (膝踝肩肘腕) | 43.4 | **40.4** | ↓3.0 | 35.0 |
| Rel 全17 | 58.0 | 56.5 | ↓1.5 | — |

### 结论
- ✅ **conf 有效**: V=2 Abs 全17 改善 4.2mm, 方向与论文一致 (论文 conf 改善 ~5.8-6.1mm)
- ⚠️ **未达论文 35.0**: KP* 还差 5.4mm; 我们 conf 增益 (3mm) < 论文 (6.1mm)
- 推测原因: 我们 anchor = Closest + 归一化 (clip_full 默认), 论文 crf_4925 用**光心 + 不归一化**;
  conf 增益在不同 anchor 下可能不同
- 下一步 (选 A 逻辑): 逐项对齐 crf_4925 清单 (光心 intersection / 不归一化 / DIM128),
  每次一项, 定位哪项补上剩余 5mm

## 控制变量审计 (确认评估=训练, 结果可信) — 已通过

对比**训练/评估运行时完整 config**(log 打印的 default+yaml+args 合并后真实值,非仅 yaml):

| 审计项 | 结论 |
|---|---|
| NETWORK 段(模型架构) | **零差异** — DIM/CONCAT_CONFIDENCE/INTERSECTION/depth/heads/mlp/ADD_3D_POS 全一致 |
| DATASET 输入字段 | **零差异** — INPUTS_NORMALIZED/NORMALIZE_CAMERAS/INTERSECTION_RAY_WITH/USE_T/CONCAT_FIRST/ZERO_TOKENS 全一致 |
| 仅有的 11 处差异 | 全是评估本该不同的: 数据路径(ROOT/AMASS_DATASET_TYPE/TEST_CMU_DATASET_NAME)、视角(TEST_VIEWS=[3,6]=V2)、dataset类(CMU真实)、MODEL_FILE、batch、TRAIN段(评估不训练,无关) |
| CMU dataset ray | `multiview_cmu_panoptic_rumpl.py:413` = concat[dir,inter,conf] **7维含conf**,与amass训练一致;conf来自真实HRNet置信度 |
| ray几何函数 | `create_3d_ray_coords`/`generate_direction_*` 在基类 `joints_dataset_rumpl`,CMU与amass **共享同一套** |
| 模型加载 | 成功无 shape mismatch (768维权重正确加载;跑完 88batch/697组,Loss 0.03-0.04 合理=铁证) |

**结论: 仅 conf 一个变量 (train/eval 都开),其余架构+输入逐字段一致。37.9mm(全17)/40.4mm(KP*) 是严格控制变量下的可信结果。**
