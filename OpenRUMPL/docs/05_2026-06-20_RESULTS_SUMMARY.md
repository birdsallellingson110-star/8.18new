# RUMPL 复现 + ST-VFT 实验结果汇总

> 更新: 2026-06-20 · 汇总所有已完成实验的精度结果与对标
> 详细分项见: [training_summary.md](training_summary.md)(w/o conf 训练) · [training_summary_conf.md](training_summary_conf.md)(带 conf) · [cmu_eval_summary.md](cmu_eval_summary.md)(CMU 评估)

---

## 一、RUMPL Baseline 复现（核心资产）

### 配置
- 数据: AMASS 合成(MHP 生成), 18 子集, **128109 样本**(单帧版 paper_single_cmu)
- 模型: `multiview_rumpl`, H=8, 12 层 VFT + 12 层 PFT, 20 epoch, batch 32, lr 1e-4(@10/15 ×0.1), Adam, MPJPE loss
- 两个配置对照(**唯一变量 = conf**, 控制变量审计逐字段通过):
  - **w/o conf** (`clip_full`): `POSEFORMER_CONCAT_CONFIDENCE_EMB=false`, VFT 工作维度 512
  - **带 conf** (`clip_full_conf`): 开 conf, VFT 工作维度 768 (dir256+inter256+conf256)

### 1.1 Synth Validation — AMASS 合成同分布(easy), 单位 cm

**(a) 训练时验证**(随机视角 k∈[2,5]):
| 指标 | w/o conf | 带 conf | 改善 |
|---|---|---|---|
| Rel (KP*) | 2.087 | **1.859** | ↓0.23 |
| Abs | 1.634 | **1.375** | ↓0.26 |

**(b) model_best 固定视角**(w/o conf; 带 conf 未单独测固定视角):
| 视角 | Rel 全17 | Rel KP* | Abs 全17 | Abs KP* |
|---|---|---|---|---|
| V=5 | 2.09 | 2.16 | 1.63 | 1.50 |
| V=2 | 4.72 | 5.01 | 4.23 | 4.18 |

### 1.2 单人 CMU 真实评估 — 171204_pose5+6, 5 相机(3/6/12/13/23), skip64, 697 组, 单位 **mm**

| 配置 | 视角 | Rel 全17 | Rel KP* | Abs 全17 | Abs KP* |
|---|---|---|---|---|---|
| **w/o conf** | **V=5** | 49.5 | 46.4 | **31.0** | **28.8** |
| **w/o conf** | **V=2** | 58.0 | 59.9 | **42.1** | **43.4** |
| **带 conf** | **V=2** | 56.5 | — | **37.9** | **40.4** |

> 带 conf 的 V=5 未测; 带 conf 的 Rel KP* 未精确算(KP* Abs 40.4 为膝踝肩肘腕 10 关节均值)。

### 1.3 对标论文 (CMU, **Abs**, mm)

| 来源 | All-KP(全17) | KP* |
|---|---|---|
| 论文 RUMPL (带 conf, 主结果) | **30.8** | **35.0** |
| 论文 RUMPL (w/o conf) | 36.6 | 41.1 |
| 我们 w/o conf V=2 | 42.1 | 43.4 |
| 我们 **带 conf** V=2 | **37.9** | **40.4** |
| 我们 w/o conf V=5 | 31.0 | 28.8 |

### 1.4 关键结论
1. **忠实复现**: w/o conf V=2 KP* **43.4 ≈ 论文 w/o conf 41.1**(差 2.3mm)。
2. **conf 有效**: 带 conf V=2 Abs 全17 **42.1→37.9(↓4.2mm)**, 方向与论文一致(论文 conf 贡献 ~5.8-6.1mm)。
3. **视角鲁棒性(核心卖点)**: V=5 KP* **28.8mm** ≪ V=2, 多视角几何冗余强; 训练用 random-views k∈[2,5] 保住 2 视角性能。
4. **残余 gap**: 带 conf 40.4 vs 论文 35.0(~5mm)。原因: (a) 论文主结果代码 `pose_3d_fuser` **未开源**(我们用 `multiview_rumpl` 实现); (b) 缺 2 个 AMASS 子集; (c) HRNet 合成↔真实域差。这些是已知系统性 gap, 不影响时序增益的可归因性。

---

## 二、ST-VFT 时序融合（研究创新, 进行中）

### 2.1 方法(路线 A: 加载 baseline 权重 + 时序微调)
- 在带 conf baseline 基础上: **加载其 VFT/PFT/编码权重** + 新增 **TFT**(时序融合) + **残差门控**(gate 初始 0)
- 数据流: ray 编码 → TFT(跨 L=5 帧时序) → VFT(跨 V 视角, baseline 权重) → PFT(跨 J 关节, baseline 权重) → 3D
- ray 表示与 baseline 完全一致(Closest intersection + 归一化 + conf), 唯一新增 = 时序

### 2.2 中等规模验证(160clip, 3 子集) — 已通过
| 验证项 | 结果 |
|---|---|
| 前置门槛(float64 铁证) | gate=0 **逐元素还原 baseline**(差 4.6e-14)→ 门控+权重+通路 100% 正确 |
| gate=0 初始 loss | **33.6mm** = baseline 量级 → 路线 A 成立 |
| gate 机制 | ReZero 陷阱(gate=0 冻 TFT), gate.grad 1.7e-2 健康(时序有梯度信号) |
| root cause 修复 | 04 room 范围错(训练 ROOM vs official room-size), mean_abs **0.77→0.49** 对齐 baseline |

> 160clip 数据太少, 时序增益(gate 长起来 loss 降)未显现 → 需全量定论。

### 2.3 待全量(进行中)
- 全量 3000clip render 中(6worker, ~2 天)
- 训练计划: gate lr=1e-3, TFT lr=3e-5(不动), epoch 30-50(一次只动 gate lr)
- 成功判据: 训练 loss < 33.6 + gate 长起来(早期信号)
- **金标准**: CMU 连续帧上 **B(gate=0) vs C(完整 ST-VFT)** 的增益(隔离"窗口输入"影响)

---

## 三、口径说明(避免混淆)
- **Rel(relative)**: pelvis(mid-hip)锚定后的 MPJPE; **Abs(absolute)**: 绝对世界坐标 MPJPE。论文主报 **Abs + KP***。
- **KP***: 论文口径子集 = 肩/肘/腕/膝/踝 10 关节(COCO 与 3D GT 定义一致的点); **全17(All-KP)**: 全部 17 关节。
- **Synth Val**: AMASS 合成同分布(easy, cm 级); **CMU 真实**: 跨域评估(真实图+HRNet, mm 级, 论文对标口径)。
- **V=2 / V=5**: 测试用视角数。
