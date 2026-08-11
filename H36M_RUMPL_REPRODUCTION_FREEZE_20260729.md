# H36M RUMPL 论文复现冻结清单（2026-07-29）

## 结论

RUMPL 的 Human3.6M 主结果已经达到可用于后续模块消融的工程复现标准。

论文 Table 2 的 RUMPL 2：

| 指标 | 论文 | 当前复现 | 差值 |
|---|---:|---:|---:|
| Absolute All-17 MPJPE | 52.500 mm | 51.874 mm | -0.626 mm |
| Absolute KP* MPJPE | 56.800 mm | 57.238 mm | +0.438 mm |

两个主指标的绝对差均小于 0.7 mm。此前 KP* 约 +2.64 mm 的主要差距已经定位并
解决：论文公开 H36M 检测脚本使用整图 `MMPoseInferencer` 的 person detector
bbox，而旧复现错误地使用 GT person bbox 直接做 top-down HRNet。

## 冻结训练权重

```text
/mnt/data/cjyoutput/h36m_paper_repro_20260728/output/
multiview_amass_rumpl/multiview_rumpl_999/
H8_h36m_r5scheduler_official_dualval_full128109_legmapfix_seed0_20260728_2026-07-29_00-49-44/
model_best.pth.tar
```

该权重来自论文路线的 AMASS/MHP 合成多视角训练，不是新生成的真实 H36M
训练集。

## 冻结测试输入

```text
/mnt/data/cjydata/datasets/h36m_rumpl_official/prepared/h36m/data/
datasets_mmpose/
annot_filtered_5_64_mmpose_hrnet_coco_inferencer_legswap/
h36m_validation.pkl
```

检测链：

```text
整张 H36M 图像
→ MMPoseInferencer
→ 默认 person detector bbox
→ HRNet-W32 COCO 384x288
→ H36M-17 映射
→ 已核验的 lower-body semantic correction
```

## 冻结评测协议

- S9/S11；
- 2-view all-pairs：每个时刻枚举 `C(4,2)=6` 个相机对；
- 12,126 个评测样本；
- Absolute world-coordinate MPJPE，不做 root/Procrustes 对齐；
- Table 2 使用 15 个动作等权平均；
- All-17 使用全部 17 点；
- KP* 使用 H36M 语义的肩、肘、腕、膝、踝共 10 点：
  `[11,14,12,15,13,16,5,2,6,3]`。

冻结结果：

```text
/mnt/data/cjyoutput/h36m_paper_repro_20260728/
H8_model_best_fullimage_inferencer_eval/table2.json
```

## 唯一未完全一致项

论文公开验证清单需要原始帧：

```text
1, 65, 129, 193, ...
```

当前百度数据包只提供：

```text
1, 6, 11, 16, ...
```

因此只能构造相邻近似帧：

```text
1, 66, 131, 196, ...
```

当前为 2,021 个四相机时刻，论文清单为 2,051 个。旧结果 bootstrap 显示该规模下
KP* 标准差约 0.31 mm；测试帧差异会产生小幅波动，但当前主指标已与论文同时对齐
到 0.7 mm 内。除非获得完整 50 fps 原始帧，不再把该不可观测差异作为阻塞项。

## 后续模块的比较规则

### 继续 AMASS/MHP 相机泛化路线

所有模块必须：

1. 从上述 H8 权重或完全相同训练 recipe 初始化/重训；
2. 使用冻结的 full-image Inferencer 验证 PKL；
3. 使用相同 all-pairs、Absolute、action-equal 评测；
4. 同时报告 All-17 和 KP*；
5. 不得重新使用 GT bbox 验证结果冒充论文口径；
6. 不得用固定单一相机对替代 all-pairs。

### 真实 H36M 训练路线

真实 H36M PKL 已生成并通过数据门禁，但“真实 H36M、无新增模块”的 H12 baseline
尚未训练。该路线必须先完成 H12 单帧原始 RUMPL，再以 H12 为控制组增加
TGR-Ray、偏置或时序模块。H8 负责证明论文代码/评测已经复现，H12 负责提供真实
数据训练下的模块对照；两者结果不能直接混在同一消融表中。

## 当前允许的表述

可以表述：

> 在公开代码、同一 HRNet 权重和可获得的相邻验证帧上，我们复现得到
> 51.87/57.24 mm，论文报告为 52.50/56.80 mm，两个指标差异均小于 0.7 mm。

不应表述：

- “逐样本完全复现”；
- “使用了与论文完全相同的验证帧”；
- 将 GT-bbox 旧结果作为正式论文基线；
- 将 H8 合成训练结果与未来 H12 真实训练模块结果直接作单变量比较。
