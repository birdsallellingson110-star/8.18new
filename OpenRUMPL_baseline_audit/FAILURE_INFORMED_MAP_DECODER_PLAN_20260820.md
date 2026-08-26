# Failure-informed uncertainty-aware MAP decoder plan

更新：2026-08-20

## 1. 固定目标与协议

本阶段只解决坐标级单帧主表，不改变输入信息量：

- 输入仅为冻结 HRNet 的 2D 坐标、置信度、相机内外参，以及由这些量确定性构造的
  世界射线、三角化候选、重投影/射线残差和人体结构量；
- 不使用 RGB、crop、heatmap、图像特征、GT 2D、测试 GT 或额外检测器；
- H36M 使用 S1/S5/S6/S7/S8 训练，S9/S11 测试，V2/V3/V4 全部相机组合，
  action-equal absolute All-17 MPJPE；
- 时序只允许在单帧主干有效后作为独立 T=9 协议，不与当前结果混写。

冻结当前同口径最好模型 K96 limb-proposal：`38.017/29.143/27.089 mm`。第一阶段
目标为 V2 低于 36.8、V3 不退化、V4 向 26.0 靠近。

## 2. 失败经验与禁止重复清单

| 历史方向 | 已有证据 | 提取的原因 | 本阶段约束 |
|---|---|---|---|
| 冻结 H76 后加 query residual | 三列变化小于 0.03 mm | 已压缩输出上的无约束残差没有新证据 | 不在最终 3D 后再挂残差头 |
| 一层 global-JV 插入 H76 | G0/G1 约 78/47/43 mm | 改变旧主干优化轨迹，且输出仍是确定性回归 | 全局编码器只预测概率因子，不直接替代全部几何 |
| SGraFormer/GraFormer 插入或替换 | 明显退化 | 作用位置和旧表征不匹配 | 不再尝试同类中间层堆叠 |
| confidence/parallax/ray scalar bias | 与真实边际收益相关性约 -0.02~-0.08 | 单个手工标量不能识别相关检测偏差 | 使用完整法矩阵、向量残差和全身上下文 |
| DLT/IRLS/重投影 selector | 明显弱于 RUMPL；GT 比预测更远离检测射线 | HRNet 存在系统性相关误差，贴射线不等于贴 GT | 人体先验必须能有控制地偏离错误射线 |
| 固定骨长 tree MAP / PointDSC 兼容图 | 最佳约 0.005 mm，或 V3/V4 退化 | 平均骨长不能判断检测偏差，离散边界图信息不足 | 骨长只作弱约束，不作主求解依据 |
| 继续调 proposal/scorer/temperature | 最新 proposal 仅提升 0.05~0.09 mm；校准跨主体退化 | 离散评分线接近局部平台 | K96 只作为初值/辅助分布，不再主攻超参 |
| 全程 K=2 长训练 | V2 37.886，但 V3/V4 62.215/46.217 | 严重视角基数分布偏移 | 一个模型覆盖 V2/V3/V4，禁止按列拼 specialist |
| 后融合 temporal/MixSTE | clean 增益多为 0.1~0.2 mm | 时间进入时视角证据已被压缩 | 若做 T=9，时间必须进入概率先验和求解前端 |
| 学习 2D/ray correction | 训练留出改善、S9/S11 退化 | 学到主体相关 detector bias | M1 不改 2D 坐标，不直接监督射线修正 |

任何新实验启动前必须逐项检查本表；若只是在旧方法上换名称或超参数，直接停止。

## 3. 已确认的瓶颈

1. RUMPL 先按关节压缩视角，再做人体建模；全身模块无法追溯某条具体射线的责任。
2. 观测似然和人体先验混在确定性回归中，没有显式不确定性和冲突处理机制。
3. V2 的 pelvis 误差约 34.0 mm，非 root 的 root-relative 误差约 42.5 mm，根位置和
   肢体结构都需要改善。
4. V2 相机对 1-4/2-3 约 47.6/46.7 mm，其余四对约 33.4 mm；主要目标是修复坏对，
   不是继续挤压正常对。
5. K96 sampled oracle 为约 36.26/27.20/25.24 mm，说明已有候选分布包含更好解，
   但离散 convex fusion 无法稳定组装。

## 4. 唯一主线：K96 初始化的概率人体先验 + 可微 MAP

### 4.1 保留与替换

保留：

- RUMPL 的相机无关世界射线；
- H76/RUMPL 候选和 K96 limb-proposal 当前最好融合结果；
- 随机相机子集和同一 HRNet 坐标协议。

替换：

- 不再把 K96 soft convex average 当最终答案；
- 不再让一个确定性 MLP 同时隐式承担观测、先验和求解；
- 新模块显式输出人体先验均值、先验精度、观测精度和 selective trust，再由 3x3
  线性系统得到每个关节的 MAP 解。

### 4.2 信息流

```text
冻结 HRNet 坐标/置信度 + 相机
                |
                +--> 完整 JxV 射线 token ------------------+
                |                                           |
                +--> 冻结 H76/E2/K96 --> anchor X0          |
                                                            v
                         permutation-equivariant JxV encoder
                               |                 |
                         pose prior mu       measurement precision
                               +--------+--------+
                                        v
                         differentiable ray-MAP solve
                                        |
                         selective trust-region update
                                        v
                                 absolute 3D pose
```

对关节 j：

\[
A_j=\lambda_j I+\sum_v w_{jv}(I-d_{jv}d_{jv}^T),
\]

\[
b_j=\lambda_j\mu_j+\sum_vw_{jv}(I-d_{jv}d_{jv}^T)o_v,
\qquad X_j^{MAP}=A_j^{-1}b_j.
\]

最终使用学习的 trust-region 门控制 `X0 -> X_MAP` 的移动幅度。初始化近似恒等，
避免未经训练的新分支破坏当前最好模型。人体先验分解为 root translation 和
root-relative articulation 两个输出，分别对应诊断出的两类误差。

M1 固定使用 HRNet confidence 作为观测权重，只学习 pose prior 和 trust；M2 才加入
上下文条件化的异方差观测精度/污染门控，避免第一轮同时改变过多因素。

## 5. 实验阶段与门控

### M1：固定观测权重的 MAP pose prior

- 冻结 E2、limb utility、K96 scorer；
- K96 输出为 `X0`；
- 完整 JxV encoder 只输出 root/root-relative prior、prior precision 和 trust；
- 不修改 2D，不学习 ray correction，不加 bone/temporal/monotonic loss；
- 主损失为 absolute 3D，辅以小权重 root-relative 3D；
- 先做 2 batch smoke 和数值/置换检查，再做 3 epoch gate screen。

进入正式训练的门槛：内部 holdout 三列均值改善至少 0.15 mm，V2 不退化，输出无
NaN，view permutation 误差小于 1e-5 m。否则先检查 prior/trust 是否塌缩，不直接
扩展 epoch、深度或随机种子。

### M2：学习观测异方差和污染门控

只在 M1 过门后增加每个 joint-view 的相对 precision，输入为完整 token 上下文、
anchor-to-ray 向量残差和 confidence。精度有界并相对 confidence 初始化，防止网络
用无限权重绕过求解器。

门槛：相对 M1，V2 的 1-4 和 2-3 各下降至少 1 mm，其他四对平均退化不超过
0.5 mm；同时检查预测 precision 与真实边际误差的训练主体留出相关性。

### M3：候选分布/反事实效用进入 prior

只把 K96 候选均值、方差、limb disagreement 和 all-minus-one-view 差异作为额外
概率证据，不增加新候选族。门槛是兑现 sampled-oracle gap，而不是继续改善训练 loss。

### T9：最后的 early probabilistic temporal prior

只在 M1--M3 的单帧模型已稳定后进行。历史帧更新 `mu/Lambda_pose`，当前帧几何项
保持不变；正常动作必须至少改善 0.5 mm，遮挡表再验证更大收益。禁止输出后平滑。

## 6. 评估与停止规则

每个被采用实验必须同时报告：

- V2/V3/V4 action-equal 和 frame-weighted；
- 六个 V2 相机对；
- pelvis/root、root-relative non-root 和逐关节误差；
- 加视角 Negative View Rate；
- 输出到观测射线距离，防止以“更贴错误射线”伪装提升；
- 至少两个 seed，第二个 seed 只在第一个通过内部门控后运行。

禁止使用 S9/S11 选 epoch、温度或超参数。checkpoint 只由训练主体的
严格留一主体选择；最终确定训练轮数后再用全部训练主体重训一次。历史
`group_index % 10 == 0` 只适合不学习人体先验的浅层候选评分，不再用于先验/纠错
网络。若一个模块连续两次未通过预注册门槛，写入失败表并关闭方向，不再靠加
epoch/加层/换 seed 重跑。

## 7. 实现边界

新代码独立放在 `OpenRUMPL_baseline_audit/`，结果写到挂载盘
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/failure_informed_map/`。不直接修改当前
dirty 的 RUMPL 主干；M1 通过门控后再考虑模块化迁回正式模型。

## 8. M1 首次 gate 结果与即时修正

M1 首次 3 epoch 使用历史 `group_index % 10 == 0` 留出：

| 口径 | K96 anchor | M1 MAP | 变化 |
|---|---:|---:|---:|
| 训练主体内部 V2/V3/V4 均值 | 17.903 | 17.787 | -0.117 |
| S9/S11 V2 | 38.011 | 38.390 | +0.379 |
| S9/S11 V3 | 29.141 | 29.504 | +0.362 |
| S9/S11 V4 | 27.079 | 27.406 | +0.327 |

结果目录：`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/failure_informed_map/m1_gate_seed0/`。

该实验判定失败，不进入 M2。三列测试集同向退化，同时内部留出持续改善，说明随机
帧留出泄露了相同训练主体的姿态/检测偏差分布，无法筛选需要跨主体泛化的 learned
prior。这与过去 learned ray correction 的失败模式一致，不能再通过延长训练或增加
attention 深度处理。

即时修正不是调模型，而是先修正科学门控：训练 S1/S5/S6/S7、完整留出 S8；只有
S8 同时改善并超过 0.15 mm 门槛，才允许固定 epoch 后使用 S1/S5/S6/S7/S8 重训。
若严格主体留出也失败，则关闭“直接预测 pose prior delta”方向，转为候选分布约束的
prior 或外部预训练且冻结的 pose density，禁止再训练同类 residual/prior head。

### 8.1 严格 S8 结果与 pose-density gate

严格训练 S1/S5/S6/S7、留出 S8 后，M1 最佳只改善 `0.042 mm`，第 3 epoch 回落到
`0.019 mm`；仍未通过 0.15 mm 门槛。该方向正式关闭。

随后参考 ICCV 2021 *Probabilistic Monocular 3D Human Pose Estimation with
Normalizing Flows* 官方代码（本地 commit `ad2fdf2`），先做低容量冻结生成式先验
证伪：在 S1/S5/S6/S7 GT root-relative pose / bone direction 上拟合收缩 Gaussian，
只重排 K96 hypotheses，系数在 S8 选择。结果最佳仅 `0.006 mm`，S9/S11 反而约
退化 `0.030 mm`。因此不移植高容量 flow；“姿态是否常见”不能区分多个都合理、但
只有一个符合当前帧的候选。

下一步改为 observation-conditioned K96 scorer：坐标仍限制在已有 hypothesis 分布，
但 scorer 显式读取每个 hypothesis 对完整 `joint x view` 射线的有符号残差。结构先
在每个 view 内做全身 joint attention，再对同一 joint 做跨 view attention，确保
身体上下文在视角压缩前参与判断。该模块与旧 G0/query residual 不同：它不修改坐标、
不替换 RUMPL 主干、以零初始化 residual score 严格退化为 K96。

## 9. 低秩二维观测偏差 oracle（2026-08-20）

在 observation-conditioned K96 严格 S8 门控失败后，进一步检验“HRNet 误差是否
主要来自每帧/每视角共享的低维 crop/尺度偏差”。实现只读取相同的 HRNet 坐标、
置信度、相机和冻结 K96 anchor：

1. 从缓存射线精确恢复归一化相机平面坐标；
2. 用当前帧 GT 投影分别拟合共享 translation、scale+translation、similarity 和
   6 参数 affine；这些只作为不可部署的 oracle 上限；
3. 另在 S1/S5/S6/S7 学习不含 camera ID 的逐关节平均二维残差，在完整 S8 选择
   shrinkage；
4. 所有修正坐标转回 RUMPL 射线，以 K96 为先验执行闭式 ray-MAP；MAP 精度只在
   S8 选择，S9/S11 一次性评估。

完整主体数为 S1/S5/S6/S7 `65,101` 组、S8 `12,946` 组、S9/S11 `2,021` 组。
冻结 K96 anchor 为 `38.011/29.141/27.079 mm`。S8 选择后结果：

| 修正 | V2 | V3 | V4 | 三列均值相对 K96 |
|---|---:|---:|---:|---:|
| identity ray-MAP | 38.010 | 29.140 | 27.076 | -0.002 mm |
| scale+translation oracle | 37.999 | 29.121 | 27.051 | -0.020 mm |
| similarity oracle | 37.999 | 29.121 | 27.051 | -0.020 mm |
| affine oracle | **37.965** | **29.066** | **26.976** | **-0.075 mm** |
| train-only mean bias, shrink=0.25 | 38.010 | 29.140 | 27.077 | -0.002 mm |

affine 虽将归一化二维平均残差从 `0.009034` 降至 `0.008273`，但相对同一
identity MAP 只带来 `0.073 mm`；两视角六个相机对仅分别改善约
`0.055/0.053/0.046/0.041/0.042/0.043 mm`，没有重点修复 `1-4` 和 `2-3`。
train-only mean bias 相对 identity MAP 还退化 `0.0003 mm`。

因此 oracle 未通过预注册的 `1.5 mm` 总增益和 `1.0 mm` 独立增益门槛，学习式
低秩二维偏差网络不启动。该结果否决的是“整个人/整视角共享的低维检测变换”，
不等于二维观测没有问题；剩余误差更像逐关节、姿态条件化且与人体深度歧义耦合的
偏差，而这种偏差在现有坐标级单帧输入下没有可迁移可观测量。

实现：`diagnose_lowrank_observation_bias_20260820.py`。

结果：
`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/lowrank_observation_bias/full_subject_gate/result.json`。

## 10. 逐关节残差 PCA oracle：低维可预测空间不足（2026-08-20）

实现：`diagnose_joint_residual_pca_oracle_20260820.py`。

正式结果：

`/mnt/data/cjyoutput/gbt_aligned_hrnet_20260820/joint_residual_pca_oracle/full_subject_gate/result.json`。

严格边界：PCA 基只由 S1/S5/S6/S7 的 HRNet 归一化相机坐标残差拟合；S8
只选择 ray-MAP prior precision；S9/S11 用固定选择评估一次。K>0 的系数由
对应 held-out 样本的 GT 2D 残差投影得到，只是 oracle 上限，不能列入论文方法
主表。K=0 是仅使用训练主体均值残差的可部署控制。

| 方法 | 累计解释残差方差 | V2 | V3 | V4 | 三档均值 | 相对 K96 增益 |
|---|---:|---:|---:|---:|---:|---:|
| K96 anchor | — | 38.011 | 29.141 | 27.079 | 31.411 | — |
| PCA K=0 | 0.0% | 38.011 | 29.141 | 27.078 | 31.410 | 0.001 |
| PCA K=1 | 12.6% | 38.011 | 29.140 | 27.077 | 31.409 | 0.001 |
| PCA K=2 | 22.9% | 38.010 | 29.140 | 27.077 | 31.409 | 0.002 |
| PCA K=4 | 40.2% | 38.000 | 29.123 | 27.054 | 31.392 | 0.018 |
| PCA K=8 oracle | 61.7% | 37.581 | 28.513 | 26.271 | 30.788 | 0.622 |
| PCA K=16 oracle | 84.8% | 36.191 | 26.624 | 23.980 | 28.931 | 2.479 |
| PCA K=32 oracle | 99.999% | 0.112 | 0.078 | 0.060 | 0.084 | 31.327 |

K=8 的六个两视角组合均有小幅改善，但没有修复单个异常组合：1-2/1-3/
1-4/2-3/2-4/3-4 分别为 33.103/36.998/47.181/46.330/30.186/31.689
mm；K96 对应为 33.590/37.493/47.575/46.718/30.567/32.125 mm。

结论：

1. 低维（K<=8）GT oracle 都只有 0.62 mm，未通过“相对 K96 至少 1.5 mm、
   且相对 K0 MAP 至少 1.0 mm”的预测器启动门槛；不训练同类低维残差头。
2. K=16 虽有 2.48 mm oracle 空间，但它已经包含大量逐关节高频误差，而且系数
   来自测试真值。坐标、置信度和相机参数本身并不直接揭示 detector 的未知误差，
   可实现预测器只能获得这个上限的一部分，风险收益不足。
3. K=32 几乎恢复完整 GT 2D，所以接近零误差是三角化正确性的 sanity check，
   不是模型提升；再次说明标定、射线和求解器没有隐藏的 20 mm 级错误。
4. 与上一节 affine oracle 仅提升 0.075 mm 合并看，当前瓶颈不是共享平移、尺度、
   仿射或低维关节偏置。后续不再重复均值 bias、低秩 correction、简单时序平滑或
   仅调 MAP/置信度权重；需要引入能够提供“当前关键点是否错、错向哪里”的新证据，
   同时仍遵守坐标级输入协议。
