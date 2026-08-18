# E3b：并行关节查询残差解码器（2026-08-14）

## 目的

E2 的候选集合评分只改变已有 3D 候选的权重，候选池 oracle 相对 H76 的上限增益只有约 0.10/0.14 mm（V3/V4）。因此本实验转向信息通路：在不替换 RUMPL/H76 的射线、tri-anchor、VFT、PFT 和共享 3D head 的前提下，给每个关节增加一个直接读取全部观测 token 的 query decoder。

## 方法来源与边界

- Geometry-Biased Transformer：固定关节 query 对观测 token 做 cross-attention；
- MVGFormer（CVPR 2024）：以粗 3D/query 为条件，经过投影/多视角特征聚合后细化；本实验使用 H76 tri-anchor 作为 query 的几何条件，但不引入图像特征或时序；
- 与旧 `GBT_SET_DECODER` 的区别：旧分支替换 H76 VFT 并已明显退化；E3b 保留 H76 输出，仅在最终输出旁加零初始化 3D residual。

## 网络位置

```text
H76 rays -> ray embedding -> [J x V] observation tokens -> original VFT/PFT/head -> y_H76
                                      \-> joint-query decoder -> Δy (zero initialized)
                                                               y = y_H76 + Δy
```

global 变体的 query 可读取全部 `J×V` token；local 变体每个 query 只读取本关节的 `V` 个 token。两者均无 camera-ID，支持 2/3/4 视角和视角置换。

## 严格设置

- 初始化：同一个 H76 checkpoint；
- 冻结：H76 全部参数冻结，只训练 E3b 新增 query、memory/anchor embedding、Transformer decoder 和 residual head；
- residual head 最后一层零初始化，训练前输出严格为 H76；
- 训练数据：真实 H36M S1/S5/S6/S7/S8，A1D+H21 输入；测试 S9/S11；
- 输入/指标：绝对 All-17 MPJPE，2/3/4 视角全组合 action-equal；
- 训练：20 epoch，`LR=1e-4`，2–4 视角随机 camera subset，视角采样权重 3:1:1；
- 代码：`/home/lixiaob/cjy/OpenRUMPL/RUMPL/lib/models/multiview_rumpl.py`、`train_rumpl.py`；启动器：`launch_gbt_query_residual_20260814.sh`；
- 输出：`/mnt/data/cjyoutput/open_source_fusion_audit_20260731/GBT_Query_Residual_E3b_20260814/`。

## 运行状态

| 变体 | GPU | 状态 | 结果 |
|---|---:|---|---|
| global joint memory | 0 | 已完成 | V2/V3/V4 = 34.8387/30.4943/29.6876 mm |
| per-joint memory | 1 | 已完成 | V2/V3/V4 = 34.8243/30.4797/29.6929 mm |

H76 对照为 V2/V3/V4 = 34.8163/30.4890/29.6913 mm。相对 H76：

- global：`+0.0224/+0.0053/-0.0037` mm；
- per-joint：`+0.0080/-0.0093/+0.0016` mm。

变化均小于 0.03 mm，属于噪声范围，不能作为有效提升或论文主创新。由于实现前置检查已通过，结论不是“代码接错”，而是当前 H76 的最终 3D 残差读出没有提供可利用的信息。停止继续堆 query depth/seed，下一条路线转为直接改善跨视角二维观测（热图/候选级融合），并保持 H76 几何主干不变。

## 实现前置检查

在随机输入、固定 2 视角、`is_training=False` 条件下，加载同一 H76 checkpoint
分别构造“关闭 E3b”和“开启 E3b（global）”的模型。零初始化残差头时两者输出为：

```text
max_abs_diff = 0.000000000e+00
mean_abs_diff = 0.000000000e+00
```

因此 E3b 在训练开始前严格保持 H76；训练 scope 日志确认仅选择新增的
`gbt_query_*` 参数，共 1,593,091 个，H76 主干没有被误更新。
