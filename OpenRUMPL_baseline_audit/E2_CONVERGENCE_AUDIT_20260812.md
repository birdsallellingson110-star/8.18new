# E2 depth-2 收敛审计（2026-08-12）

## 目的

验证 E2（Set Transformer + joint-level counterfactual candidate utility）是否
因为只训练 15 轮而没有收敛。模型、数据、损失、候选集合和评估协议全部保持
与原 E2 seed0 相同，只从原第 14 轮最佳 checkpoint 继续训练到总 30 轮。

## 配置

- 原 checkpoint：
  `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/E_set_transformer_depth2/model_best.pth.tar`
- 续训输出：
  `/mnt/data/cjyoutput/open_source_fusion_audit_20260731/Counterfactual_View_Utility_20260811/E_set_transformer_depth2_extend30_seed0/`
- attention depth：2
- 原训练：10 轮 direct + 5 轮 GHT；续训第 15–29 轮继续 GHT
- batch size：512；seed：0；训练主体内部 holdout 选择 checkpoint
- S9/S11 只在最终选定 checkpoint 上评估一次

## 结果

| checkpoint | holdout V3/V4 平均指标 | S9/S11 V3 | S9/S11 V4 |
|---|---:|---:|---:|
| 原 E2 第14轮 | 18.5437 | 29.8195 | 29.0502 |
| 续训最佳第29轮 | **18.4510** | **29.8108** | **29.0380** |

续训在 holdout 上改善 0.0927 mm，但正式测试集只改善 0.0087 mm（V3）和
0.0122 mm（V4）。第 24、26、27 轮有小幅波动，第 29 轮达到本次最佳；训练
损失仍缓慢下降，但泛化到 S9/S11 的收益已经基本饱和。

## 结论

1. 原 E2 15 轮并非严格意义上的完全收敛；延长到 30 轮可以继续降低内部
   holdout 指标。
2. 但长训对 S9/S11 几乎没有影响，不能把“训练轮数不足”当作 E2 当前误差
   的主要原因。
3. E2 depth=2 仍是当前最稳定的单帧结构；后续不再优先堆训练轮数、attention
   深度或重复时序模块。
4. 下一阶段应做零训练候选池诊断：比较 confidence-weighted DLT、robust
   IRLS、pairwise hypotheses 等新增候选是否降低 oracle。如果 oracle 不降，
   不训练二阶段融合器；如果 oracle 明显下降，再把 E2 utility scorer 接到
   扩展候选池上。

完整训练日志和结果：

- `.../E_set_transformer_depth2_extend30_seed0/train.log`
- `.../E_set_transformer_depth2_extend30_seed0/result.json`

