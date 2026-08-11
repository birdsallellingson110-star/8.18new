# Gated Global Joint-View 实验记录（2026-07-22）

## 目的

上一轮普通global-JV相对no-drop curriculum5在V2改善`1.093 mm`，但V3/V4退化`0.126/0.060 mm`。本轮使用ReZero式可学习残差门控：

```text
X_out = X + alpha * (GlobalJV(X) - X), alpha初始化为0
```

训练起点严格保持原VFT输入，由数据学习全局上下文贡献。全局层不使用ray-distance/confidence bias；论文式偏置仍保留在原VFT的同关节跨视角注意力中。

## 双分支

| 分支 | GPU | global gate | token removal | 对照 |
|---|---:|---:|---:|---|
| `nodrop` | 0 | learnable, init 0 | 0 | no-drop curriculum5 seed0 |
| `stage5drop` | 1 | learnable, init 0 | 10%，仅epoch 0-4 | stage5-removal seed0 |

共同设置：seed0、scratch、20 epoch、epoch 0-4固定V2、epoch 5-19 random-K、无蒸馏及其他辅助loss。训练后统一评估V2/V3/V4/V5全部26组合，并记录最终learned gate值。

## 路径

- 服务：`rumpl-gbt-global-jv-gated-20260722.service`
- 日志：`/mnt/data/cjyoutput/rumpl_gbt_global_jv_gated_ablation_20260722.log`
- 最终报告：`/mnt/data/cjyoutput/rumpl_gbt_global_jv_gated_ablation_20260722.txt`
- 代码快照：`/mnt/data/cjyoutput/experiment_records/rumpl_gbt_global_jv_gated_20260722/`
- nodrop输出：`/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/rumpl_gbt_global_jv_gated_nodrop_20260722_2026-07-22_14-59-04/`
- stage5drop输出：`/mnt/data/cjyoutput/output/multiview_amass_rumpl/multiview_rumpl_999/rumpl_gbt_global_jv_gated_stage5drop_20260722_2026-07-22_14-59-04/`

## 启动核验

- 启动时间：2026-07-22 14:59 CST。
- 两分支日志均为`GLOBAL_JV depth=1 biased=0 gated=1`。
- 两分支均为`VIEW_SAMPLER epoch=0 mode=fixed-2 fixed_epochs=5`。
- stage5drop额外为`GBT_TOKEN_DROPOUT rate=0.1 active_epochs=5`。
- 首批前向与反向正常，无NaN、OOM或配置异常。
