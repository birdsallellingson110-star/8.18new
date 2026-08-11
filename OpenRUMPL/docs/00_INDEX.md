# RUMPL 项目文档索引(按时间)

本项目从"复现RUMPL"到"多视角→少视角自蒸馏(相机配置鲁棒性)"的完整实验/思路记录。

| # | 日期 | 文档 | 一句话 |
|---|------|------|--------|
| 01 | 06-10 | RUMPL_REPRODUCE_GUIDE | 复现RUMPL的环境/数据/训练指南 |
| 02 | 06-12 | training_summary | 首轮训练结果小结 |
| 03 | 06-15 | cmu_eval_summary | CMU跨数据集评估口径与结果 |
| 04 | 06-18 | training_summary_conf | conf版训练结果(baseline KP* abs 40.40踩实) |
| 05 | 06-20 | RESULTS_SUMMARY | 阶段性结果汇总 |
| 06 | 06-20 | EXPERIMENT_PROGRESS_AND_PLAN | 实验进度与计划 |
| 07 | 06-26 | STVFT_PROGRESS_AND_GPU_RESUME | 时序(ST-VFT)进度;时序被判定死路 |
| 08 | 06-27 | CAA_MILESTONE | 转向CAA/置信度方向的里程碑 |
| 09 | 06-28 | PAPERS_SUMMARY_AND_DIRECTION | 10篇论文调研 + V=2优化方向(骨长/三角化) |
| 10 | 07-05 | NEXT_MODULE_PLAN | 蒸馏baseline之上的下一模块规划(DeProPose/S2DHand/DVGaze) |

## 当前主线(截至最新)
- **方法**: 多视角→少视角自蒸馏(train时 teacher全视角 no_grad 蒸给 student随机少视角)
- **故事**: 对相机配置鲁棒的少视角3D姿态(RUMPL对相机俯视几何敏感 → 蒸馏注入满视角先验)
- **结果**: CMU pose5/6 V=2 全10相机配置, 配置平均 -2.02mm(3-seed)/ seed1全降 -2.42mm, 最坏配置-5~6mm
- **进行中**: leg-weight扫描(治[3,13]微退); 下一模块=DeProPose可靠度融合(见10)
- **死路(勿重试)**: 骨长约束、时序、换2D检测器、RCG、特征/合并蒸馏

> 记忆锚点见 `~/.claude/.../memory/distillation-fewview-works.md` 等。最新细节以对话为准。
