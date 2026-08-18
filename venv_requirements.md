# rumpl_venv310 依赖表

从当前虚拟环境 `rumpl_venv310` 于 2026-08-18 导出。完整可安装清单见 [`requirements.txt`](requirements.txt)。

## 环境

| 项 | 版本 |
| --- | --- |
| Python | 3.10.20 |
| pip | 26.1.2 |
| PyTorch | 2.1.0+cu118 |
| torchvision | 0.16.0+cu118 |
| CUDA | 11.8 |

## 核心依赖

| 类别 | 包 | 版本 |
| --- | --- | --- |
| 深度学习 | torch | 2.1.0+cu118 |
| 深度学习 | torchvision | 0.16.0+cu118 |
| 深度学习 | timm | 1.0.27 |
| 深度学习 | einops | 0.8.2 |
| 深度学习 | triton | 2.1.0 |
| OpenMMLab | mmcv | 2.1.0 |
| OpenMMLab | mmengine | 0.10.7 |
| OpenMMLab | mmdet | 3.3.0 |
| OpenMMLab | mmpose | 1.3.2 |
| 数值/数据 | numpy | 1.26.4 |
| 数值/数据 | scipy | 1.15.3 |
| 数值/数据 | pandas | 2.3.3 |
| 数值/数据 | h5py | 3.16.0 |
| 视觉 | opencv-python | 4.11.0.86 |
| 视觉 | pillow | 12.2.0 |
| 视觉 | scikit-image | 0.25.2 |
| 视觉 | pycocotools | 2.0.11 |
| 3D/几何 | pymvg | 2.1.0 |
| 3D/几何 | shapely | 2.1.2 |
| 3D/几何 | trimesh | 4.12.2 |
| 3D/几何 | pyrender | 0.1.45 |
| 人体模型 | chumpy | 0.70 |
| 人体模型 | amass | 1.0.1 |
| 训练工具 | tensorboardX | 2.6.5 |
| 训练工具 | wandb | 0.27.2 |
| 训练工具 | tqdm | 4.68.2 |
| 训练工具 | easydict | 1.13 |
| 检测 | ultralytics | 8.4.63 |

完整 108 个 pinned 包见 `requirements.txt`。

## 重建虚拟环境

```bash
python3.10 -m venv rumpl_venv310
rumpl_venv310/bin/pip install -U pip
rumpl_venv310/bin/pip install -r requirements.txt
```

若 `mmcv==2.1.0` 安装失败，再单独装 CUDA/Torch 匹配的 OpenMMLab wheel：

```bash
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```
