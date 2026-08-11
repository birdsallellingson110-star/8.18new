#!/usr/bin/env bash
# =============================================================================
# RUMPL 环境一键搭建 + 自检 (本机: 2x RTX 4090 / sm_89)
#
# 关键设计：
#   - conda 环境建在挂载盘 /mnt/data/cjydata/envs/rumpl（系统盘只剩 ~52G）
#   - pip 缓存也指到挂载盘，避免装依赖撑爆系统盘
#   - 分阶段安装，每段失败立即停；自检放最后
#
# 用法：
#   bash setup_rumpl_env.sh           # 全装
#   bash setup_rumpl_env.sh check     # 只跑自检（环境已建好时）
#   bash setup_rumpl_env.sh mmlab     # 只重装 mmpose/mmdet/mmcv（最易出错的一段）
# =============================================================================
set -euo pipefail

ENV_PREFIX="/mnt/data/cjydata/envs/rumpl"
PY_VER="3.10"
export PIP_CACHE_DIR="/mnt/data/cjydata/.pip_cache"
mkdir -p "$PIP_CACHE_DIR" /mnt/data/cjydata/envs

# conda 函数加载
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

run_in_env() { conda run -p "$ENV_PREFIX" "$@"; }

# -----------------------------------------------------------------------------
create_env() {
  if [ -d "$ENV_PREFIX" ]; then
    echo "==> 环境已存在: $ENV_PREFIX (跳过创建)"
  else
    echo "==> 在挂载盘创建 conda 环境: $ENV_PREFIX (python=$PY_VER)"
    conda create -y -p "$ENV_PREFIX" "python=$PY_VER"
  fi
}

install_torch() {
  echo "==> [1/4] PyTorch 2.0.1 + cu118 (4090 sm_89 适用)"
  run_in_env pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
}

install_core() {
  echo "==> [2/4] 核心依赖 (numpy 必须 1.x)"
  run_in_env pip install \
    "numpy==1.23.5" \
    pickle5 tqdm PyYAML easydict opencv-python tensorboard scipy matplotlib \
    h5py einops timm wandb torchinfo
}

install_smpl() {
  echo "==> [3/4] AMASS / SMPL+H / 渲染依赖 (pyrender 需 EGL)"
  run_in_env pip install human_body_prior body_visualizer trimesh pyrender plotly
  # amass 框架：若已 clone 到 OpenRUMPL/amass 则可编辑安装
  if [ -d "amass" ]; then
    echo "    -> 检测到 ./amass，可编辑安装"
    run_in_env pip install -e ./amass
  else
    echo "    -> 未发现 ./amass，请手动: git clone https://github.com/nghorbani/amass.git && pip install -e amass"
  fi
}

install_mmlab() {
  echo "==> [4/4] OpenMMLab 栈 (mmpose 1.3.2 需 mmcv 2.x；用 openmim 解析匹配 torch 的 wheel)"
  run_in_env pip install -U openmim
  # mim 会自动挑与当前 torch/cuda 匹配的 mmcv 预编译 wheel
  run_in_env mim install "mmengine"
  run_in_env mim install "mmcv>=2.0.1,<2.2.0"
  run_in_env mim install "mmdet==3.3.0"
  run_in_env mim install "mmpose==1.3.2"
}

# -----------------------------------------------------------------------------
self_check() {
  echo
  echo "================= 自检 ================="
  run_in_env python - <<'PY'
import sys
ok = True
def chk(name, fn):
    global ok
    try:
        fn(); print(f"  ✓ {name}")
    except Exception as e:
        ok = False; print(f"  ✗ {name}: {type(e).__name__}: {e}")

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

def _torch():
    import torch
    assert torch.cuda.is_available(), "CUDA 不可用"
    cc = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"      GPU0 = {name}, compute_cap = {cc[0]}.{cc[1]}, n_gpu = {torch.cuda.device_count()}")
    assert cc[0] < 9 or (cc[0]==8), f"算力 {cc} 可能不被 torch2.0.1 支持 (需 sm<=8.9)"
    # 真正跑一个 kernel，确认不是 'no kernel image'
    x = torch.randn(8, 8, device='cuda'); (x @ x).sum().item()

def _numpy():
    import numpy as np
    assert np.__version__.startswith("1."), f"numpy {np.__version__} 不是 1.x"
    print(f"      numpy = {np.__version__}")

def _mmpose():
    from mmpose.apis import MMPoseInferencer  # noqa

def _smpl():
    from human_body_prior.body_model.body_model import BodyModel  # noqa

def _pyrender_egl():
    import pyrender, numpy as np
    scene = pyrender.Scene()
    r = pyrender.OffscreenRenderer(64, 64)  # 触发 EGL 上下文
    r.delete()

chk("torch + CUDA kernel", _torch)
chk("numpy 1.x", _numpy)
chk("mmpose import", _mmpose)
chk("human_body_prior (SMPL+H)", _smpl)
chk("pyrender EGL offscreen", _pyrender_egl)

import torch
print(f"      torch = {torch.__version__}")
sys.exit(0 if ok else 1)
PY
  echo "========================================"
}

# -----------------------------------------------------------------------------
case "${1:-all}" in
  check) self_check ;;
  mmlab) create_env; install_mmlab; self_check ;;
  all)
    create_env
    install_torch
    install_core
    install_smpl
    install_mmlab
    self_check
    echo
    echo "完成。每次使用前激活： conda activate $ENV_PREFIX"
    echo "并 export CUDA_DEVICE_ORDER=PCI_BUS_ID PYOPENGL_PLATFORM=egl"
    ;;
  *) echo "未知参数: $1 (可用: all | check | mmlab)"; exit 1 ;;
esac
