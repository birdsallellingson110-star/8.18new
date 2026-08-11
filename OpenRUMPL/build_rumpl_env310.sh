#!/usr/bin/env bash
# 重建 RUMPL 环境：python3.10 + torch2.1.0+cu118 + mmcv2.1.0(预编译) + mmpose1.3.2 + mmdet3.3.0 + numpy1.x
# 原因：mmpose1.3.2 要 mmcv<2.2.0 → mmcv2.1.0(无cp312,仅到torch2.1) → torch2.1 → numpy1.x
# RUMPL/MHP 不用 pytorch3d，降级 torch 无损。amass/hbp 走 .pth，HRNet 权重复用师弟缓存。
set -euo pipefail
export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache
export UV_PYTHON_INSTALL_DIR=/mnt/data/cjydata/uv-python
export UV_LINK_MODE=copy
UV=/mnt/data/cjydata/uv-bin/uv
VENV=/home/lixiaob/cjy/rumpl_venv310
VPY=$VENV/bin/python

echo "==> [0] 建 python3.10 venv: $VENV"
$UV venv "$VENV" --python 3.10

echo "==> [1] torch 2.1.0 + torchvision 0.16.0 (cu118)"
$UV pip install --python "$VPY" torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

echo "==> [2] numpy 1.x + 核心科学栈"
$UV pip install --python "$VPY" "numpy==1.26.4" scipy matplotlib pandas opencv-python tqdm pyyaml plotly h5py tables scikit-image json_tricks

echo "==> [3] mmcv 2.1.0 (openmmlab 预编译 wheel, 匹配 torch2.1/cu118)"
$UV pip install --python "$VPY" mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

echo "==> [4] mmengine + mmdet 3.3.0 + mmpose 1.3.2"
$UV pip install --python "$VPY" mmengine "mmdet==3.3.0" "mmpose==1.3.2"

echo "==> [5] 渲染 + RUMPL 训练依赖"
$UV pip install --python "$VPY" pyrender trimesh easydict yacs wandb tensorboardX einops timm torchinfo shapely pymvg
# body_visualizer 从 uv 缓存的源码装(github 被墙)；先尝试本地缓存目录
BV_SRC=$(ls -d /mnt/data/cjydata/.uv_cache/git-v0/checkouts/*/*/body_visualizer 2>/dev/null | head -1)
if [ -n "${BV_SRC:-}" ]; then cp -a "$(dirname "$BV_SRC")/body_visualizer" "$VENV/lib/python3.10/site-packages/"; echo "  body_visualizer 已从缓存拷入"; fi

echo "==> [6] 挂载 c2i 的 amass / human_body_prior 源码 (.pth)"
printf '%s\n%s\n' /mnt/data/dataset/c2i/amass/src /mnt/data/dataset/c2i/human_body_prior/src \
  > "$VENV/lib/python3.10/site-packages/c2i_sources.pth"

echo "==> [7] 自检"
export PYOPENGL_PLATFORM=egl TORCH_HOME=/mnt/data/dataset/c2i/torch
"$VPY" - <<'PY'
ok=True
def chk(n,fn):
    global ok
    try: fn(); print(f"  ✓ {n}")
    except Exception as e:
        ok=False; print(f"  ✗ {n}: {type(e).__name__}: {e}")
import numpy as np, torch
print(f"  [info] py3.10 | numpy {np.__version__} | torch {torch.__version__} cuda{torch.version.cuda}")
def _t():
    assert torch.cuda.is_available(); x=torch.randn(16,16,device='cuda');(x@x).sum().item()
    print(f"      GPU={torch.cuda.get_device_name(0)} n={torch.cuda.device_count()}")
chk("torch CUDA kernel", _t)
chk("mmcv.ops MSDA", lambda: __import__('mmcv.ops',fromlist=['MultiScaleDeformableAttention']).MultiScaleDeformableAttention)
chk("mmpose.apis", lambda: __import__('mmpose.apis',fromlist=['MMPoseInferencer']).MMPoseInferencer)
chk("mmdet", lambda: __import__('mmdet'))
chk("human_body_prior", lambda: __import__('human_body_prior.body_model.body_model',fromlist=['BodyModel']))
chk("amass", lambda: __import__('amass'))
chk("body_visualizer", lambda: __import__('body_visualizer.mesh.mesh_viewer',fromlist=['MeshViewer']))
def _pyr():
    import pyrender; r=pyrender.OffscreenRenderer(64,64); r.delete()
chk("pyrender EGL", _pyr)
import sys; print("\n  结果:", "全部通过 ✅" if ok else "有失败 ❌"); sys.exit(0 if ok else 1)
PY
