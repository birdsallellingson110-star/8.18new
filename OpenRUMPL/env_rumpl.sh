# source 此文件即可进入 RUMPL 环境
#   source /home/lixiaob/cjy/OpenRUMPL/env_rumpl.sh
#
# 环境: python3.10 + torch2.1.0+cu118 + mmcv2.1.0(预编译) + mmpose1.3.2 + mmdet3.3.0 + numpy1.26.4
#   - 选这套是因为 mmpose1.3.2 要 mmcv<2.2.0，mmcv2.1.0 预编译仅到 torch2.1/py3.11
#   - amass / human_body_prior 源码经 c2i_sources.pth 挂载自 /mnt/data/dataset/c2i
#   - HRNet 权重复用师弟缓存 (TORCH_HOME)
#   (旧的 rumpl_venv 是 py3.12+torch2.7，mmcv._ext ABI 不兼容，已弃用)

source /home/lixiaob/cjy/rumpl_venv310/bin/activate

export PYOPENGL_PLATFORM=egl                 # pyrender 走 EGL 离屏
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # 双卡按总线编号，避免索引错乱
export TORCH_HOME=/mnt/data/dataset/c2i/torch  # 复用师弟的 HRNet 权重缓存
export UV_CACHE_DIR=/mnt/data/cjydata/.uv_cache
export PIP_CACHE_DIR=/mnt/data/cjydata/.pip_cache
export XDG_CACHE_HOME=/mnt/data/cjydata/.cache   # 所有 ~/.cache 写入转挂载盘, 防系统盘被占

echo "[rumpl] venv=rumpl_venv310 | python=$(python --version 2>&1 | awk '{print $2}')"
echo "[rumpl] PYOPENGL_PLATFORM=$PYOPENGL_PLATFORM  CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER  TORCH_HOME=$TORCH_HOME"
echo "[rumpl] 输出→/mnt/data/cjyoutput  数据→/mnt/data/cjydata"
