#!/bin/bash
# BGE-Large-ZH-v1.5 模型下载脚本
# 用法: bash download_model.sh
# 模型大小: ~2.1GB
# 运行本项目前必须先下载模型

set -e

MODEL_DIR="$(dirname "$0")/BGE-Large-ZH-v1.5"

# 方式1: 从 ModelScope 下载（推荐，国内速度快）
install_modelscope() {
    pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/
}
download_from_modelscope() {
    echo "正在从 ModelScope 下载 BGE-Large-ZH-v1.5..."
    python3 -c "
from modelscope import snapshot_download
snapshot_download('BAAI/BGE-Large-ZH-v1.5', cache_dir='/tmp/bge_cache')
import shutil, os
src = os.path.join('/tmp/bge_cache', 'BAAI/BGE-Large-ZH-v1.5')
dst = '$MODEL_DIR'
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print('下载完成！模型路径:', dst)
"
}

# 方式2: 从 HuggingFace 下载（备选，需要科学上网）
download_from_huggingface() {
    echo "正在从 HuggingFace 下载 BAAI/BGE-Large-ZH-v1.5..."
    pip install huggingface-hub
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/BGE-Large-ZH-v1.5', local_dir='$MODEL_DIR')
print('下载完成！模型路径: $MODEL_DIR')
"
}

# 主流程
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/pytorch_model.bin" ]; then
    echo "模型已存在，跳过下载。"
    exit 0
fi

echo "========================================"
echo " BGE-Large-ZH-v1.5 模型下载"
echo "========================================"
echo "1) 从 ModelScope 下载（国内推荐）"
echo "2) 从 HuggingFace 下载（备选）"
echo "========================================"
read -p "请选择下载方式 (1/2): " choice

case $choice in
    1)
        install_modelscope
        download_from_modelscope
        ;;
    2)
        download_from_huggingface
        ;;
    *)
        echo "无效选择，使用 ModelScope"
        install_modelscope
        download_from_modelscope
        ;;
esac
