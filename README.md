# 知识库管理系统 (Knowledge Base)

基于 RAG（检索增强生成）的本地知识库管理系统。上传文档 → 自动切片 → 向量化 → 语义搜索。

## 技术栈

- **后端**: Python FastAPI
- **前端**: 单页 HTML（原生 JS）
- **向量库**: ChromaDB + LangChain
- **嵌入模型**: BAAI/BGE-Large-ZH-v1.5
- **文档解析**: PyMuPDF, python-docx, python-pptx, openpyxl

## 快速开始

### 1. 下载模型（必须）

本项目依赖 BGE-Large-ZH-v1.5 嵌入模型（~2.1GB）。

```bash
# 方式一：自动下载（推荐，国内速度快）
bash download_model.sh

# 方式二：手动使用 ModelScope
pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/
python3 -c "
from modelscope import snapshot_download
import shutil, os
snapshot_download('BAAI/BGE-Large-ZH-v1.5', cache_dir='/tmp/bge_cache')
src = os.path.join('/tmp/bge_cache', 'BAAI/BGE-Large-ZH-v1.5')
shutil.copytree(src, 'BGE-Large-ZH-v1.5')
"

# 方式三：HuggingFace（需科学上网）
pip install -U huggingface-hub
huggingface-cli download BAAI/BGE-Large-ZH-v1.5 --local-dir BGE-Large-ZH-v1.5
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 启动服务

```bash
python3 -m uvicorn backend.main:app --host 0.0.0.0 --port 39082
```

### 4. 使用

浏览器打开 `http://localhost:39082/`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/knowledge-bases | 列出所有知识库 |
| POST | /api/knowledge-bases | 创建知识库 |
| POST | /api/knowledge-bases/{name}/upload | 上传文件 |
| GET | /api/knowledge-bases/{name}/files | 查看文件列表 |
| POST | /api/knowledge-bases/{name}/search | 搜索 |
| DELETE | /api/knowledge-bases/{name} | 删除知识库 |
| DELETE | /api/knowledge-bases/{name}/files/{id} | 删除文件 |

## 支持的文件格式

TXT, MD, CSV, JSON, HTML, DOCX, PDF, PPTX, XLSX, 图片（自动OCR描述），以及常见源代码文件

## 项目结构

```
knowledge-base/
├── backend/
│   ├── main.py              # FastAPI 主程序
│   ├── kb_engine.py         # 知识库引擎（ChromaDB + 向量化）
│   ├── document_loader.py   # 文档解析器
│   ├── multimodal.py        # 图片处理
│   └── __init__.py
├── frontend/
│   └── index.html           # 前端页面
├── docs/                    # 文档
├── download_model.sh        # 模型下载脚本
└── README.md
```
