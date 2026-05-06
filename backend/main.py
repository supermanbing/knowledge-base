from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import logging

# 详细日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kb")
REQUEST_LOG_TEMPLATE = '  {method} {path} → {status} ({client})'
UPLOAD_LOG_TEMPLATE = '  UPLOAD: {files} | chunk_size={cs} overlap={co}'

from backend.document_loader import process_document
from backend.kb_engine import KnowledgeBaseEngine


BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BACKEND_DIR / "uploads"
CHROMA_DIR = BACKEND_DIR / "chroma_db"
MODEL_PATH = "/home/projects/knowledge-base/BGE-Large-ZH-v1.5/"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
PORT = 39082
SUPPORTED_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".vue",
    ".java", ".cpp", ".c", ".h", ".hpp", ".rs", ".go",
    ".rb", ".php", ".sh", ".bash", ".zsh",
    ".sql", ".css", ".scss", ".less",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".svg",
    ".csv", ".json", ".html", ".htm",
    ".docx", ".pdf", ".pptx",
    ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp",
}
CHUNK_PARAMS_FILE = BACKEND_DIR / "chunk_params.json"


class KBCreateRequest(BaseModel):
    name: str = Field(..., description="知识库名称")


class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索问题")
    k: int = Field(5, ge=1, le=20, description="返回条数")


class ChunkParamsRequest(BaseModel):
    chunk_size: int = Field(..., gt=0, description="分块大小")
    chunk_overlap: int = Field(..., ge=0, description="分块重叠")


app = FastAPI(title="知识库管理系统", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start = time.time()
    body = None
    is_upload = request.url.path.endswith("/upload")
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        status = response.status_code
        client = request.client.host if request.client else "?"
        path = request.url.path
        logger.info(REQUEST_LOG_TEMPLATE.format(method=request.method, path=path, status=status, client=client) + f" {duration:.0f}ms")
        if status >= 400 and body:
            try:
                files = [f.filename for f in body.getlist("files")] if body else []
                cs = body.get("chunk_size", "?")
                co = body.get("chunk_overlap", "?")
                logger.warning(UPLOAD_LOG_TEMPLATE.format(files=files, cs=cs, co=co))
            except Exception:
                pass
        if status >= 400:
            resp_body = b""
            async for chunk in response.body_iterator:
                resp_body += chunk
            logger.warning(f"  错误响应: {resp_body.decode()[:200]}")
            from starlette.responses import Response
            return Response(content=resp_body, status_code=status, media_type="application/json")
        return response
    except Exception as e:
        logger.error(f"  请求异常: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


def json_error(message: str, status_code: int = 400) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": message})


try:
    kb_engine = KnowledgeBaseEngine(persist_dir=str(CHROMA_DIR), model_path=MODEL_PATH)
    ENGINE_ERROR = ""
except Exception as exc:  # pragma: no cover - 依赖缺失时保留服务可启动性
    kb_engine = None
    ENGINE_ERROR = str(exc)


def load_chunk_params() -> dict[str, int]:
    if not CHUNK_PARAMS_FILE.exists():
        return {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}
    try:
        import json

        data = json.loads(CHUNK_PARAMS_FILE.read_text(encoding="utf-8"))
        chunk_size = int(data.get("chunk_size", DEFAULT_CHUNK_SIZE))
        chunk_overlap = int(data.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP))
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("invalid chunk params")
        return {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    except Exception:
        return {"chunk_size": DEFAULT_CHUNK_SIZE, "chunk_overlap": DEFAULT_CHUNK_OVERLAP}


def save_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    import json

    CHUNK_PARAMS_FILE.write_text(
        json.dumps({"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def require_engine() -> KnowledgeBaseEngine:
    if kb_engine is None:
        raise RuntimeError(f"知识库引擎初始化失败: {ENGINE_ERROR}")
    return kb_engine


def validate_kb_exists(name: str) -> None:
    engine = require_engine()
    if name not in engine.list_knowledge_bases():
        raise ValueError("知识库不存在")


def save_upload_file(upload: UploadFile) -> tuple[str, Path]:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("暂不支持的文件格式，支持 TXT/MD/CSV/JSON/HTML/DOCX/PDF/PPTX/XLSX/图片/源代码等")
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / unique_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return unique_name, destination


@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError) -> JSONResponse:
    return json_error(str(exc), status_code=400)


@app.exception_handler(RuntimeError)
async def runtime_error_handler(_, exc: RuntimeError) -> JSONResponse:
    return json_error(str(exc), status_code=500)


@app.get("/api/knowledge-bases")
async def list_knowledge_bases() -> dict[str, Any]:
    engine = require_engine()
    return {"items": engine.list_knowledge_bases()}


@app.post("/api/knowledge-bases", response_model=None)
async def create_knowledge_base(payload: KBCreateRequest) -> dict[str, Any] | JSONResponse:
    engine = require_engine()
    created = engine.create_knowledge_base(payload.name)
    if not created:
        return json_error("知识库已存在", status_code=409)
    return {"message": "创建成功"}


@app.delete("/api/knowledge-bases/{name}", response_model=None)
async def delete_knowledge_base(name: str) -> dict[str, Any] | JSONResponse:
    engine = require_engine()
    upload_paths = engine.get_file_upload_paths(name) if name in engine.list_knowledge_bases() else []
    deleted = engine.delete_knowledge_base(name)
    if not deleted:
        return json_error("知识库不存在", status_code=404)
    for upload_path in upload_paths:
        disk_path = UPLOAD_DIR / upload_path
        if disk_path.exists():
            disk_path.unlink()
    return {"message": "删除成功"}


@app.get("/api/chunk-params")
async def get_chunk_params() -> dict[str, int]:
    return load_chunk_params()


@app.put("/api/chunk-params")
async def update_chunk_params(payload: ChunkParamsRequest) -> dict[str, Any]:
    if payload.chunk_overlap >= payload.chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")
    save_chunk_params(payload.chunk_size, payload.chunk_overlap)
    return {"message": "分块参数已更新"}


@app.post("/api/knowledge-bases/{name}/upload")
async def upload_files(
    name: str,
    files: list[UploadFile] = File(...),
    chunk_size: int | None = Form(None),
    chunk_overlap: int | None = Form(None),
) -> dict[str, Any]:
    params = load_chunk_params()
    if chunk_size is None:
        chunk_size = params["chunk_size"]
    if chunk_overlap is None:
        chunk_overlap = params["chunk_overlap"]
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能小于 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size")

    validate_kb_exists(name)
    engine = require_engine()
    uploaded_files: list[dict[str, Any]] = []

    for upload in files:
        if not upload.filename:
            continue
        file_id = uuid.uuid4().hex
        unique_name, saved_path = save_upload_file(upload)
        try:
            chunks = process_document(saved_path, chunk_size, chunk_overlap)
            if not chunks:
                raise ValueError(f"文件 {upload.filename} 未解析出有效内容")
            metadatas = []
            for index, _chunk in enumerate(chunks):
                metadatas.append(
                    {
                        "file_id": file_id,
                        "file_name": upload.filename,
                        "file_type": Path(upload.filename).suffix.lower(),
                        "upload_path": unique_name,
                        "chunk_index": index,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                    }
                )
            doc_ids = engine.add_documents(name, chunks, metadatas)
            uploaded_files.append(
                {
                    "file_id": file_id,
                    "file_name": upload.filename,
                    "chunk_count": len(doc_ids),
                }
            )
        except Exception:
            if saved_path.exists():
                saved_path.unlink()
            raise
        finally:
            upload.file.close()

    return {"message": "上传成功", "items": uploaded_files}


@app.get("/api/knowledge-bases/{name}/files")
async def get_files(name: str) -> dict[str, Any]:
    validate_kb_exists(name)
    engine = require_engine()
    return {"items": engine.get_files(name)}


@app.delete("/api/knowledge-bases/{name}/files/{file_id}", response_model=None)
async def delete_file(name: str, file_id: str) -> dict[str, Any] | JSONResponse:
    validate_kb_exists(name)
    engine = require_engine()
    files = engine.get_files(name)
    target = next((item for item in files if item["file_id"] == file_id), None)
    if target is None:
        return json_error("文件不存在", status_code=404)

    deleted_count = engine.delete_file_documents(name, file_id)
    upload_path = target.get("upload_path", "")
    if upload_path:
        disk_path = UPLOAD_DIR / upload_path
        if disk_path.exists():
            disk_path.unlink()

    return {"message": "删除成功", "deleted_chunks": deleted_count}


@app.post("/api/knowledge-bases/{name}/search")
async def search(name: str, payload: SearchRequest) -> dict[str, Any]:
    validate_kb_exists(name)
    engine = require_engine()
    query = payload.query.strip()
    if not query:
        raise ValueError("搜索内容不能为空")
    items = engine.search(name, query, k=payload.k)
    return {"items": items}


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("缺少依赖 uvicorn，请先安装。") from exc

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info",
        access_log=True,
    )
