from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

try:
    import chromadb
except ImportError:  # pragma: no cover - 运行时依赖检查
    chromadb = None  # type: ignore[assignment]

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - 运行时依赖检查
    Chroma = None  # type: ignore[assignment]

try:
    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
except ImportError:  # pragma: no cover - 运行时依赖检查
    HuggingFaceBgeEmbeddings = None  # type: ignore[assignment]


def _sanitize_collection_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip())
    normalized = normalized.strip("._-")
    if not normalized:
        normalized = f"kb_{uuid.uuid4().hex[:8]}"
    if len(normalized) < 3:
        normalized = f"kb_{normalized}"
    return normalized[:63]


class KnowledgeBaseEngine:
    def __init__(self, persist_dir: str, model_path: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = model_path
        self.metadata_dir = self.persist_dir / "_meta"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.alias_file = self.metadata_dir / "collections.json"
        self.alias_map = self._load_alias_map()

        self._require_dependencies()
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _require_dependencies(self) -> None:
        missing = []
        if chromadb is None:
            missing.append("chromadb")
        if Chroma is None:
            missing.append("langchain-chroma")
        if HuggingFaceBgeEmbeddings is None:
            missing.append("langchain-community")
        if missing:
            joined = ", ".join(missing)
            raise RuntimeError(f"缺少依赖: {joined}，请先安装。")

    def _load_alias_map(self) -> dict[str, str]:
        if not self.alias_file.exists():
            return {}
        try:
            return json.loads(self.alias_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_alias_map(self) -> None:
        self.alias_file.write_text(
            json.dumps(self.alias_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _resolve_collection_name(self, name: str) -> str:
        return self.alias_map.get(name, name)

    def _ensure_collection_alias(self, name: str) -> str:
        collection_name = self.alias_map.get(name)
        if collection_name:
            return collection_name
        collection_name = _sanitize_collection_name(name)
        base_name = collection_name
        index = 1
        existing = set(self.alias_map.values())
        while collection_name in existing:
            collection_name = f"{base_name}_{index}"
            index += 1
        self.alias_map[name] = collection_name
        self._save_alias_map()
        return collection_name

    def list_knowledge_bases(self) -> list[str]:
        names = []
        collections = self.client.list_collections()
        reverse_alias = {value: key for key, value in self.alias_map.items()}
        for collection in collections:
            real_name = getattr(collection, "name", "")
            names.append(reverse_alias.get(real_name, real_name))
        return sorted(set(names))

    def create_knowledge_base(self, name: str) -> bool:
        kb_name = name.strip()
        if not kb_name:
            raise ValueError("知识库名称不能为空")
        if kb_name in self.alias_map:
            return False
        collection_name = self._ensure_collection_alias(kb_name)
        self.client.get_or_create_collection(name=collection_name)
        return True

    def delete_knowledge_base(self, name: str) -> bool:
        collection_name = self.alias_map.get(name)
        if not collection_name:
            return False
        self.client.delete_collection(collection_name)
        self.alias_map.pop(name, None)
        self._save_alias_map()
        return True

    def get_or_create_collection(self, name: str) -> Chroma:
        if name not in self.alias_map:
            self.create_knowledge_base(name)
        collection_name = self._resolve_collection_name(name)
        return Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
        )

    def add_documents(
        self,
        kb_name: str,
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> list[str]:
        if len(texts) != len(metadatas):
            raise ValueError("文本数量与元数据数量不一致")
        store = self.get_or_create_collection(kb_name)
        doc_ids = [metadata.get("doc_id") or str(uuid.uuid4()) for metadata in metadatas]
        for metadata, doc_id in zip(metadatas, doc_ids):
            metadata["doc_id"] = doc_id
        store.add_texts(texts=texts, metadatas=metadatas, ids=doc_ids)
        return doc_ids

    def delete_documents(self, kb_name: str, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        store = self.get_or_create_collection(kb_name)
        store.delete(ids=doc_ids)

    def delete_file_documents(self, kb_name: str, file_id: str) -> int:
        store = self.get_or_create_collection(kb_name)
        data = store.get(where={"file_id": file_id}, include=["metadatas"])
        ids = data.get("ids", [])
        if ids:
            store.delete(ids=ids)
        return len(ids)

    def get_file_upload_paths(self, kb_name: str) -> list[str]:
        store = self.get_or_create_collection(kb_name)
        data = store.get(include=["metadatas"])
        paths: set[str] = set()
        for metadata in data.get("metadatas", []):
            if not metadata:
                continue
            upload_path = metadata.get("upload_path")
            if upload_path:
                paths.add(str(upload_path))
        return sorted(paths)

    def search(self, kb_name: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        store = self.get_or_create_collection(kb_name)
        docs = store.similarity_search_with_score(query, k=k)
        results: list[dict[str, Any]] = []
        for doc, score in docs:
            metadata = dict(doc.metadata or {})
            results.append(
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": metadata,
                }
            )
        return results

    def get_files(self, kb_name: str) -> list[dict[str, Any]]:
        store = self.get_or_create_collection(kb_name)
        data = store.get(include=["metadatas"])
        unique: dict[str, dict[str, Any]] = {}
        for metadata in data.get("metadatas", []):
            if not metadata:
                continue
            file_id = metadata.get("file_id")
            if not file_id or file_id in unique:
                continue
            unique[file_id] = {
                "file_id": file_id,
                "file_name": metadata.get("file_name", ""),
                "file_type": metadata.get("file_type", ""),
                "upload_path": metadata.get("upload_path", ""),
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
            }
        return sorted(unique.values(), key=lambda item: item["file_name"])
