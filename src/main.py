import base64
import hashlib
import json
import logging
import mimetypes
import re
import uuid
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from openai import OpenAI, RateLimitError
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from src.config import settings


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class DocumentRecord(BaseModel):
    source_path: str
    doc_id: str
    content_hash: str
    index_signature: str | None = None
    status: str
    chunk_count: int = 0
    extracted_text_path: str | None = None
    indexed_at: str | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)
    limit: int | None = Field(default=None, ge=1, le=20)


class SyncResponse(BaseModel):
    indexed: list[DocumentRecord]
    skipped: list[str]
    failed: list[DocumentRecord]
    aborted_reason: str | None = None


class ManifestStore:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.manifest_path.write_text("{}", encoding="utf-8")

    def load(self) -> dict[str, DocumentRecord]:
        raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {key: DocumentRecord.model_validate(value) for key, value in raw.items()}

    def save(self, records: dict[str, DocumentRecord]) -> None:
        payload = {key: record.model_dump(mode="json") for key, record in records.items()}
        self.manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def upsert(self, record: DocumentRecord) -> None:
        records = self.load()
        records[record.source_path] = record
        self.save(records)

    def list_records(self) -> list[DocumentRecord]:
        records = self.load()
        return sorted(records.values(), key=lambda item: item.source_path)


class VisionParser:
    OCR_PROMPT = (
        "Извлеки текст с изображения документа. "
        "Сохрани порядок чтения, абзацы и важные числовые значения. "
        "Не добавляй комментарии, пояснения или markdown. "
        "Если текст неразборчив, пропускай только нечитаемые фрагменты."
    )

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def extract_text(self, image_path: Path) -> str:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.OCR_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                        },
                    ],
                }
            ],
        )
        text = response.choices[0].message.content or ""
        return text.strip()


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.vector_size = self.model.get_sentence_embedding_dimension()
        if self.vector_size is None:
            raise RuntimeError("Не удалось определить размерность эмбеддингов.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prepared_texts = [self._prepare_text(text, is_query=False) for text in texts]
        vectors = self.model.encode(
            prepared_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prepared_text = self._prepare_text(text, is_query=True)
        vector = self.model.encode(
            [prepared_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    def _prepare_text(self, text: str, is_query: bool) -> str:
        normalized = text.strip()
        if "multilingual-e5" in self.model_name:
            prefix = "query: " if is_query else "passage: "
            return f"{prefix}{normalized}"
        return normalized


class QdrantIndex:
    def __init__(self, url: str, api_key: str | None, collection_name: str, vector_size: int) -> None:
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key, check_compatibility=False)
        self._ensure_collection(vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        collections = self.client.get_collections().collections
        exists = any(collection.name == self.collection_name for collection in collections)
        if exists:
            collection_info = self.client.get_collection(self.collection_name)
            configured_size = collection_info.config.params.vectors.size
            if configured_size == vector_size:
                return
            logger.warning(
                "Размерность коллекции %s не совпадает с новой embedding моделью: %s != %s. Коллекция будет пересоздана.",
                self.collection_name,
                configured_size,
                vector_size,
            )
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=vector_size,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

    def replace_document(self, doc_id: str, points: list[qdrant_models.PointStruct]) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="doc_id",
                            match=qdrant_models.MatchValue(value=doc_id),
                        )
                    ]
                )
            ),
        )
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, vector: list[float], limit: int) -> list[qdrant_models.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            with_payload=True,
        )


class RAGService:
    def __init__(self) -> None:
        processed_dir = Path(settings.processed_data_dir)
        processed_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = Path(settings.raw_data_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = ManifestStore(processed_dir / "manifest.json")
        self.parser = VisionParser(
            base_url=settings.vlm_base_url,
            api_key=settings.vlm_api_key,
            model=settings.vlm_model,
        )
        self.embedder = EmbeddingService(settings.embedding_model)
        self.index = QdrantIndex(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection,
            vector_size=self.embedder.vector_size,
        )
        self.answer_client = self._build_answer_client()
        self.index_signature = self._build_index_signature()

    def _build_answer_client(self) -> tuple[OpenAI, str] | None:
        if not settings.answer_model:
            return None
        base_url = settings.answer_base_url or settings.vlm_base_url
        api_key = settings.answer_api_key or settings.vlm_api_key
        return OpenAI(base_url=base_url, api_key=api_key), settings.answer_model

    def sync(self) -> SyncResponse:
        indexed: list[DocumentRecord] = []
        skipped: list[str] = []
        failed: list[DocumentRecord] = []
        aborted_reason: str | None = None
        known_records = self.manifest.load()

        for image_path in sorted(self._iter_images()):
            relative_path = image_path.relative_to(self.raw_dir).as_posix()
            content_hash = self._sha256(image_path)
            existing = known_records.get(relative_path)
            if (
                existing
                and existing.content_hash == content_hash
                and existing.status == "indexed"
                and existing.index_signature == self.index_signature
            ):
                skipped.append(relative_path)
                continue

            try:
                record = self._index_image(image_path, relative_path, content_hash, existing)
                indexed.append(record)
                known_records[relative_path] = record
                self.manifest.upsert(record)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Не удалось обработать %s", relative_path)
                failed_record = DocumentRecord(
                    source_path=relative_path,
                    doc_id=self._doc_id(relative_path),
                    content_hash=content_hash,
                    status="failed",
                    error=str(exc),
                )
                failed.append(failed_record)
                known_records[relative_path] = failed_record
                self.manifest.upsert(failed_record)
                if self._is_quota_error(exc):
                    aborted_reason = (
                        "Индексация остановлена: VLM API вернул insufficient_quota. "
                        "Проверьте биллинг или замените провайдера OCR/VLM."
                    )
                    break

        self.manifest.save(known_records)
        return SyncResponse(
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            aborted_reason=aborted_reason,
        )

    def list_documents(self) -> list[DocumentRecord]:
        return self.manifest.list_records()

    def save_uploads(self, files: list[UploadFile]) -> list[str]:
        saved_files: list[str] = []
        for upload in files:
            if not upload.filename:
                continue
            destination = self.raw_dir / Path(upload.filename).name
            if destination.suffix.lower() not in IMAGE_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат: {upload.filename}")
            destination.write_bytes(upload.file.read())
            saved_files.append(destination.relative_to(self.raw_dir).as_posix())
        return saved_files

    def answer(self, question: str, limit: int | None) -> dict[str, Any]:
        effective_limit = limit or settings.search_limit
        candidate_limit = max(effective_limit, settings.retrieval_candidate_limit)
        query_vector = self.embedder.embed_query(question)
        hits = self.index.search(query_vector, candidate_limit)
        contexts = [
            {
                "score": hit.score,
                "source_path": hit.payload.get("source_path"),
                "group_name": hit.payload.get("group_name"),
                "chunk_index": hit.payload.get("chunk_index"),
                "text": hit.payload.get("text"),
            }
            for hit in hits
        ]
        reranked_contexts = self._rerank_contexts(question, contexts)
        selected_contexts = reranked_contexts[:effective_limit]
        response: dict[str, Any] = {
            "question": question,
            "matches": selected_contexts,
            "sources": self._collect_sources(selected_contexts),
            "answer": None,
            "answer_generated": False,
        }
        if not selected_contexts or not self.answer_client:
            return response

        client, model = self.answer_client
        prompt_context = "\n\n".join(
            f"Источник: {item['source_path']}\nФрагмент:\n{item['text']}" for item in selected_contexts
        )
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты внутренний помощник компании. Отвечай только на основе переданного контекста. "
                        "Если данных недостаточно, прямо скажи об этом. "
                        "В конце ответа добавь строку 'Источники:' и перечисли source_path, на которые опирался."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nКонтекст:\n{prompt_context}",
                },
            ],
        )
        response["answer"] = completion.choices[0].message.content
        response["answer_generated"] = True
        return response

    def _rerank_contexts(self, question: str, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        for item in contexts:
            rerank_score = self._hybrid_score(question, item.get("text") or "", float(item.get("score") or 0.0))
            enriched = dict(item)
            enriched["rerank_score"] = rerank_score
            ranked.append(enriched)
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked

    def _hybrid_score(self, question: str, text: str, dense_score: float) -> float:
        normalized_question = self._normalize_text(question)
        normalized_text = self._normalize_text(text)

        query_tokens = self._tokenize(normalized_question)
        text_tokens = self._tokenize(normalized_text)
        overlap_score = self._token_overlap_score(query_tokens, text_tokens)
        trigram_score = self._char_ngram_similarity(normalized_question, normalized_text, n=3)
        phrase_bonus = 0.0
        if normalized_question and normalized_question in normalized_text:
            phrase_bonus = 0.25

        return dense_score * 0.55 + overlap_score * 0.25 + trigram_score * 0.20 + phrase_bonus

    @staticmethod
    def _normalize_text(text: str) -> str:
        lowered = text.lower().replace("ё", "е")
        return re.sub(r"\s+", " ", lowered).strip()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zа-я0-9]+", text)

    @staticmethod
    def _token_overlap_score(query_tokens: list[str], text_tokens: list[str]) -> float:
        if not query_tokens or not text_tokens:
            return 0.0
        query_counter = Counter(query_tokens)
        text_counter = Counter(text_tokens)
        shared = sum(min(text_counter[token], query_counter[token]) for token in query_counter)
        return shared / max(len(query_tokens), 1)

    @staticmethod
    def _char_ngram_similarity(left: str, right: str, n: int) -> float:
        if len(left) < n or len(right) < n:
            return 0.0
        left_ngrams = {left[index : index + n] for index in range(len(left) - n + 1)}
        right_ngrams = {right[index : index + n] for index in range(len(right) - n + 1)}
        if not left_ngrams or not right_ngrams:
            return 0.0
        intersection = len(left_ngrams & right_ngrams)
        union = len(left_ngrams | right_ngrams)
        return intersection / union

    @staticmethod
    def _collect_sources(contexts: list[dict[str, Any]]) -> list[str]:
        seen: dict[str, None] = {}
        for item in contexts:
            source_path = item.get("source_path")
            if source_path:
                seen[source_path] = None
        return list(seen.keys())

    def _iter_images(self) -> list[Path]:
        return [path for path in self.raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]

    def _index_image(
        self,
        image_path: Path,
        relative_path: str,
        content_hash: str,
        existing: DocumentRecord | None,
    ) -> DocumentRecord:
        text = self._load_cached_text(existing)
        if text is None:
            text = self.parser.extract_text(image_path)
        if not text:
            raise RuntimeError("VLM не вернула текст для изображения.")

        output_path = self._processed_text_path(relative_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

        doc_id = self._doc_id(relative_path)
        chunks = self._chunk_text(text)
        if not chunks:
            raise RuntimeError("После чанкинга документ оказался пустым.")

        vectors = self.embedder.embed_documents(chunks)
        group_name = relative_path.split("_page-")[0] if "_page-" in relative_path else Path(relative_path).stem
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}:{index}"))
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "doc_id": doc_id,
                        "source_path": relative_path,
                        "group_name": group_name,
                        "chunk_index": index,
                        "text": chunk,
                    },
                )
            )

        self.index.replace_document(doc_id=doc_id, points=points)
        return DocumentRecord(
            source_path=relative_path,
            doc_id=doc_id,
            content_hash=content_hash,
            index_signature=self.index_signature,
            status="indexed",
            chunk_count=len(chunks),
            extracted_text_path=output_path.as_posix(),
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _chunk_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        words = normalized.split(" ")
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        if overlap >= chunk_size:
            raise RuntimeError("chunk_overlap должен быть меньше chunk_size.")

        chunks: list[str] = []
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            piece = words[start : start + chunk_size]
            if piece:
                chunks.append(" ".join(piece))
            if start + chunk_size >= len(words):
                break
        return chunks

    def _processed_text_path(self, relative_path: str) -> Path:
        target = Path(settings.processed_data_dir) / Path(relative_path)
        return target.with_suffix(".txt")

    @staticmethod
    def _load_cached_text(existing: DocumentRecord | None) -> str | None:
        if existing is None or not existing.extracted_text_path:
            return None
        text_path = Path(existing.extracted_text_path)
        if not text_path.exists():
            return None
        return text_path.read_text(encoding="utf-8").strip() or None

    @staticmethod
    def _sha256(file_path: Path) -> str:
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    @staticmethod
    def _doc_id(relative_path: str) -> str:
        return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()

    def _build_index_signature(self) -> str:
        payload = {
            "embedding_model": settings.embedding_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_quota_error(exc: Exception) -> bool:
        return isinstance(exc, RateLimitError) and "insufficient_quota" in str(exc)


rag_service: RAGService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global rag_service
    rag_service = RAGService()
    yield


app = FastAPI(
    title="Internal Document RAG",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentRecord])
def list_documents() -> list[DocumentRecord]:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
    return rag_service.list_documents()


@app.post("/documents/upload")
def upload_documents(files: list[UploadFile] = File(...)) -> dict[str, Any]:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
    saved_files = rag_service.save_uploads(files)
    sync_result = rag_service.sync()
    return {
        "saved_files": saved_files,
        "indexed_count": len(sync_result.indexed),
        "skipped_count": len(sync_result.skipped),
        "failed": [item.model_dump(mode="json") for item in sync_result.failed],
    }


@app.post("/documents/sync", response_model=SyncResponse)
def sync_documents() -> SyncResponse:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
    return rag_service.sync()


@app.post("/query")
def query_documents(request: QueryRequest) -> dict[str, Any]:
    if rag_service is None:
        raise HTTPException(status_code=503, detail="Сервис еще не инициализирован")
    return rag_service.answer(request.question, request.limit)