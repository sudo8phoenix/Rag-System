"""End-to-end orchestration across parsing, chunking, retrieval, LLM, and TTS."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.chunkers.registry import ChunkingRegistry
from src.config.settings import AppConfig
from src.embeddings.orchestrator import EmbeddingOrchestrator
from src.llm.base import LLMResponse
from src.llm.groq_wrapper import GroqLLM
from src.llm.ollama_wrapper import OllamaLLM
from src.models.chunk import Chunk
from src.models.document import Document
from src.parsers.registry import ParserRegistry
from src.tts.orchestrator import TTSOrchestrator
from src.voice.voice_input import VoiceInput, VoiceInputResult


@dataclass(frozen=True)
class IngestionResult:
    """Result of parsing, chunking, and indexing one or more source documents."""

    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineResult:
    """Structured result for a pipeline run."""

    query: str
    response_text: str = ""
    audio_path: Path | None = None
    audio_played: bool = False
    transcribed_text: str | None = None
    voice_confidence: float | None = None
    speech_detected: bool = False
    documents: list[dict[str, Any]] = field(default_factory=list)
    chunks: list[dict[str, Any]] = field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    ingest_errors: list[dict[str, str]] = field(default_factory=list)
    llm_model: str | None = None
    success: bool = True
    error_stage: str | None = None
    error_message: str | None = None


class PipelineOrchestrator:
    """High-level orchestration for the Phase 1 RAG pipeline."""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        parser_registry: ParserRegistry | None = None,
        chunking_registry: ChunkingRegistry | None = None,
        embedding_orchestrator: EmbeddingOrchestrator | None = None,
        llm: GroqLLM | None = None,
        tts: TTSOrchestrator | None = None,
        voice_input: VoiceInput | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.parser_registry = parser_registry or ParserRegistry()
        self.chunking_registry = chunking_registry or ChunkingRegistry(self.config)
        self.embedding_orchestrator = embedding_orchestrator or EmbeddingOrchestrator.from_config(
            self.config
        )
        self.llm = llm or self._build_llm_from_config(self.config)
        self.tts = tts or TTSOrchestrator.from_app_config(self.config)
        self.voice_input = voice_input or VoiceInput()
        self.logger = logger or logging.getLogger(__name__)

    def _build_llm_from_config(self, config: AppConfig):
        provider = config.llm.provider
        if provider == "groq":
            return GroqLLM.from_app_config(config)
        if provider == "ollama":
            return OllamaLLM.from_app_config(config)

        raise ValueError(f"Unsupported LLM provider: {provider}")

    @classmethod
    def from_config(cls, config: AppConfig) -> "PipelineOrchestrator":
        """Build an orchestrator from application config."""

        return cls(config=config)

    def _log_event(self, level: int, stage: str, message: str, **payload: Any) -> None:
        record = {
            "component": "pipeline",
            "stage": stage,
            "message": message,
            **payload,
        }
        self.logger.log(level, json.dumps(record, default=str, sort_keys=True))

    def _iter_source_paths(self, source_paths: Sequence[str | Path] | None) -> Iterable[Path]:
        if not source_paths:
            return []
        return (Path(source_path) for source_path in source_paths)

    def ingest_documents(self, source_paths: Sequence[str | Path]) -> IngestionResult:
        """Parse, chunk, and index a batch of source documents."""

        documents: list[Document] = []
        chunks: list[Chunk] = []
        errors: list[dict[str, str]] = []

        for source_path in self._iter_source_paths(source_paths):
            try:
                document = self.parser_registry.parse_file(source_path)
                document_chunks = self.chunking_registry.chunk_document(document)
                if document_chunks:
                    self.embedding_orchestrator.index_chunks(document_chunks)
                documents.append(document)
                chunks.extend(document_chunks)
                self._log_event(
                    logging.INFO,
                    "ingest",
                    "document ingested",
                    path=str(source_path),
                    chunks=len(document_chunks),
                )
            except Exception as exc:  # pragma: no cover - exercised through pipeline tests
                error_info = {
                    "path": str(source_path),
                    "error": str(exc),
                    "type": type(exc).__name__,
                }
                errors.append(error_info)
                self._log_event(
                    logging.WARNING,
                    "ingest",
                    "document ingestion failed",
                    **error_info,
                )

        return IngestionResult(documents=documents, chunks=chunks, errors=errors)

    def _resolve_query(
        self,
        query: str | None,
        *,
        use_voice: bool,
        voice_duration_seconds: float,
        language: str | None,
    ) -> tuple[str, VoiceInputResult | None]:
        if query is not None and query.strip():
            return query.strip(), None

        if not use_voice and self.config.voice.input_mode != "voice":
            raise ValueError("query must be provided when voice input is disabled")

        voice_result = self.voice_input.capture_and_transcribe(
            duration_seconds=voice_duration_seconds,
            push_to_talk=True,
            language=language or self.config.voice.language,
        )
        if not voice_result.text.strip():
            raise ValueError("voice input did not produce a transcribed query")
        return voice_result.text.strip(), voice_result

    def _build_context_items(self, retrieved_chunks: Sequence[Any]) -> list[dict[str, Any]]:
        context_items: list[dict[str, Any]] = []
        for result in retrieved_chunks:
            chunk = getattr(result, "chunk", None)
            if chunk is None:
                continue

            context_items.append(
                {
                    "text": chunk.text,
                    "source": chunk.source_doc.filename,
                    "source_type": chunk.source_doc.source_type,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "score": getattr(result, "score", None),
                    "distance": getattr(result, "distance", None),
                    "metadata": dict(getattr(result, "metadata", {})),
                }
            )
        return context_items

    def _serialize_documents(self, documents: Sequence[Document]) -> list[dict[str, Any]]:
        return [document.to_dict() for document in documents]

    def _serialize_chunks(self, chunks: Sequence[Chunk]) -> list[dict[str, Any]]:
        return [chunk.to_dict() for chunk in chunks]

    def answer(
        self,
        query: str | None = None,
        *,
        source_paths: Sequence[str | Path] | None = None,
        use_voice: bool = False,
        voice_duration_seconds: float = 5.0,
        voice_language: str | None = None,
        top_k: int | None = None,
        system_prompt: str | None = None,
        output_path: str | Path | None = None,
        block: bool = True,
        ingest_sources: bool = True,
        raise_on_error: bool = False,
    ) -> PipelineResult:
        """Run the full Phase 1 pipeline and return a structured result."""

        try:
            resolved_query, voice_result = self._resolve_query(
                query,
                use_voice=use_voice,
                voice_duration_seconds=voice_duration_seconds,
                language=voice_language,
            )
        except Exception as exc:
            self._log_event(
                logging.ERROR,
                "voice",
                "query resolution failed",
                error=str(exc),
                type=type(exc).__name__,
            )
            if raise_on_error:
                raise
            return PipelineResult(
                query=query.strip() if isinstance(query, str) and query.strip() else "",
                success=False,
                error_stage="voice",
                error_message=str(exc),
            )

        ingestion = IngestionResult()
        if ingest_sources and source_paths:
            ingestion = self.ingest_documents(source_paths)

        effective_top_k = top_k if top_k is not None else self.config.retrieval.top_k

        try:
            retrieved_chunks = self.embedding_orchestrator.search(
                resolved_query,
                top_k=effective_top_k,
            )
            self._log_event(
                logging.INFO,
                "retrieval",
                "retrieval completed",
                query=resolved_query,
                results=len(retrieved_chunks),
                top_k=effective_top_k,
            )
        except Exception as exc:
            self._log_event(
                logging.ERROR,
                "retrieval",
                "retrieval failed",
                query=resolved_query,
                error=str(exc),
                type=type(exc).__name__,
            )
            if raise_on_error:
                raise
            return PipelineResult(
                query=resolved_query,
                transcribed_text=voice_result.text if voice_result else None,
                voice_confidence=voice_result.confidence if voice_result else None,
                speech_detected=bool(voice_result and voice_result.speech_detected),
                documents=self._serialize_documents(ingestion.documents),
                chunks=self._serialize_chunks(ingestion.chunks),
                ingest_errors=list(ingestion.errors),
                success=False,
                error_stage="retrieval",
                error_message=str(exc),
            )

        context_items = self._build_context_items(retrieved_chunks)

        try:
            llm_response: LLMResponse = self.llm.generate(
                resolved_query,
                context_items=context_items,
                system_prompt=system_prompt,
            )
            self._log_event(
                logging.INFO,
                "llm",
                "response generated",
                model=llm_response.model,
                response_length=len(llm_response.text),
            )
        except Exception as exc:
            self._log_event(
                logging.ERROR,
                "llm",
                "response generation failed",
                query=resolved_query,
                error=str(exc),
                type=type(exc).__name__,
            )
            if raise_on_error:
                raise
            return PipelineResult(
                query=resolved_query,
                transcribed_text=voice_result.text if voice_result else None,
                voice_confidence=voice_result.confidence if voice_result else None,
                speech_detected=bool(voice_result and voice_result.speech_detected),
                documents=self._serialize_documents(ingestion.documents),
                chunks=self._serialize_chunks(ingestion.chunks),
                retrieved_chunks=[result.to_dict() for result in retrieved_chunks],
                ingest_errors=list(ingestion.errors),
                success=False,
                error_stage="llm",
                error_message=str(exc),
            )

        audio_path: Path | None = None
        audio_played = False
        try:
            tts_result = self.tts.speak(llm_response.text, output_path=output_path, block=block)
            audio_path = tts_result.audio_path
            audio_played = tts_result.played
            self._log_event(
                logging.INFO,
                "tts",
                "audio synthesized",
                audio_path=str(audio_path),
                played=audio_played,
            )
        except Exception as exc:
            self._log_event(
                logging.WARNING,
                "tts",
                "audio synthesis skipped",
                query=resolved_query,
                error=str(exc),
                type=type(exc).__name__,
            )

        return PipelineResult(
            query=resolved_query,
            response_text=llm_response.text,
            audio_path=audio_path,
            audio_played=audio_played,
            transcribed_text=voice_result.text if voice_result else None,
            voice_confidence=voice_result.confidence if voice_result else None,
            speech_detected=bool(voice_result and voice_result.speech_detected),
            documents=self._serialize_documents(ingestion.documents),
            chunks=self._serialize_chunks(ingestion.chunks),
            retrieved_chunks=[result.to_dict() for result in retrieved_chunks],
            ingest_errors=list(ingestion.errors),
            llm_model=llm_response.model,
        )
