"""Gradio UI for the Phase 1 RAG pipeline."""

from __future__ import annotations

import html
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.loader import load_config
from src.config.settings import AppConfig
from src.pipeline import PipelineOrchestrator
from src.tts.orchestrator import TTSOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_PROFILE_DIR = PROJECT_ROOT / "config" / "profiles"

STT_ENGINE_CHOICES = ["faster-whisper", "whisper", "speechrecognition"]
WHISPER_MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]
CHUNK_STRATEGY_CHOICES = [
    "line",
    "character",
    "paragraph",
    "heading_hierarchy",
    "row_based",
    "array_item",
    "slide_based",
    "tag_based",
    "chapter_based",
    "semantic",
    "token",
]
VECTOR_STORE_CHOICES = ["faiss", "chroma", "qdrant"]
TTS_ENGINE_CHOICES = list(TTSOrchestrator.SUPPORTED_ENGINES)

INPUT_MODE_CHOICES = ["voice", "text"]
CHUNK_UNIT_CHOICES = ["characters", "tokens"]
LLM_PROVIDER_CHOICES = ["groq", "ollama", "openai", "anthropic", "litellm"]

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg0: #f4efe7;
    --bg1: #efe2d1;
    --ink: #1d1713;
    --ink-soft: #4a3a31;
    --accent: #0f766e;
    --accent-2: #c2410c;
    --panel: #fff8ef;
    --line: #e3d7c8;
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif;
    background:
        radial-gradient(circle at 15% 10%, rgba(194, 65, 12, 0.18), transparent 28%),
        radial-gradient(circle at 85% 5%, rgba(15, 118, 110, 0.16), transparent 25%),
        linear-gradient(145deg, var(--bg0), var(--bg1));
    color: var(--ink);
}

.block-title h1 {
    letter-spacing: 0.02em;
    font-weight: 700;
}

.panel {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: 0 16px 40px rgba(29, 23, 19, 0.08);
}

.gr-button-primary {
    background: linear-gradient(120deg, var(--accent), #115e59) !important;
    color: #f9fafb !important;
    border: none !important;
}

.gr-button-secondary {
    background: linear-gradient(120deg, var(--accent-2), #9a3412) !important;
    color: #fff7ed !important;
    border: none !important;
}

.mono textarea, .mono input {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--ink-soft) !important;
}
"""


def _as_int(value: float | int) -> int:
    return int(value)


def _parse_csv_tokens(raw_value: str) -> list[str]:
    if not raw_value.strip():
        return []
    normalized = raw_value.replace("\n", ",")
    return [token.strip() for token in normalized.split(",") if token.strip()]


def _parse_int_tokens(raw_value: str) -> list[int]:
    values: list[int] = []
    for token in _parse_csv_tokens(raw_value):
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def _parse_source_paths(raw_paths: str) -> list[str]:
    if not raw_paths.strip():
        return []

    parsed_paths: list[str] = []
    for raw_line in raw_paths.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Preserve existing filesystem paths even when filenames contain commas.
        if Path(line).expanduser().exists():
            parsed_paths.append(line)
            continue

        if "," in line:
            parsed_paths.extend(token.strip() for token in line.split(",") if token.strip())
            continue

        parsed_paths.append(line)

    return parsed_paths


def _normalize_uploaded_files(uploaded_files: Any) -> list[str]:
    if not uploaded_files:
        return []

    if isinstance(uploaded_files, (str, Path)):
        return [str(uploaded_files)]

    normalized_paths: list[str] = []
    for uploaded_file in uploaded_files:
        if isinstance(uploaded_file, (str, Path)):
            normalized_paths.append(str(uploaded_file))
            continue

        file_path = getattr(uploaded_file, "name", None)
        if file_path:
            normalized_paths.append(str(file_path))

    return normalized_paths


def list_available_profiles(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
) -> list[str]:
    """Return profile choices for UI dropdowns."""

    profiles = [f"default:{Path(config_path).name}"]
    directory = Path(profile_dir)
    if not directory.exists():
        return profiles

    for profile_file in sorted(directory.glob("*.yaml")):
        profiles.append(profile_file.stem)
    return profiles


def load_profile_config(
    profile_name: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    profile_dir: str | Path = DEFAULT_PROFILE_DIR,
) -> AppConfig:
    """Load a selected profile, falling back to default config path."""

    if profile_name.startswith("default:"):
        return load_config(config_path)

    candidate = Path(profile_dir) / f"{profile_name}.yaml"
    return load_config(candidate)


def summarize_validation_errors(errors: list[str]) -> str:
    """Build a concise validation summary grouped by config section."""

    if not errors:
        return "Validation passed."

    grouped: dict[str, int] = {}
    for error in errors:
        section = error.split(":", 1)[0].split(".", 1)[0].strip() or "general"
        grouped[section] = grouped.get(section, 0) + 1

    summary_parts = [f"{section}={count}" for section, count in sorted(grouped.items())]
    details = "\n- " + "\n- ".join(errors[:8])
    if len(errors) > 8:
        details += f"\n- ... {len(errors) - 8} more"
    return (
        f"Validation failed ({len(errors)} issue(s)): "
        + ", ".join(summary_parts)
        + details
    )


def config_snapshot(values: dict[str, Any]) -> dict[str, Any]:
    """Normalize settings payload for dirty-state comparisons."""

    snapshot = dict(values)
    for key in (
        "chunk_size",
        "chunk_overlap",
        "retrieval_top_k",
        "llm_max_tokens",
        "ui_port",
    ):
        if key in snapshot and snapshot[key] is not None:
            snapshot[key] = int(snapshot[key])

    for key in (
        "llm_temperature",
        "tts_rate",
        "tts_volume",
        "voice_vad_threshold",
        "retrieval_bm25_weight",
    ):
        if key in snapshot and snapshot[key] is not None:
            snapshot[key] = float(snapshot[key])

    for key in ("heading_levels", "target_tags"):
        if key in snapshot and isinstance(snapshot[key], str):
            snapshot[key] = ", ".join(_parse_csv_tokens(snapshot[key]))

    for key in ("retrieval_rerank_min_score",):
        if key in snapshot and snapshot[key] not in (None, ""):
            snapshot[key] = float(snapshot[key])
        elif key in snapshot:
            snapshot[key] = None

    return snapshot


def is_dirty_config(
    current_values: dict[str, Any], baseline_values: dict[str, Any]
) -> bool:
    """Return True when current values differ from baseline values."""

    return config_snapshot(current_values) != config_snapshot(baseline_values)


def _normalize_score(value: Any, distance: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(distance, (int, float)):
        return max(0.0, min(1.0, 1.0 / (1.0 + float(distance))))
    return 0.0


def render_retrieved_chunks_html(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Render retrieved chunks with metadata and confidence coloring."""

    if not retrieved_chunks:
        return (
            "<p>No retrieved chunks yet. Run a query to inspect retrieval evidence.</p>"
        )

    cards: list[str] = []
    for index, item in enumerate(retrieved_chunks, start=1):
        chunk = item.get("chunk", {})
        source_doc = chunk.get("source_doc", {})
        source = source_doc.get("filename", "unknown")
        strategy = chunk.get("strategy_used", "unknown")
        score = item.get("score")
        distance = item.get("distance")
        normalized = _normalize_score(score, distance)
        hue = int(8 + (normalized * 112))
        border_color = f"hsl({hue}, 70%, 42%)"

        metadata = chunk.get("metadata", {})
        metadata_items = "".join(
            f"<li><strong>{html.escape(str(key))}</strong>: {html.escape(str(value))}</li>"
            for key, value in sorted(metadata.items())
        )
        if not metadata_items:
            metadata_items = "<li>No metadata</li>"

        preview = html.escape(str(chunk.get("text", "")))
        preview = preview if len(preview) <= 800 else preview[:800] + "..."

        cards.append(
            """
            <details style="margin: 0 0 10px 0; border: 1px solid {border}; border-radius: 12px; background: #fffdf8;">
              <summary style="cursor: pointer; padding: 10px 12px; font-weight: 600;">
                #{idx} | {source} | strategy={strategy} | score={score_repr}
              </summary>
              <div style="padding: 8px 12px 12px 12px;">
                <p style="margin: 0 0 8px 0;"><strong>chunk_id:</strong> {chunk_id}</p>
                <pre style="white-space: pre-wrap; font-family: 'IBM Plex Mono', monospace; font-size: 0.86rem; margin: 0 0 10px 0;">{preview}</pre>
                <ul style="margin: 0; padding-left: 20px;">{metadata_items}</ul>
              </div>
            </details>
            """.format(
                border=border_color,
                idx=index,
                source=html.escape(str(source)),
                strategy=html.escape(str(strategy)),
                score_repr=html.escape(str(round(normalized, 3))),
                chunk_id=html.escape(str(chunk.get("chunk_id", "unknown"))),
                preview=preview,
                metadata_items=metadata_items,
            )
        )

    return "\n".join(cards)


def render_execution_trace(
    *,
    query: str,
    use_voice: bool,
    source_paths: list[str],
    top_k: int,
    response_text: str,
    transcribed_text: str,
    retrieved_chunks: list[dict[str, Any]],
    ingest_errors: list[dict[str, str]],
    tts_error: str | None,
) -> str:
    """Build a collapsible-ready trace summary for ingest and retrieval flows."""

    mode = "voice" if use_voice else "text"
    line_items = [
        f"- mode: {mode}",
        f"- query: {query}",
        f"- transcribed_query: {transcribed_text or 'n/a'}",
        f"- source_paths: {len(source_paths)}",
        f"- top_k: {top_k}",
        f"- retrieved_chunks: {len(retrieved_chunks)}",
        f"- ingest_errors: {len(ingest_errors)}",
        f"- tts_status: {'warning' if tts_error else 'ok'}",
        f"- response_chars: {len(response_text)}",
    ]

    if ingest_errors:
        line_items.append("- ingest_error_details:")
        line_items.extend(
            [
                f"  - {error.get('path', 'unknown')}: {error.get('error', 'unknown error')}"
                for error in ingest_errors[:5]
            ]
        )

    return "\n".join(line_items)


def append_history_entry(
    history: list[dict[str, str]],
    *,
    query: str,
    transcribed_query: str,
    response: str,
    audio_path: str | None,
    status: str,
) -> list[dict[str, str]]:
    """Append a conversation turn for the UI history panel."""

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "transcribed_query": transcribed_query,
        "response": response,
        "audio_path": audio_path or "",
        "status": status,
    }
    return [*history, entry]


def history_rows(history: list[dict[str, str]]) -> list[list[str]]:
    """Return tabular rows for history display."""

    rows: list[list[str]] = []
    for index, entry in enumerate(history, start=1):
        query_preview = entry.get("query", "")[:70]
        response_preview = entry.get("response", "")[:90]
        rows.append(
            [
                str(index),
                entry.get("timestamp", ""),
                query_preview,
                response_preview,
                "yes" if entry.get("audio_path") else "no",
            ]
        )
    return rows


def format_ingest_result_summary(results: list[dict[str, Any]]) -> str:
    """Build summary text for upload-triggered ingest."""

    if not results:
        return "No files ingested yet."

    successes = sum(1 for item in results if item.get("status") == "success")
    failures = len(results) - successes
    total_chunks = sum(int(item.get("chunks", 0)) for item in results)
    lines = [
        f"Processed files: {len(results)} | success={successes} | failures={failures} | chunks={total_chunks}"
    ]
    for item in results:
        lines.append(
            f"- {item.get('name', 'unknown')}: {item.get('status', 'unknown')} (chunks={item.get('chunks', 0)})"
        )
        if item.get("error"):
            lines.append(f"  error: {item['error']}")
    return "\n".join(lines)


def config_to_ui_defaults(config: AppConfig) -> dict[str, Any]:
    """Map validated app config to initial UI field values."""

    return {
        "stt_engine": config.voice.stt_engine,
        "whisper_model": config.voice.whisper_model,
        "voice_input_mode": config.voice.input_mode,
        "voice_language": config.voice.language,
        "voice_vad_enabled": config.voice.vad_enabled,
        "voice_vad_threshold": config.voice.vad_threshold,
        "chunk_strategy": config.chunking.strategy,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "chunk_unit": config.chunking.chunk_unit,
        "chunk_min_size": config.chunking.min_chunk_size,
        "chunk_max_size": config.chunking.max_chunk_size,
        "chunk_respect_sentence_boundaries": config.chunking.respect_sentence_boundaries,
        "chunk_prepend_metadata": config.chunking.prepend_metadata,
        "chunk_heading_levels": ", ".join(
            str(level) for level in config.chunking.heading_levels
        ),
        "chunk_rows_per_chunk": config.chunking.rows_per_chunk,
        "chunk_include_headers": config.chunking.include_headers,
        "chunk_include_notes": config.chunking.include_notes,
        "chunk_target_tags": ", ".join(config.chunking.target_tags),
        "chunk_semantic_similarity_threshold": config.chunking.semantic_similarity_threshold,
        "embedding_model": config.embedding.model,
        "vector_store": config.embedding.vector_store,
        "retrieval_top_k": config.retrieval.top_k,
        "retrieval_hybrid_search": config.retrieval.hybrid_search,
        "retrieval_bm25_weight": config.retrieval.bm25_weight,
        "retrieval_rerank": config.retrieval.rerank,
        "retrieval_rerank_model": config.retrieval.rerank_model,
        "retrieval_rerank_candidate_pool": config.retrieval.rerank_candidate_pool,
        "retrieval_rerank_min_score": config.retrieval.rerank_min_score,
        "llm_provider": config.llm.provider,
        "llm_model": config.llm.model,
        "llm_base_url": config.llm.base_url,
        "llm_temperature": config.llm.temperature,
        "llm_max_tokens": config.llm.max_tokens,
        "llm_system_prompt": config.llm.system_prompt,
        "llm_api_key": config.llm.api_key or "",
        "tts_engine": config.tts.engine,
        "tts_rate": config.tts.rate,
        "tts_volume": config.tts.volume,
        "tts_mute": config.tts.mute,
        "ui_host": config.ui.host,
        "ui_port": config.ui.port,
        "ui_show_sources": config.ui.show_sources,
        "ui_show_agent_trace": config.ui.show_agent_trace,
    }


def config_to_ui_values(config: AppConfig) -> list[Any]:
    """Return UI field values in component order for config tabs."""

    defaults = config_to_ui_defaults(config)
    return [
        defaults["stt_engine"],
        defaults["whisper_model"],
        defaults["chunk_strategy"],
        defaults["chunk_size"],
        defaults["chunk_overlap"],
        defaults["embedding_model"],
        defaults["vector_store"],
        defaults["llm_model"],
        defaults["llm_temperature"],
        defaults["llm_api_key"],
        defaults["tts_engine"],
        defaults["tts_rate"],
        defaults["tts_volume"],
        defaults["tts_mute"],
    ]


def build_config_payload(
    *,
    stt_engine: str,
    whisper_model: str,
    chunk_strategy: str,
    chunk_size: float | int,
    chunk_overlap: float | int,
    embedding_model: str,
    vector_store: str,
    llm_model: str,
    llm_temperature: float,
    llm_api_key: str,
    tts_engine: str,
    tts_rate: float,
    tts_volume: float,
    tts_mute: bool,
    base_config: AppConfig,
    voice_input_mode: str | None = None,
    voice_language: str | None = None,
    voice_vad_enabled: bool | None = None,
    voice_vad_threshold: float | None = None,
    chunk_unit: str | None = None,
    chunk_min_size: float | int | None = None,
    chunk_max_size: float | int | None = None,
    chunk_respect_sentence_boundaries: bool | None = None,
    chunk_prepend_metadata: bool | None = None,
    chunk_heading_levels: str | None = None,
    chunk_rows_per_chunk: float | int | None = None,
    chunk_include_headers: bool | None = None,
    chunk_include_notes: bool | None = None,
    chunk_target_tags: str | None = None,
    chunk_semantic_similarity_threshold: float | None = None,
    retrieval_top_k: float | int | None = None,
    retrieval_hybrid_search: bool | None = None,
    retrieval_bm25_weight: float | None = None,
    retrieval_rerank: bool | None = None,
    retrieval_rerank_model: str | None = None,
    retrieval_rerank_candidate_pool: float | int | None = None,
    retrieval_rerank_min_score: float | None = None,
    llm_provider: str | None = None,
    llm_base_url: str | None = None,
    llm_max_tokens: float | int | None = None,
    llm_system_prompt: str | None = None,
    ui_host: str | None = None,
    ui_port: float | int | None = None,
    ui_show_sources: bool | None = None,
    ui_show_agent_trace: bool | None = None,
) -> dict[str, Any]:
    """Create a YAML payload from UI controls."""

    payload = base_config.model_dump(mode="python")
    payload["voice"]["stt_engine"] = stt_engine
    payload["voice"]["whisper_model"] = whisper_model
    payload["chunking"]["strategy"] = chunk_strategy
    payload["chunking"]["chunk_size"] = _as_int(chunk_size)
    payload["chunking"]["chunk_overlap"] = _as_int(chunk_overlap)
    payload["embedding"]["model"] = embedding_model.strip()
    payload["embedding"]["vector_store"] = vector_store
    payload["llm"]["model"] = llm_model.strip()
    payload["llm"]["temperature"] = float(llm_temperature)
    payload["llm"]["api_key"] = llm_api_key.strip() if llm_api_key else None
    payload["tts"]["engine"] = tts_engine
    payload["tts"]["rate"] = float(tts_rate)
    payload["tts"]["volume"] = float(tts_volume)
    payload["tts"]["mute"] = bool(tts_mute)

    if voice_input_mode is not None:
        payload["voice"]["input_mode"] = voice_input_mode
    if voice_language is not None:
        payload["voice"]["language"] = (
            voice_language.strip() or payload["voice"]["language"]
        )
    if voice_vad_enabled is not None:
        payload["voice"]["vad_enabled"] = bool(voice_vad_enabled)
    if voice_vad_threshold is not None:
        payload["voice"]["vad_threshold"] = float(voice_vad_threshold)

    if chunk_unit is not None:
        payload["chunking"]["chunk_unit"] = chunk_unit
    if chunk_min_size is not None:
        payload["chunking"]["min_chunk_size"] = _as_int(chunk_min_size)
    if chunk_max_size is not None:
        payload["chunking"]["max_chunk_size"] = _as_int(chunk_max_size)
    if chunk_respect_sentence_boundaries is not None:
        payload["chunking"]["respect_sentence_boundaries"] = bool(
            chunk_respect_sentence_boundaries
        )
    if chunk_prepend_metadata is not None:
        payload["chunking"]["prepend_metadata"] = bool(chunk_prepend_metadata)
    if chunk_heading_levels is not None:
        parsed_levels = _parse_int_tokens(chunk_heading_levels)
        payload["chunking"]["heading_levels"] = (
            parsed_levels or payload["chunking"]["heading_levels"]
        )
    if chunk_rows_per_chunk is not None:
        payload["chunking"]["rows_per_chunk"] = _as_int(chunk_rows_per_chunk)
    if chunk_include_headers is not None:
        payload["chunking"]["include_headers"] = bool(chunk_include_headers)
    if chunk_include_notes is not None:
        payload["chunking"]["include_notes"] = bool(chunk_include_notes)
    if chunk_target_tags is not None:
        payload["chunking"]["target_tags"] = _parse_csv_tokens(chunk_target_tags)
    if chunk_semantic_similarity_threshold is not None:
        payload["chunking"]["semantic_similarity_threshold"] = float(
            chunk_semantic_similarity_threshold
        )

    if retrieval_top_k is not None:
        payload["retrieval"]["top_k"] = _as_int(retrieval_top_k)
    if retrieval_hybrid_search is not None:
        payload["retrieval"]["hybrid_search"] = bool(retrieval_hybrid_search)
    if retrieval_bm25_weight is not None:
        payload["retrieval"]["bm25_weight"] = float(retrieval_bm25_weight)
    if retrieval_rerank is not None:
        payload["retrieval"]["rerank"] = bool(retrieval_rerank)
    if retrieval_rerank_model is not None:
        payload["retrieval"]["rerank_model"] = retrieval_rerank_model.strip()
    if retrieval_rerank_candidate_pool is not None:
        payload["retrieval"]["rerank_candidate_pool"] = _as_int(
            retrieval_rerank_candidate_pool
        )
    if retrieval_rerank_min_score is not None:
        payload["retrieval"]["rerank_min_score"] = float(retrieval_rerank_min_score)

    if llm_provider is not None:
        payload["llm"]["provider"] = llm_provider
    if llm_base_url is not None:
        payload["llm"]["base_url"] = llm_base_url.strip()
    if llm_max_tokens is not None:
        payload["llm"]["max_tokens"] = _as_int(llm_max_tokens)
    if llm_system_prompt is not None:
        payload["llm"]["system_prompt"] = llm_system_prompt

    if ui_host is not None:
        payload["ui"]["host"] = ui_host.strip()
    if ui_port is not None:
        payload["ui"]["port"] = _as_int(ui_port)
    if ui_show_sources is not None:
        payload["ui"]["show_sources"] = bool(ui_show_sources)
    if ui_show_agent_trace is not None:
        payload["ui"]["show_agent_trace"] = bool(ui_show_agent_trace)

    return payload


def validate_config_payload(
    payload: dict[str, Any],
) -> tuple[AppConfig | None, list[str]]:
    """Validate payload and return inline-friendly errors."""

    try:
        return AppConfig.model_validate(payload), []
    except ValidationError as exc:
        errors: list[str] = []
        for issue in exc.errors():
            location = ".".join(str(part) for part in issue.get("loc", []))
            errors.append(f"{location}: {issue.get('msg', 'invalid value')}")
        return None, errors


def save_config_to_path(config: AppConfig, config_path: str | Path) -> None:
    """Persist config to YAML."""

    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="python")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def _resolve_server_port(host: str, preferred_port: int, max_attempts: int = 25) -> int:
    for candidate in range(preferred_port, preferred_port + max_attempts):
        if _is_port_available(host, candidate):
            return candidate
    return preferred_port


def create_gradio_app(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    orchestrator: PipelineOrchestrator | None = None,
):
    """Build and return the Gradio Blocks app."""

    try:
        # Workaround for Typer/Click incompatibility: patch Click.Choice to support type subscripting
        import click

        if not hasattr(click.Choice, "__class_getitem__"):
            click.Choice.__class_getitem__ = classmethod(lambda cls, params: cls)

        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "Gradio is required. Install it with `pip install gradio`."
        ) from exc

    config = load_config(config_path)
    defaults = config_to_ui_defaults(config)
    runtime_state: dict[str, Any] = {
        "config": config,
        "config_path": Path(config_path),
        "orchestrator": orchestrator,
        "ingested_paths": set(),
    }

    config_field_order = [
        "stt_engine",
        "whisper_model",
        "voice_input_mode",
        "voice_language",
        "voice_vad_enabled",
        "voice_vad_threshold",
        "chunk_strategy",
        "chunk_size",
        "chunk_overlap",
        "chunk_unit",
        "chunk_min_size",
        "chunk_max_size",
        "chunk_respect_sentence_boundaries",
        "chunk_prepend_metadata",
        "chunk_heading_levels",
        "chunk_rows_per_chunk",
        "chunk_include_headers",
        "chunk_include_notes",
        "chunk_target_tags",
        "chunk_semantic_similarity_threshold",
        "embedding_model",
        "vector_store",
        "retrieval_top_k",
        "retrieval_hybrid_search",
        "retrieval_bm25_weight",
        "retrieval_rerank",
        "retrieval_rerank_model",
        "retrieval_rerank_candidate_pool",
        "retrieval_rerank_min_score",
        "llm_provider",
        "llm_model",
        "llm_base_url",
        "llm_temperature",
        "llm_max_tokens",
        "llm_system_prompt",
        "llm_api_key",
        "tts_engine",
        "tts_rate",
        "tts_volume",
        "tts_mute",
        "ui_host",
        "ui_port",
        "ui_show_sources",
        "ui_show_agent_trace",
    ]

    def _profile_path_label(path: str | Path) -> str:
        return str(Path(path).resolve())

    def _ensure_orchestrator() -> PipelineOrchestrator:
        if runtime_state["orchestrator"] is None:
            runtime_state["orchestrator"] = PipelineOrchestrator.from_config(
                runtime_state["config"]
            )
        return runtime_state["orchestrator"]

    def _form_values_to_dict(*values: Any) -> dict[str, Any]:
        return dict(zip(config_field_order, values))

    def _config_values_from_defaults(default_values: dict[str, Any]) -> list[Any]:
        return [default_values[field] for field in config_field_order]

    def _history_choice_labels(history: list[dict[str, str]]) -> list[str]:
        labels: list[str] = []
        for index, entry in enumerate(history, start=1):
            labels.append(f"Turn {index} | {entry.get('timestamp', '')}")
        return labels

    def _new_source_paths(resolved_source_paths: list[str]) -> list[str]:
        return [
            path
            for path in resolved_source_paths
            if path not in runtime_state["ingested_paths"]
        ]

    def _mark_ingested_from_result(
        candidate_paths: list[str],
        ingest_errors: list[dict[str, str]],
    ) -> None:
        error_paths = {error.get("path", "") for error in ingest_errors}
        for path in candidate_paths:
            if path not in error_paths:
                runtime_state["ingested_paths"].add(path)

    def _complete_query_outputs(
        *,
        response_text: str,
        audio_path: str | None,
        transcribed_text: str,
        status_text: str,
        retrieved_chunks: list[dict[str, Any]],
        trace_text: str,
        history: list[dict[str, str]],
    ) -> tuple[Any, ...]:
        return (
            response_text,
            audio_path,
            transcribed_text,
            status_text,
            render_retrieved_chunks_html(retrieved_chunks),
            trace_text,
            history,
            history_rows(history),
            gr.update(choices=_history_choice_labels(history), value=None),
        )

    def run_text_query(
        query: str,
        source_paths: str,
        uploaded_files: Any,
        top_k: float,
        history: list[dict[str, str]],
    ) -> tuple[Any, ...]:
        query = query.strip()
        history = history or []
        if not query:
            return _complete_query_outputs(
                response_text="",
                audio_path=None,
                transcribed_text="",
                status_text="Please enter a query before running.",
                retrieved_chunks=[],
                trace_text="- status: validation_failed\n- reason: empty query",
                history=history,
            )

        resolved_source_paths = _parse_source_paths(source_paths)
        resolved_source_paths.extend(_normalize_uploaded_files(uploaded_files))
        candidate_paths = _new_source_paths(resolved_source_paths)

        try:
            pipeline = _ensure_orchestrator()
            result = pipeline.answer(
                query,
                source_paths=candidate_paths,
                top_k=_as_int(top_k),
                ingest_sources=bool(candidate_paths),
            )
        except Exception as exc:  # pragma: no cover - guarded by integration behavior
            return _complete_query_outputs(
                response_text="",
                audio_path=None,
                transcribed_text="",
                status_text=f"UI query failed: {exc}",
                retrieved_chunks=[],
                trace_text=f"- status: failed\n- stage: ui\n- error: {exc}",
                history=history,
            )

        if not result.success:
            status_text = (
                f"Pipeline failed at {result.error_stage}: {result.error_message}"
            )
            trace_text = render_execution_trace(
                query=query,
                use_voice=False,
                source_paths=candidate_paths,
                top_k=_as_int(top_k),
                response_text="",
                transcribed_text=result.transcribed_text or "",
                retrieved_chunks=result.retrieved_chunks,
                ingest_errors=result.ingest_errors,
                tts_error=result.tts_error,
            )
            return _complete_query_outputs(
                response_text="",
                audio_path=None,
                transcribed_text=result.transcribed_text or "",
                status_text=status_text,
                retrieved_chunks=result.retrieved_chunks,
                trace_text=trace_text,
                history=history,
            )

        _mark_ingested_from_result(candidate_paths, result.ingest_errors)
        response_text = (result.response_text or "").strip() or (
            "The model returned an empty response. Please try re-running your query."
        )
        status_text = "Ready"
        if result.ingest_errors:
            status_text = f"{status_text} | ingest warnings={len(result.ingest_errors)}"
        if result.tts_error:
            status_text = f"{status_text} | tts warning={result.tts_error}"

        trace_text = render_execution_trace(
            query=query,
            use_voice=False,
            source_paths=candidate_paths,
            top_k=_as_int(top_k),
            response_text=response_text,
            transcribed_text=result.transcribed_text or "",
            retrieved_chunks=result.retrieved_chunks,
            ingest_errors=result.ingest_errors,
            tts_error=result.tts_error,
        )
        history = append_history_entry(
            history,
            query=query,
            transcribed_query=result.transcribed_text or "",
            response=response_text,
            audio_path=str(result.audio_path) if result.audio_path else None,
            status=status_text,
        )

        return _complete_query_outputs(
            response_text=response_text,
            audio_path=str(result.audio_path) if result.audio_path else None,
            transcribed_text=result.transcribed_text or "",
            status_text=status_text,
            retrieved_chunks=result.retrieved_chunks,
            trace_text=trace_text,
            history=history,
        )

    def run_voice_query(
        source_paths: str,
        uploaded_files: Any,
        top_k: float,
        voice_duration_seconds: float,
        history: list[dict[str, str]],
    ) -> tuple[Any, ...]:
        history = history or []
        resolved_source_paths = _parse_source_paths(source_paths)
        resolved_source_paths.extend(_normalize_uploaded_files(uploaded_files))
        candidate_paths = _new_source_paths(resolved_source_paths)

        try:
            pipeline = _ensure_orchestrator()
            result = pipeline.answer(
                query=None,
                use_voice=True,
                voice_duration_seconds=float(voice_duration_seconds),
                source_paths=candidate_paths,
                top_k=_as_int(top_k),
                ingest_sources=bool(candidate_paths),
            )
        except Exception as exc:  # pragma: no cover - guarded by integration behavior
            return _complete_query_outputs(
                response_text="",
                audio_path=None,
                transcribed_text="",
                status_text=f"Push-to-talk failed: {exc}",
                retrieved_chunks=[],
                trace_text=f"- status: failed\n- stage: voice\n- error: {exc}",
                history=history,
            )

        if not result.success:
            status_text = (
                f"Voice query failed at {result.error_stage}: {result.error_message}"
            )
            trace_text = render_execution_trace(
                query=result.query,
                use_voice=True,
                source_paths=candidate_paths,
                top_k=_as_int(top_k),
                response_text="",
                transcribed_text=result.transcribed_text or "",
                retrieved_chunks=result.retrieved_chunks,
                ingest_errors=result.ingest_errors,
                tts_error=result.tts_error,
            )
            return _complete_query_outputs(
                response_text="",
                audio_path=None,
                transcribed_text=result.transcribed_text or "",
                status_text=status_text,
                retrieved_chunks=result.retrieved_chunks,
                trace_text=trace_text,
                history=history,
            )

        _mark_ingested_from_result(candidate_paths, result.ingest_errors)
        response_text = (result.response_text or "").strip() or (
            "The model returned an empty response. Please try re-running your query."
        )
        status_text = "Ready"
        if result.voice_confidence is not None:
            status_text = (
                f"{status_text} | voice confidence={result.voice_confidence:.2f}"
            )
        if result.ingest_errors:
            status_text = f"{status_text} | ingest warnings={len(result.ingest_errors)}"
        if result.tts_error:
            status_text = f"{status_text} | tts warning={result.tts_error}"

        trace_text = render_execution_trace(
            query=result.query,
            use_voice=True,
            source_paths=candidate_paths,
            top_k=_as_int(top_k),
            response_text=response_text,
            transcribed_text=result.transcribed_text or "",
            retrieved_chunks=result.retrieved_chunks,
            ingest_errors=result.ingest_errors,
            tts_error=result.tts_error,
        )
        history = append_history_entry(
            history,
            query=result.query,
            transcribed_query=result.transcribed_text or "",
            response=response_text,
            audio_path=str(result.audio_path) if result.audio_path else None,
            status=status_text,
        )
        return _complete_query_outputs(
            response_text=response_text,
            audio_path=str(result.audio_path) if result.audio_path else None,
            transcribed_text=result.transcribed_text or "",
            status_text=status_text,
            retrieved_chunks=result.retrieved_chunks,
            trace_text=trace_text,
            history=history,
        )

    def ingest_uploaded_files(uploaded_files: Any) -> Any:
        file_paths = _normalize_uploaded_files(uploaded_files)
        if not file_paths:
            yield "No uploaded files selected.", "No files ingested yet.", ""
            return

        pipeline = _ensure_orchestrator()
        results: list[dict[str, Any]] = []
        total = len(file_paths)
        start_time = time.monotonic()

        for index, file_path in enumerate(file_paths, start=1):
            basename = Path(file_path).name
            ingestion = pipeline.ingest_documents([file_path])
            if ingestion.errors:
                first_error = ingestion.errors[0].get("error", "unknown error")
                results.append(
                    {
                        "name": basename,
                        "status": "error",
                        "chunks": 0,
                        "error": first_error,
                    }
                )
            else:
                chunk_count = len(ingestion.chunks)
                results.append(
                    {
                        "name": basename,
                        "status": "success",
                        "chunks": chunk_count,
                        "error": "",
                    }
                )
                runtime_state["ingested_paths"].add(file_path)

            elapsed = max(0.01, time.monotonic() - start_time)
            eta = max(0.0, (elapsed / index) * (total - index))
            status = f"Ingest progress {index}/{total} ({(index / total) * 100:.0f}%) | ETA ~{eta:.1f}s"
            yield status, format_ingest_result_summary(results), "\n".join(file_paths)

        yield (
            f"Ingest completed for {total} file(s).",
            format_ingest_result_summary(results),
            "\n".join(file_paths),
        )

    def _select_history_turn(
        selection: str, history: list[dict[str, str]]
    ) -> tuple[str | None, str]:
        history = history or []
        if not selection:
            return None, ""
        try:
            index = int(selection.split("|", 1)[0].replace("Turn", "").strip()) - 1
        except Exception:
            return None, ""
        if index < 0 or index >= len(history):
            return None, ""
        entry = history[index]
        return (entry.get("audio_path") or None, entry.get("response", ""))

    def _clear_history() -> (
        tuple[list[dict[str, str]], list[list[str]], Any, str | None, str]
    ):
        return [], [], gr.update(choices=[], value=None), None, ""

    def apply_config(*values: Any) -> tuple[str, str, dict[str, Any], str]:
        form_values = _form_values_to_dict(*values)
        payload = build_config_payload(
            base_config=runtime_state["config"], **form_values
        )
        validated_config, errors = validate_config_payload(payload)
        validation_summary = summarize_validation_errors(errors)
        if errors:
            return (
                validation_summary,
                "Configuration not applied.",
                runtime_state["baseline"],
                "Unsaved changes detected.",
            )

        assert validated_config is not None
        save_config_to_path(validated_config, runtime_state["config_path"])
        runtime_state["config"] = validated_config
        runtime_state["orchestrator"] = PipelineOrchestrator.from_config(
            validated_config
        )
        runtime_state["ingested_paths"].clear()
        baseline = config_to_ui_defaults(validated_config)
        runtime_state["baseline"] = baseline
        return (
            validation_summary,
            f"Configuration saved and applied to {runtime_state['config_path']}.",
            baseline,
            "No unsaved changes.",
        )

    def reset_to_active_config() -> tuple[Any, ...]:
        active_config = load_config(runtime_state["config_path"])
        runtime_state["config"] = active_config
        runtime_state["orchestrator"] = PipelineOrchestrator.from_config(active_config)
        runtime_state["ingested_paths"].clear()
        baseline = config_to_ui_defaults(active_config)
        runtime_state["baseline"] = baseline
        return (
            *_config_values_from_defaults(baseline),
            baseline,
            "No unsaved changes.",
        )

    def refresh_profiles(current_value: str | None) -> Any:
        profiles = list_available_profiles(config_path=config_path)
        selected = current_value if current_value in profiles else profiles[0]
        return gr.update(choices=profiles, value=selected)

    def load_profile_into_form(profile_name: str) -> tuple[Any, ...]:
        try:
            profile_config = load_profile_config(profile_name, config_path=config_path)
        except Exception as exc:  # pragma: no cover - guarded by integration behavior
            return (
                *_config_values_from_defaults(runtime_state["baseline"]),
                runtime_state["baseline"],
                f"Profile load failed: {exc}",
            )

        runtime_state["config"] = profile_config
        runtime_state["orchestrator"] = PipelineOrchestrator.from_config(profile_config)
        runtime_state["ingested_paths"].clear()
        profile_defaults = config_to_ui_defaults(profile_config)
        runtime_state["baseline"] = profile_defaults
        return (
            *_config_values_from_defaults(profile_defaults),
            profile_defaults,
            "No unsaved changes.",
        )

    def dirty_state_message(baseline: dict[str, Any], *values: Any) -> str:
        current = _form_values_to_dict(*values)
        return (
            "Unsaved changes detected."
            if is_dirty_config(current, baseline or {})
            else "No unsaved changes."
        )

    runtime_state["baseline"] = dict(defaults)

    with gr.Blocks(title="Voice Agentic RAG") as app:
        gr.Markdown("""
            <div class="block-title">
              <h1>Voice Agentic RAG Console</h1>
                            <p>Run text queries or press Push to Talk for live microphone capture.</p>
            </div>
            """)

        history_state = gr.State([])
        baseline_state = gr.State(dict(defaults))

        with gr.Tab("Assistant"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_classes=["panel"]):
                    query_box = gr.Textbox(
                        label="Your Query",
                        placeholder="Ask a question grounded in your source files...",
                        lines=3,
                    )
                    uploaded_files = gr.File(
                        label="Upload Source Files",
                        file_count="multiple",
                        type="filepath",
                    )
                    ingest_progress_box = gr.Textbox(
                        label="Ingest Progress",
                        interactive=False,
                    )
                    ingest_result_box = gr.Textbox(
                        label="Ingest Results",
                        lines=6,
                        interactive=False,
                        elem_classes=["mono"],
                    )
                    source_paths_box = gr.Textbox(
                        label="Source Paths (optional)",
                        placeholder="One file path per line",
                        lines=4,
                        elem_classes=["mono"],
                    )
                    top_k_slider = gr.Slider(
                        label="Top-K Retrieval",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=config.retrieval.top_k,
                    )
                    voice_duration_slider = gr.Slider(
                        label="Push-to-Talk Duration (seconds)",
                        minimum=1,
                        maximum=15,
                        step=1,
                        value=5,
                    )
                    with gr.Row():
                        ask_button = gr.Button("Run Query", variant="primary")
                        talk_button = gr.Button("Push to Talk", variant="secondary")

                    status_box = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=5, elem_classes=["panel"]):
                    response_box = gr.Textbox(
                        label="LLM Response", lines=12, interactive=False
                    )
                    audio_player = gr.Audio(
                        label="TTS Playback", type="filepath", interactive=False
                    )
                    transcribed_box = gr.Textbox(
                        label="Transcribed Query (for voice flow)",
                        interactive=False,
                    )

            retrieved_chunks_box = gr.HTML(label="Retrieved Chunks", value="")
            with gr.Accordion("Execution Trace", open=False):
                trace_box = gr.Markdown(value="")

            with gr.Row():
                with gr.Column(elem_classes=["panel"]):
                    history_table = gr.Dataframe(
                        headers=["Turn", "Timestamp", "Query", "Response", "Audio"],
                        datatype=["str", "str", "str", "str", "str"],
                        value=[],
                        interactive=False,
                        label="Conversation History",
                    )
                with gr.Column(elem_classes=["panel"]):
                    history_selector = gr.Dropdown(
                        label="Replay Past Turn",
                        choices=[],
                        value=None,
                    )
                    history_audio_player = gr.Audio(
                        label="History Playback",
                        type="filepath",
                        interactive=False,
                    )
                    history_response_box = gr.Textbox(
                        label="History Response",
                        lines=6,
                        interactive=False,
                    )
                    clear_history_button = gr.Button(
                        "Clear History", variant="secondary"
                    )

            uploaded_files.change(
                fn=ingest_uploaded_files,
                inputs=[uploaded_files],
                outputs=[ingest_progress_box, ingest_result_box, source_paths_box],
            )

            ask_button.click(
                fn=run_text_query,
                inputs=[
                    query_box,
                    source_paths_box,
                    uploaded_files,
                    top_k_slider,
                    history_state,
                ],
                outputs=[
                    response_box,
                    audio_player,
                    transcribed_box,
                    status_box,
                    retrieved_chunks_box,
                    trace_box,
                    history_state,
                    history_table,
                    history_selector,
                ],
            )

            talk_button.click(
                fn=run_voice_query,
                inputs=[
                    source_paths_box,
                    uploaded_files,
                    top_k_slider,
                    voice_duration_slider,
                    history_state,
                ],
                outputs=[
                    response_box,
                    audio_player,
                    transcribed_box,
                    status_box,
                    retrieved_chunks_box,
                    trace_box,
                    history_state,
                    history_table,
                    history_selector,
                ],
            )

            history_selector.change(
                fn=_select_history_turn,
                inputs=[history_selector, history_state],
                outputs=[history_audio_player, history_response_box],
            )

            clear_history_button.click(
                fn=_clear_history,
                outputs=[
                    history_state,
                    history_table,
                    history_selector,
                    history_audio_player,
                    history_response_box,
                ],
            )

        with gr.Tab("Configuration"):
            gr.Markdown(
                "Advanced settings are grouped below. Load a profile, validate, and apply to config.yaml."
            )

            with gr.Row():
                profile_selector = gr.Dropdown(
                    label="Configuration Profile",
                    choices=list_available_profiles(config_path=config_path),
                    value=list_available_profiles(config_path=config_path)[0],
                )
                refresh_profiles_button = gr.Button(
                    "Refresh Profiles", variant="secondary"
                )
                load_profile_button = gr.Button("Load Profile", variant="secondary")

            with gr.Row():
                save_apply_button = gr.Button("Save and Apply", variant="primary")
                reset_defaults_button = gr.Button(
                    "Reset to Active Config", variant="secondary"
                )
                check_changes_button = gr.Button(
                    "Check Unsaved Changes", variant="secondary"
                )

            with gr.Accordion("Voice", open=True):
                with gr.Row():
                    stt_engine = gr.Dropdown(
                        label="STT Engine",
                        choices=STT_ENGINE_CHOICES,
                        value=defaults["stt_engine"],
                    )
                    whisper_model = gr.Dropdown(
                        label="Whisper Model",
                        choices=WHISPER_MODEL_CHOICES,
                        value=defaults["whisper_model"],
                    )
                    voice_input_mode = gr.Dropdown(
                        label="Input Mode",
                        choices=INPUT_MODE_CHOICES,
                        value=defaults["voice_input_mode"],
                    )
                    voice_language = gr.Textbox(
                        label="Language", value=defaults["voice_language"]
                    )
                with gr.Row():
                    voice_vad_enabled = gr.Checkbox(
                        label="VAD Enabled", value=defaults["voice_vad_enabled"]
                    )
                    voice_vad_threshold = gr.Slider(
                        label="VAD Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=defaults["voice_vad_threshold"],
                    )

            with gr.Accordion("Chunking", open=False):
                with gr.Row():
                    chunk_strategy = gr.Dropdown(
                        label="Strategy",
                        choices=CHUNK_STRATEGY_CHOICES,
                        value=defaults["chunk_strategy"],
                    )
                    chunk_size = gr.Number(
                        label="Chunk Size", value=defaults["chunk_size"], precision=0
                    )
                    chunk_overlap = gr.Number(
                        label="Chunk Overlap",
                        value=defaults["chunk_overlap"],
                        precision=0,
                    )
                    chunk_unit = gr.Dropdown(
                        label="Chunk Unit",
                        choices=CHUNK_UNIT_CHOICES,
                        value=defaults["chunk_unit"],
                    )
                with gr.Row():
                    chunk_min_size = gr.Number(
                        label="Min Chunk Size",
                        value=defaults["chunk_min_size"],
                        precision=0,
                    )
                    chunk_max_size = gr.Number(
                        label="Max Chunk Size",
                        value=defaults["chunk_max_size"],
                        precision=0,
                    )
                    chunk_rows_per_chunk = gr.Number(
                        label="Rows per Chunk",
                        value=defaults["chunk_rows_per_chunk"],
                        precision=0,
                    )
                    chunk_semantic_similarity_threshold = gr.Slider(
                        label="Semantic Similarity Threshold",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=defaults["chunk_semantic_similarity_threshold"],
                    )
                with gr.Row():
                    chunk_respect_sentence_boundaries = gr.Checkbox(
                        label="Respect Sentence Boundaries",
                        value=defaults["chunk_respect_sentence_boundaries"],
                    )
                    chunk_prepend_metadata = gr.Checkbox(
                        label="Prepend Metadata",
                        value=defaults["chunk_prepend_metadata"],
                    )
                    chunk_include_headers = gr.Checkbox(
                        label="Include Headers", value=defaults["chunk_include_headers"]
                    )
                    chunk_include_notes = gr.Checkbox(
                        label="Include Notes", value=defaults["chunk_include_notes"]
                    )
                with gr.Row():
                    chunk_heading_levels = gr.Textbox(
                        label="Heading Levels (comma-separated)",
                        value=defaults["chunk_heading_levels"],
                    )
                    chunk_target_tags = gr.Textbox(
                        label="Target Tags (comma-separated)",
                        value=defaults["chunk_target_tags"],
                    )

            with gr.Accordion("Retrieval & Embeddings", open=False):
                with gr.Row():
                    embedding_model = gr.Textbox(
                        label="Embedding Model", value=defaults["embedding_model"]
                    )
                    vector_store = gr.Dropdown(
                        label="Vector Store",
                        choices=VECTOR_STORE_CHOICES,
                        value=defaults["vector_store"],
                    )
                    retrieval_top_k = gr.Slider(
                        label="Top-K",
                        minimum=1,
                        maximum=20,
                        step=1,
                        value=defaults["retrieval_top_k"],
                    )
                    retrieval_bm25_weight = gr.Slider(
                        label="BM25 Weight",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=defaults["retrieval_bm25_weight"],
                    )
                with gr.Row():
                    retrieval_hybrid_search = gr.Checkbox(
                        label="Hybrid Search", value=defaults["retrieval_hybrid_search"]
                    )
                    retrieval_rerank = gr.Checkbox(
                        label="Rerank", value=defaults["retrieval_rerank"]
                    )
                    retrieval_rerank_candidate_pool = gr.Number(
                        label="Rerank Candidate Pool",
                        value=defaults["retrieval_rerank_candidate_pool"],
                        precision=0,
                    )
                    retrieval_rerank_min_score = gr.Number(
                        label="Rerank Min Score",
                        value=defaults["retrieval_rerank_min_score"],
                    )
                retrieval_rerank_model = gr.Textbox(
                    label="Rerank Model", value=defaults["retrieval_rerank_model"]
                )

            with gr.Accordion("LLM, TTS, and UI", open=False):
                with gr.Row():
                    llm_provider = gr.Dropdown(
                        label="LLM Provider",
                        choices=LLM_PROVIDER_CHOICES,
                        value=defaults["llm_provider"],
                    )
                    llm_model = gr.Textbox(
                        label="LLM Model", value=defaults["llm_model"]
                    )
                    llm_base_url = gr.Textbox(
                        label="LLM Base URL", value=defaults["llm_base_url"]
                    )
                    llm_api_key = gr.Textbox(
                        label="LLM API Key",
                        value=defaults["llm_api_key"],
                        type="password",
                    )
                with gr.Row():
                    llm_temperature = gr.Slider(
                        label="LLM Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=defaults["llm_temperature"],
                    )
                    llm_max_tokens = gr.Number(
                        label="LLM Max Tokens",
                        value=defaults["llm_max_tokens"],
                        precision=0,
                    )
                llm_system_prompt = gr.Textbox(
                    label="System Prompt", value=defaults["llm_system_prompt"], lines=4
                )
                with gr.Row():
                    tts_engine = gr.Dropdown(
                        label="TTS Engine",
                        choices=TTS_ENGINE_CHOICES,
                        value=defaults["tts_engine"],
                    )
                    tts_rate = gr.Slider(
                        label="TTS Rate",
                        minimum=0.1,
                        maximum=2.0,
                        step=0.05,
                        value=defaults["tts_rate"],
                    )
                    tts_volume = gr.Slider(
                        label="TTS Volume",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=defaults["tts_volume"],
                    )
                    tts_mute = gr.Checkbox(label="Mute TTS", value=defaults["tts_mute"])
                with gr.Row():
                    ui_host = gr.Textbox(label="UI Host", value=defaults["ui_host"])
                    ui_port = gr.Number(
                        label="UI Port", value=defaults["ui_port"], precision=0
                    )
                    ui_show_sources = gr.Checkbox(
                        label="Show Sources", value=defaults["ui_show_sources"]
                    )
                    ui_show_agent_trace = gr.Checkbox(
                        label="Show Agent Trace", value=defaults["ui_show_agent_trace"]
                    )

            validation_summary_box = gr.Textbox(
                label="Validation Summary", interactive=False, lines=6
            )
            config_status = gr.Textbox(
                label="Save / Apply Status", interactive=False, lines=3
            )
            dirty_state_box = gr.Textbox(label="Dirty State", interactive=False)

            config_inputs = [
                stt_engine,
                whisper_model,
                voice_input_mode,
                voice_language,
                voice_vad_enabled,
                voice_vad_threshold,
                chunk_strategy,
                chunk_size,
                chunk_overlap,
                chunk_unit,
                chunk_min_size,
                chunk_max_size,
                chunk_respect_sentence_boundaries,
                chunk_prepend_metadata,
                chunk_heading_levels,
                chunk_rows_per_chunk,
                chunk_include_headers,
                chunk_include_notes,
                chunk_target_tags,
                chunk_semantic_similarity_threshold,
                embedding_model,
                vector_store,
                retrieval_top_k,
                retrieval_hybrid_search,
                retrieval_bm25_weight,
                retrieval_rerank,
                retrieval_rerank_model,
                retrieval_rerank_candidate_pool,
                retrieval_rerank_min_score,
                llm_provider,
                llm_model,
                llm_base_url,
                llm_temperature,
                llm_max_tokens,
                llm_system_prompt,
                llm_api_key,
                tts_engine,
                tts_rate,
                tts_volume,
                tts_mute,
                ui_host,
                ui_port,
                ui_show_sources,
                ui_show_agent_trace,
            ]

            save_apply_button.click(
                fn=apply_config,
                inputs=config_inputs,
                outputs=[
                    validation_summary_box,
                    config_status,
                    baseline_state,
                    dirty_state_box,
                ],
            )

            reset_defaults_button.click(
                fn=reset_to_active_config,
                outputs=[*config_inputs, baseline_state, dirty_state_box],
            )

            refresh_profiles_button.click(
                fn=refresh_profiles,
                inputs=[profile_selector],
                outputs=[profile_selector],
            )

            load_profile_button.click(
                fn=load_profile_into_form,
                inputs=[profile_selector],
                outputs=[*config_inputs, baseline_state, dirty_state_box],
            )

            check_changes_button.click(
                fn=dirty_state_message,
                inputs=[baseline_state, *config_inputs],
                outputs=[dirty_state_box],
            )

    return app


def launch_gradio_app(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """Create and launch the Gradio app from config defaults."""

    config = load_config(config_path)
    app = create_gradio_app(config_path=config_path)
    server_name = host or config.ui.host
    preferred_port = port or config.ui.port
    server_port = _resolve_server_port(server_name, preferred_port)
    app.launch(
        server_name=server_name,
        server_port=server_port,
        css=APP_CSS,
    )


if __name__ == "__main__":
    launch_gradio_app()
