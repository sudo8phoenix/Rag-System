"""Gradio UI for the Phase 1 RAG pipeline."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.loader import load_config
from src.config.settings import AppConfig
from src.pipeline import PipelineOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

STT_ENGINE_CHOICES = ["faster-whisper", "whisper", "speechrecognition"]
WHISPER_MODEL_CHOICES = ["tiny", "base", "small", "medium", "large"]
CHUNK_STRATEGY_CHOICES = ["line", "character", "paragraph"]
VECTOR_STORE_CHOICES = ["faiss", "chroma", "qdrant"]
TTS_ENGINE_CHOICES = ["pyttsx3", "gtts", "kokoro", "bark", "elevenlabs"]

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


def _parse_source_paths(raw_paths: str) -> list[str]:
    if not raw_paths.strip():
        return []

    normalized = raw_paths.replace(",", "\n")
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def config_to_ui_defaults(config: AppConfig) -> dict[str, Any]:
    """Map validated app config to initial UI field values."""

    return {
        "stt_engine": config.voice.stt_engine,
        "whisper_model": config.voice.whisper_model,
        "chunk_strategy": config.chunking.strategy,
        "chunk_size": config.chunking.chunk_size,
        "chunk_overlap": config.chunking.chunk_overlap,
        "embedding_model": config.embedding.model,
        "vector_store": config.embedding.vector_store,
        "llm_model": config.llm.model,
        "llm_temperature": config.llm.temperature,
        "tts_engine": config.tts.engine,
        "tts_rate": config.tts.rate,
        "tts_volume": config.tts.volume,
    }


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
    tts_engine: str,
    tts_rate: float,
    tts_volume: float,
    base_config: AppConfig,
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
    payload["tts"]["engine"] = tts_engine
    payload["tts"]["rate"] = float(tts_rate)
    payload["tts"]["volume"] = float(tts_volume)
    return payload


def validate_config_payload(payload: dict[str, Any]) -> tuple[AppConfig | None, list[str]]:
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
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("Gradio is required. Install it with `pip install gradio`.") from exc

    config = load_config(config_path)
    defaults = config_to_ui_defaults(config)
    runtime_state: dict[str, Any] = {"config": config, "orchestrator": orchestrator}

    def _ensure_orchestrator() -> PipelineOrchestrator:
        if runtime_state["orchestrator"] is None:
            runtime_state["orchestrator"] = PipelineOrchestrator.from_config(runtime_state["config"])
        return runtime_state["orchestrator"]

    def _format_retrieved_chunks(retrieved_chunks: list[dict[str, Any]]) -> str:
        retrieved_lines: list[str] = []
        for index, item in enumerate(retrieved_chunks, start=1):
            chunk = item.get("chunk", {})
            source = chunk.get("source_doc", {}).get("filename", "unknown")
            strategy = chunk.get("strategy_used", "unknown")
            score = item.get("score")
            preview = chunk.get("text", "")
            retrieved_lines.append(
                f"[{index}] {source} | strategy={strategy} | score={score}\\n{preview[:450]}"
            )
        return "\\n\\n".join(retrieved_lines)

    def run_text_query(
        query: str,
        source_paths: str,
        top_k: float,
    ) -> tuple[str, str | None, str, str, str]:
        query = query.strip()
        if not query:
            return "", None, "", "Please enter a query before running.", ""

        try:
            pipeline = _ensure_orchestrator()
            result = pipeline.answer(
                query,
                source_paths=_parse_source_paths(source_paths),
                top_k=_as_int(top_k),
                ingest_sources=bool(source_paths.strip()),
            )
            if not result.success:
                return (
                    "",
                    None,
                    "",
                    f"Pipeline failed at {result.error_stage}: {result.error_message}",
                    "",
                )

            retrieved_text = _format_retrieved_chunks(result.retrieved_chunks)

            return (
                result.response_text,
                str(result.audio_path) if result.audio_path else None,
                result.transcribed_text or "",
                "Ready" if not retrieved_text else "Ready - retrieved context shown below",
                retrieved_text,
            )
        except Exception as exc:  # pragma: no cover - guarded by integration behavior
            return "", None, "", f"UI query failed: {exc}", ""

    def run_voice_query(
        source_paths: str,
        top_k: float,
        voice_duration_seconds: float,
    ) -> tuple[str, str | None, str, str, str]:
        try:
            pipeline = _ensure_orchestrator()
            result = pipeline.answer(
                query=None,
                use_voice=True,
                voice_duration_seconds=float(voice_duration_seconds),
                source_paths=_parse_source_paths(source_paths),
                top_k=_as_int(top_k),
                ingest_sources=bool(source_paths.strip()),
            )
            if not result.success:
                return (
                    "",
                    None,
                    result.transcribed_text or "",
                    f"Voice query failed at {result.error_stage}: {result.error_message}",
                    "",
                )

            retrieved_text = _format_retrieved_chunks(result.retrieved_chunks)
            status = "Ready" if not retrieved_text else "Ready - retrieved context shown below"
            if result.voice_confidence is not None:
                status = f"{status} | voice confidence={result.voice_confidence:.2f}"

            return (
                result.response_text,
                str(result.audio_path) if result.audio_path else None,
                result.transcribed_text or "",
                status,
                retrieved_text,
            )
        except Exception as exc:  # pragma: no cover - guarded by integration behavior
            return "", None, "", f"Push-to-talk failed: {exc}", ""

    def apply_config(
        stt_engine: str,
        whisper_model: str,
        chunk_strategy: str,
        chunk_size: float,
        chunk_overlap: float,
        embedding_model: str,
        vector_store: str,
        llm_model: str,
        llm_temperature: float,
        tts_engine: str,
        tts_rate: float,
        tts_volume: float,
    ) -> str:
        payload = build_config_payload(
            stt_engine=stt_engine,
            whisper_model=whisper_model,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm_model=llm_model,
            llm_temperature=llm_temperature,
            tts_engine=tts_engine,
            tts_rate=tts_rate,
            tts_volume=tts_volume,
            base_config=runtime_state["config"],
        )

        validated_config, errors = validate_config_payload(payload)
        if errors:
            return "Validation failed:\n- " + "\n- ".join(errors)

        assert validated_config is not None
        save_config_to_path(validated_config, config_path)
        runtime_state["config"] = validated_config
        runtime_state["orchestrator"] = PipelineOrchestrator.from_config(validated_config)
        return f"Configuration saved and applied to {config_path}."

    with gr.Blocks(title="Voice Agentic RAG") as app:
        gr.Markdown(
            """
            <div class="block-title">
              <h1>Voice Agentic RAG Console</h1>
                            <p>Run text queries or press Push to Talk for live microphone capture.</p>
            </div>
            """
        )

        with gr.Tab("Assistant"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=7, elem_classes=["panel"]):
                    query_box = gr.Textbox(
                        label="Your Query",
                        placeholder="Ask a question grounded in your source files...",
                        lines=3,
                    )
                    source_paths_box = gr.Textbox(
                        label="Source Paths (optional)",
                        placeholder="One file path per line, or comma-separated paths",
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
                    response_box = gr.Textbox(label="LLM Response", lines=12, interactive=False)
                    audio_player = gr.Audio(label="TTS Playback", type="filepath", interactive=False)
                    transcribed_box = gr.Textbox(
                        label="Transcribed Query (for voice flow)",
                        interactive=False,
                    )

            retrieved_chunks_box = gr.Textbox(
                label="Retrieved Chunks",
                lines=12,
                interactive=False,
                elem_classes=["mono", "panel"],
            )

            ask_button.click(
                fn=run_text_query,
                inputs=[query_box, source_paths_box, top_k_slider],
                outputs=[
                    response_box,
                    audio_player,
                    transcribed_box,
                    status_box,
                    retrieved_chunks_box,
                ],
            )

            talk_button.click(
                fn=run_voice_query,
                inputs=[source_paths_box, top_k_slider, voice_duration_slider],
                outputs=[
                    response_box,
                    audio_player,
                    transcribed_box,
                    status_box,
                    retrieved_chunks_box,
                ],
            )

        with gr.Tab("Configuration"):
            gr.Markdown(
                "Adjust core pipeline settings, validate them, and apply directly to config.yaml."
            )

            with gr.Row():
                with gr.Column(elem_classes=["panel"]):
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
                    chunk_strategy = gr.Dropdown(
                        label="Chunk Strategy",
                        choices=CHUNK_STRATEGY_CHOICES,
                        value=defaults["chunk_strategy"],
                    )
                    chunk_size = gr.Number(label="Chunk Size", value=defaults["chunk_size"], precision=0)
                    chunk_overlap = gr.Number(
                        label="Chunk Overlap", value=defaults["chunk_overlap"], precision=0
                    )
                    embedding_model = gr.Textbox(
                        label="Embedding Model",
                        value=defaults["embedding_model"],
                    )

                with gr.Column(elem_classes=["panel"]):
                    vector_store = gr.Dropdown(
                        label="Vector Store",
                        choices=VECTOR_STORE_CHOICES,
                        value=defaults["vector_store"],
                    )
                    llm_model = gr.Textbox(label="LLM Model", value=defaults["llm_model"])
                    llm_temperature = gr.Slider(
                        label="LLM Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=defaults["llm_temperature"],
                    )
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

            config_status = gr.Textbox(label="Validation / Save Status", interactive=False, lines=6)
            save_apply_button = gr.Button("Save and Apply", variant="primary")

            save_apply_button.click(
                fn=apply_config,
                inputs=[
                    stt_engine,
                    whisper_model,
                    chunk_strategy,
                    chunk_size,
                    chunk_overlap,
                    embedding_model,
                    vector_store,
                    llm_model,
                    llm_temperature,
                    tts_engine,
                    tts_rate,
                    tts_volume,
                ],
                outputs=[config_status],
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
