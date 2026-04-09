# Voice-Based Agentic RAG System

A modular Retrieval-Augmented Generation (RAG) system with voice input and output, multi-format document ingestion, configurable chunking, semantic retrieval, and local-first runtime options.

## Status

- Phase 1 to Phase 3 are implemented and validated in tests.
- Phase 4 (advanced LangGraph agent loop and dynamic tool routing) is planned next.

## Core Capabilities

- Voice input pipeline with configurable STT engines:
	- `faster-whisper`
	- `whisper`
	- `speechrecognition`
- Text and voice query modes.
- Multi-format parsing and normalization.
- Configurable chunking with per-format overrides.
- Embeddings and vector stores:
	- `BAAI/bge-m3` and `nomic-embed-text`
	- `chroma`, `faiss`, `qdrant`
- Retrieval features:
	- Dense retrieval
	- Hybrid dense + BM25 retrieval
	- Cross-encoder reranking
- LLM providers:
	- `ollama`
	- `groq`
- TTS engines:
	- `pyttsx3`
	- `gtts`
	- `kokoro`
	- `bark`
	- `elevenlabs`
- Gradio UI for end-to-end usage.

## Supported Document Formats

- Text and markup: `.txt`, `.md`, `.mdx`, `.html`, `.htm`, `.xml`
- Office docs: `.docx`, `.doc`, `.odt`
- PDF: `.pdf`
- Spreadsheets and tabular: `.xlsx`, `.xlsm`, `.xls`, `.csv`, `.tsv`
- Structured data: `.json`, `.jsonl`, `.ndjson`
- Publishing and slides: `.epub`, `.pptx`, `.ppt`

## Architecture

The system is organized into modular boundaries so each stage can evolve independently:

- `src/config`: schema and YAML loading
- `src/parsers`: parser registry and format-specific parsers
- `src/chunkers`: chunk strategy implementations and dispatch
- `src/embeddings`: embedder, vector store, retriever, orchestration
- `src/llm`: LLM wrappers and prompt execution
- `src/voice`: microphone capture, VAD, and STT orchestration
- `src/tts`: TTS engines and playback orchestration
- `src/ui`: Gradio application
- `src/pipeline.py`: end-to-end orchestration (`PipelineOrchestrator`)

See `ARCHITECTURE.md` for the full component flow.

## Project Layout

```
Rag-System/
	config/
		config.yaml
	data/
		tts/
	logs/
	src/
		pipeline.py
		chunkers/
		config/
		embeddings/
		llm/
		models/
		parsers/
		tts/
		ui/
		voice/
	tests/
```

## Prerequisites

- Python 3.11+
- Pip
- Optional but recommended for local LLM mode:
	- Ollama running locally when `llm.provider: ollama`

Notes:

- `PyAudio` may require system-level audio libraries depending on your OS.
- Some format parsers rely on optional native dependencies in the environment.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Review and adjust `config/config.yaml`.
4. Launch the Gradio app.

Example:

```bash
cd Rag-System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ui.gradio_app
```

Default UI endpoint:

- `http://127.0.0.1:7860`

## Configuration

Main configuration file:

- `config/config.yaml`

Top-level config sections:

- `voice`
- `chunking`
- `embedding`
- `retrieval`
- `llm`
- `agent`
- `tts`
- `ui`

Useful defaults in the current config:

- `voice.input_mode: text`
- `embedding.vector_store: chroma`
- `retrieval.rerank: true`
- `retrieval.rerank_min_score: null`
- `tts.mute: true`

## Running The Pipeline In Code

```python
from src.config.loader import load_config
from src.pipeline import PipelineOrchestrator

config = load_config("config/config.yaml")
orchestrator = PipelineOrchestrator.from_config(config)

result = orchestrator.answer(
		query="Summarize the main points from these files",
		source_paths=["./data/example.pdf", "./data/notes.md"],
		ingest_sources=True,
)

print(result.response_text)
print(result.success)
```

## Testing

Run all tests:

```bash
pytest -q
```

Run targeted suites:

```bash
pytest tests/unit/chunkers -q
pytest tests/unit/embeddings -q
pytest tests/integration -q
```

## Current Milestone Summary

- Parser and chunking framework implemented across planned formats.
- Semantic and token-based chunkers implemented and covered by tests.
- Retrieval supports hybrid search and reranking.
- Chroma is the default vector store, with FAISS and Qdrant support available.
- End-to-end orchestration available via `PipelineOrchestrator` and Gradio UI.

## Roadmap

Next major step is Phase 4:

- LangGraph ReAct-style state graph
- Dynamic tool routing during reasoning
- Expanded memory and agent trace capabilities

## License

Internal project for internship task execution. Add your license terms here if this repository will be distributed.
