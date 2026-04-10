# Voice-Based Agentic RAG System

A modular Retrieval-Augmented Generation (RAG) system with voice input and output, multi-format document ingestion, configurable chunking, semantic retrieval, and local-first runtime options.

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

### Required
- **Python 3.11 or newer**
- **Pip** package manager

### System-Level Dependencies

#### macOS
```bash
# For audio support (PyAudio)
brew install portaudio

# For video/encoding support
brew install ffmpeg
```

#### Linux (Ubuntu/Debian)
```bash
# For audio support (PyAudio)
sudo apt-get install portaudio19-dev python3-pyaudio

# For encoding and voice support
sudo apt-get install ffmpeg libsndfile1
```

#### Windows
- Install Python 3.11+ with pip
- PyAudio wheels are included in requirements.txt
- FFmpeg: Download from https://ffmpeg.org/download.html or `choco install ffmpeg`

### LLM Provider Setup

**For local LLM mode (recommended for privacy): Install Ollama**

1. Download and install from https://ollama.ai
2. Pull a model (default config uses `mistral`):
   ```bash
   ollama pull mistral
   ```
   Other popular options: `neural-chat`, `orca-mini`, `llama2`

3. Start the Ollama service in a separate terminal:
   ```bash
   ollama serve
   ```
   Ollama will run on `http://127.0.0.1:11434` by default.

4. Verify it's running:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

**For remote LLM mode: Groq API (free)**

1. Sign up at https://console.groq.com
2. Create an API key
3. Add to `.env` file:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. Update `config/config.yaml`:
   ```yaml
   llm:
     provider: groq
     model: mixtral-8x7b-32768
   ```

### Optional Dependencies

- `PyAudio` may require system-level audio libraries (see above by OS)
- Some document parsers (PDF, Office) depend on native libraries
- Microphone and speaker access required for voice features

## Quick Start

### Text-Only Mode (No Voice, No Ollama)

```bash
cd Rag-System
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.ui.gradio_app
```

Then update `config/config.yaml`:
```yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
voice:
  input_mode: text
```

### Local LLM Mode with Ollama (Recommended)

**Prerequisites**: System libraries installed (see Prerequisites section), Ollama installed and running

```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Set up and run the app
cd Rag-System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.ui.gradio_app
```

The default `config/config.yaml` is already configured for Ollama:
```yaml
llm:
  provider: ollama
  model: mistral
  base_url: http://127.0.0.1:11434
```

### With Voice Features

Same steps as Ollama mode above, then in the UI:
- Go to **Settings** tab
- Set `voice.input_mode: voice` 
- Choose an STT model (`small` is good for most machines)
- Allow microphone access when prompted

Default UI endpoint: `http://127.0.0.1:7860`

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
