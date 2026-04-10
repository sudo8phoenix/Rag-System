# User Guide

## What This Project Does

This project is a voice-capable retrieval-augmented generation system. You can ask questions in text or by voice, optionally supply one or more source documents, and get a grounded answer back as text and, when enabled, audio.

The default entry point is the Gradio web app in [src/ui/gradio_app.py](src/ui/gradio_app.py). The main orchestration path is implemented in [src/pipeline.py](src/pipeline.py).

## Prerequisites

- Python 3.11 or newer
- A local virtual environment in `.venv`
- The project dependencies installed from `requirements.txt`
- Optional services depending on how you configure the app:
  - Ollama for local LLM inference
  - Groq API key for remote LLM access
  - Working microphone and speaker access for voice mode

Some voice and TTS features depend on optional native packages. If a backend is missing, the app falls back where possible or surfaces a clear error.

## Installation

From the repository root:

```bash
cd Rag-System
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using the local environment already present in this workspace, activate that same `.venv` before running the app or tests.

## Quick Start

Launch the UI with:

```bash
python -m src.ui.gradio_app
```

By default the app runs on `http://127.0.0.1:7860`.

The basic workflow is:

1. Open the UI.
2. Check or adjust settings in the configuration tab.
3. Enter a question in the text box or use push-to-talk.
4. Optionally provide source file paths.
5. Submit the request and review the response, retrieved chunks, and audio output.

## Configuration

The active configuration lives in `config/config.yaml`. The most important sections are:

- `voice`: STT engine, Whisper model, VAD, and input mode
- `chunking`: chunking strategy, size, overlap, and per-format overrides
- `embedding`: embedding model and vector store choice
- `retrieval`: top-k, hybrid retrieval, and reranking controls
- `llm`: provider, model, temperature, and system prompt
- `tts`: TTS engine, rate, volume, and mute flag
- `ui`: host, port, and UI visibility settings

A few defaults are worth knowing:

- `voice.input_mode` defaults to text in the UI when you are not using microphone capture.
- `embedding.vector_store` defaults to `chroma`.
- `retrieval.rerank` is enabled by default.
- `tts.mute` can suppress audio playback even when synthesis succeeds.

## Text Queries

For text mode, type a query directly into the UI and optionally add source files. The pipeline will parse the files, chunk them, index the chunks, retrieve context, and generate an answer.

A code-level example looks like this:

```python
from src.config.loader import load_config
from src.pipeline import PipelineOrchestrator

config = load_config("config/config.yaml")
orchestrator = PipelineOrchestrator.from_config(config)

result = orchestrator.answer(
    query="Summarize the uploaded documents.",
    source_paths=["./data/example.pdf", "./data/notes.md"],
    ingest_sources=True,
)

print(result.response_text)
```

## Voice Queries

Voice mode uses microphone capture, voice activity detection, and speech-to-text. When enabled, the app records audio, isolates speech segments, transcribes them, and sends the transcription through the same retrieval pipeline as a text query.

If voice capture fails, the most common causes are:

- Microphone permissions are blocked by the OS
- `sounddevice` or `pyaudio` is missing
- `silero-vad` or `faster-whisper` is missing
- The selected STT model is too large for the available hardware

## Working With Documents

Supported documents include text, Markdown, office documents, PDF, spreadsheets, JSON, HTML, XML, EPUB, and slide decks.

For best results:

- Use source files with clean structure and headings when possible.
- Keep document sets focused if you want precise answers.
- Use the configuration page to choose a chunking strategy that matches the file type.
- If a source format is unsupported or the parser fails, the registry falls back to text parsing where possible.

The response view includes the retrieved chunks used as grounding evidence. That makes it easier to verify whether the answer came from the right source material.

## TTS Output

If TTS is enabled and `tts.mute` is false, the system writes an audio file and plays it back through the configured engine.

Supported engines in the current codebase are:

- `pyttsx3`
- `gtts`
- `kokoro`

If the preferred engine fails, the orchestrator tries the remaining supported engines before returning an error.

## Troubleshooting

If the app does not start, check the following first:

- The virtual environment is active.
- `config/config.yaml` is valid YAML.
- The selected LLM provider is reachable.
- The configured embedding model and vector store are available.
- Your OS allows microphone and speaker access.

If you see empty answers, verify that the source files were actually provided and that retrieval returned relevant chunks.

If you hear no audio, check `tts.mute` and confirm the backend dependency is installed.

If Groq fails with an authentication error, set a valid `GROQ_API_KEY` in your environment or in the config UI.

## Useful Files

- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [config/config.yaml](config/config.yaml)
- [src/pipeline.py](src/pipeline.py)
- [src/ui/gradio_app.py](src/ui/gradio_app.py)
