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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If you are using the local environment already present in this workspace, activate that same `.venv` before running the app or tests.

### System Requirements

Before installing Python packages, ensure system-level dependencies are available:

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg libsndfile1
```

**Windows:**
- Download FFmpeg from https://ffmpeg.org/download.html
- PyAudio wheels are handled by pip

### Ollama Setup (Recommended for Local LLM)

1. Download and install from https://ollama.ai
2. In a separate terminal, start Ollama:
   ```bash
   ollama serve
   ```
3. Pull the default model (`mistral`):
   ```bash
   ollama pull mistral
   ```
4. Verify it's running:
   ```bash
   curl http://127.0.0.1:11434/api/tags
   ```

**Keep the Ollama service running while using the app.** If you close it, LLM queries will fail with a connection error.

### Environment Variables

Create or update a `.env` file in the project root for API keys and custom settings:

```
# Groq API (required only if using groq as LLM provider)
GROQ_API_KEY=your_groq_api_key_here

# ============================================================================
# Ollama Configuration (for local LLM mode)
# ============================================================================
# 
# Ollama Setup:
#   1. Install from https://ollama.ai
#   2. Pull model: ollama pull mistral
#   3. Start service: ollama serve
#   4. Keep running while using the app
#
# Default Ollama address (uncomment to override):
# OLLAMA_BASE_URL=http://127.0.0.1:11434
#
# To use Ollama, update config/config.yaml:
#   llm:
#     provider: ollama
#     model: mistral
#     base_url: http://127.0.0.1:11434

# Optional: Override default Groq base URL
# GROQ_BASE_URL=https://api.groq.com/openai/v1
```

The app will automatically load these variables when it starts. No additional code changes needed.

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

### Setup Issues

**Virtual environment not activating**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows CMD
.venv\Scripts\activate.bat

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

**Pip install fails with PyAudio errors**
- Install system audio libraries first (see Installation section)
- macOS: `brew install portaudio`
- Linux: `sudo apt-get install portaudio19-dev`
- Windows: Download PyAudio wheels or use conda

### LLM Provider Issues

**`ConnectionError: Cannot reach Ollama at http://127.0.0.1:11434`**
1. Is Ollama installed? Download from https://ollama.ai
2. Is the service running? In a separate terminal: `ollama serve`
3. Did you pull a model? `ollama pull mistral`
4. Check it's accessible: `curl http://127.0.0.1:11434/api/tags`

**`Provider error: mistral not found`**
- Pull the model: `ollama pull mistral`
- List available models: `ollama list`
- Update `config/config.yaml` with an available model name

**Groq API fails with 401 Unauthorized**
1. Verify `GROQ_API_KEY` is set in `.env`
2. Check the key is valid at https://console.groq.com
3. Restart the app after updating `.env`

**App starts but LLM responses are very slow**
- Ollama model is running on CPU (normal for local setup)
- Try a smaller model: `ollama pull orca-mini` and update config
- Check available system RAM with `ollama list`

### Configuration Issues

**`Invalid YAML in config/config.yaml`**
- Verify indentation (spaces, not tabs)
- Use a YAML validator: https://www.yamllint.com
- Check quotes around special characters

**Selected embedding model not found**
- Model is being downloaded on first use (check logs)
- If stuck, manually download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"`

### Document & Retrieval Issues

**Empty or irrelevant answers**
- Are source files actually being provided? Check the UI upload
- Did the retrieval return chunks? View in the UI response panel
- Try adjusting `retrieval.top_k` in config (default: 5)
- For better results, use well-structured documents with headings

**`ParserError: Unsupported format` for a document**
- Check if the file extension is in the `Supported Document Formats` list in README
- Verify the file is not corrupted
- Try converting to `.pdf` or `.txt` as fallback

**Vector store errors or stuck indexing**
- Delete the vector store cache: `rm -rf data/chroma_db`
- Restart the app
- Re-upload and re-ingest documents

### Voice & Audio Issues

**Microphone not working**
- Check OS permissions: Settings → Privacy & Security → Microphone (macOS)
- Verify `sounddevice` is installed: `python -c "import sounddevice"`
- Test system microphone: `python -c "import sounddevice as sd; print(sd.query_devices())"`
- Restart the app after granting permissions

**`silero-vad` or `faster-whisper` missing**
- These are in requirements.txt and should install automatically
- If missing, try: `pip install --upgrade faster-whisper silero-vad`

**Voice input records but transcription fails**
- STT model is downloading on first use (check logs, be patient)
- `whisper_model: small` is recommended for fast inference
- Check GPU memory if using larger models

**No audio output even with `tts.mute: false`**
1. Check `tts.engine` is installed: `pyttsx3`, `gtts`, or `kokoro`
2. Verify speakers are working: `python -c "import pygame; pygame.mixer.init()"`
3. Check OS volume is not muted
4. View logs for TTS backend errors

### General Debugging

**Enable verbose logging**
- Increase log level in `config/config.yaml` (if available)
- Check logs in the `logs/` directory

**Check dependencies are installed correctly**
```bash
python -c "from src.config.loader import load_config; config = load_config('config/config.yaml'); print('Config loaded OK')"
```

**If all else fails, reset to defaults**
```bash
# Backup current config
cp config/config.yaml config/config.yaml.backup

# Delete caches
rm -rf data/chroma_db logs/*.log

# Restart from Quick Start in README
```

## Useful Files

- [README.md](README.md)
- [ARCHITECTURE.md](ARCHITECTURE.md)
- [config/config.yaml](config/config.yaml)
- [src/pipeline.py](src/pipeline.py)
- [src/ui/gradio_app.py](src/ui/gradio_app.py)
