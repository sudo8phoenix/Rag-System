# Architecture

## Overview

The Voice-Based Agentic RAG System is organized as a modular pipeline so each stage can evolve independently while preserving a stable data contract between components.

Core flow:

1. Voice input is captured and segmented with VAD.
2. Speech segments are transcribed to text (STT).
3. Documents are parsed and converted to normalized text.
4. Chunking strategies convert text into structured chunks with metadata.
5. Embedding and vector indexing enable semantic retrieval.
6. Retrieved context is injected into an LLM prompt.
7. LLM output is synthesized to speech via TTS.

## High-Level Component Diagram (ASCII)

```text
+--------------------+      +--------------------+      +--------------------+
|  Voice Input       | ---> |  STT + Query Text  | ---> |  Orchestrator      |
|  (Mic + VAD)       |      |  (faster-whisper)  |      |  (Pipeline Core)   |
+--------------------+      +--------------------+      +--------------------+
                                                                |
                                                                v
                                                        +--------------------+
                                                        |  Parser Registry   |
                                                        |  (by file format)  |
                                                        +--------------------+
                                                                |
                                                                v
                                                        +--------------------+
                                                        |  Chunking Engine   |
                                                        |  (line/char/para)  |
                                                        +--------------------+
                                                                |
                                                                v
                                                        +--------------------+
                                                        |  Embeddings +      |
                                                        |  Vector Store      |
                                                        |  (FAISS first)     |
                                                        +--------------------+
                                                                |
                                                                v
                                                        +--------------------+
                                                        |  Retriever         |
                                                        |  (Top-k chunks)    |
                                                        +--------------------+
                                                                |
                                                                v
+--------------------+      +--------------------+      +--------------------+
|  TTS Output        | <--- |  LLM Response      | <--- |  Prompt Builder    |
|  (pyttsx3)         |      |  (Ollama llama3)   |      |  (context + query) |
+--------------------+      +--------------------+      +--------------------+
```

## Module Boundaries

- `src/config/`: Pydantic schemas, YAML loading, environment overrides
- `src/models/`: Shared data contracts (`Document`, `Chunk`)
- `src/parsers/`: File-format-specific parsing plus dispatcher
- `src/chunkers/`: Strategy implementations and selection logic
- `src/embeddings/`: Embedding model wrapper, index storage, retrieval
- `src/llm/`: Ollama integration and prompt execution
- `src/tts/`: Text-to-speech engines and playback
- `src/voice/`: Microphone capture, VAD, STT orchestration
- `src/ui/`: Gradio application and configuration controls
- `src/pipeline.py`: End-to-end orchestration across modules

## Data Contracts

- `Document`: normalized source text + source metadata
- `Chunk`: chunk text + source references + strategy metadata
- Retrieval result: chunk reference + similarity score + source metadata

## Non-Functional Notes

- Config-first development: all major runtime parameters should be configurable.
- Metadata preservation: every transform keeps source provenance.
- Graceful degradation: if voice or TTS fails, system should continue in text mode.
- Testability: each module should support unit tests with deterministic inputs.
