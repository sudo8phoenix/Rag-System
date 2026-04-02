# Voice-Based Agentic RAG System

A modular Retrieval-Augmented Generation (RAG) system with voice input/output, multi-format document ingestion, chunking strategies, embeddings, retrieval, and agentic orchestration.

## Phase 1 Scope
- Core project structure and configuration
- Voice input pipeline (VAD + STT)
- Parsing for initial document formats
- Chunking, embeddings, retrieval, LLM, and TTS wiring
- Basic Gradio interface

## Project Layout
- `src/` application source code
- `data/` runtime data and vector stores
- `config/` YAML config and environment templates
- `logs/` runtime logs

## Quick Start
1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies from `requirements.txt`.
3. Copy `.env.example` to `.env` and configure secrets.
4. Add default settings to `config/config.yaml` (next setup task).
