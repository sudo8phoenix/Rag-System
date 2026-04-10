# Developer Guide

## Scope

This repository is organized around a small set of stable contracts. Most extensions should fit into one of these layers:

- configuration models in [src/config/settings.py](src/config/settings.py)
- document models in [src/models/](src/models/)
- parsers in [src/parsers/](src/parsers/)
- chunkers in [src/chunkers/](src/chunkers/)
- embeddings and retrieval in [src/embeddings/](src/embeddings/)
- LLM adapters in [src/llm/](src/llm/)
- TTS backends in [src/tts/](src/tts/)
- voice capture in [src/voice/](src/voice/)
- orchestration in [src/pipeline.py](src/pipeline.py)
- UI in [src/ui/gradio_app.py](src/ui/gradio_app.py)

The important rule is to extend the existing registry or orchestrator rather than bypassing it.

## Working Principles

1. Preserve metadata as documents move through the pipeline.
2. Keep new behavior behind config when practical.
3. Prefer small adapters over large cross-cutting changes.
4. Add or update tests with every behavioral change.
5. Keep optional dependencies optional and provide fallbacks where sensible.

## Adding a Parser

Parser base classes live in [src/parsers/base.py](src/parsers/base.py). A parser should inherit from `BaseParser`, set `source_type`, declare supported extensions, and return a normalized `Document`.

Recommended steps:

1. Implement a concrete parser class.
2. Use `BaseParser._build_document()` to construct the result.
3. Add the parser to `ParserRegistry._parsers` in [src/parsers/registry.py](src/parsers/registry.py).
4. Add or update unit tests under `tests/unit/parsers/`.
5. If the format needs conversion or fallback behavior, add it to the registry instead of duplicating logic in the UI or pipeline.

Keep parser output structured. If the parser can preserve headings, table layout, page numbers, row ranges, or chapter titles, store that in `original_metadata` so downstream chunkers can use it.

## Adding a Chunker

Chunker base classes live in [src/chunkers/base.py](src/chunkers/base.py). A chunker should inherit from `BaseChunker`, set `strategy_name`, and implement `chunk(document, config)`.

Recommended steps:

1. Build the chunk list from the `Document` text.
2. Use `BaseChunker._base_metadata()` and `BaseChunker._build_chunk()` so the metadata stays consistent.
3. Register the strategy in `ChunkingRegistry._chunkers` in [src/chunkers/registry.py](src/chunkers/registry.py).
4. Add tests that verify chunk boundaries and metadata.

If the strategy depends on format-specific configuration, use `ChunkingConfig.effective_for_format()` rather than reading raw config values directly.

## Adding Retrieval or Embedding Behavior

Embedding contracts are defined in [src/embeddings/base.py](src/embeddings/base.py). Dense retrieval uses `BaseEmbedder`, `BaseVectorStore`, and `SemanticRetriever`. Sparse retrieval uses `BM25Retriever`. Hybrid ranking is handled by `HybridRetriever`, and cross-encoder reranking is handled by `CrossEncoderReranker`.

When adding a new backend:

- keep the backend class focused on the storage or model adapter
- do not embed retrieval policy in the backend itself
- let `EmbeddingOrchestrator` decide when to use hybrid search or reranking
- preserve `SearchResult` metadata so the UI can show traceability

If you add a new vector store, wire it through `create_vector_store()` in [src/embeddings/orchestrator.py](src/embeddings/orchestrator.py) and make sure `EmbeddingConfig.vector_store` can select it.

## Adding an LLM Provider

LLM wrappers live in [src/llm/](src/llm/). The current pattern is a thin provider adapter plus shared prompt helpers in [src/llm/prompting.py](src/llm/prompting.py).

A good provider implementation should:

- accept an `LLMConfig`
- expose a `generate()` method that returns `LLMResponse`
- expose a streaming method only if the provider supports it cleanly
- translate provider failures into `LLMConnectionError` or `LLMProviderError`
- build prompts through the shared helper so grounding stays consistent

## Adding a TTS Backend

TTS base classes and errors live in [src/tts/base.py](src/tts/base.py). Concrete engines should inherit from `BaseTTSBackend` and implement `synthesize_to_file()`.

Use the orchestrator in [src/tts/orchestrator.py](src/tts/orchestrator.py) to select and fall back between engines. Do not reimplement fallback logic in the UI.

A backend should respect `TTSConfig` for voice, rate, volume, and mute behavior.

## Voice Components

Voice input is intentionally decomposed into three parts:

- `MicrophoneCapture` in [src/voice/mic_capture.py](src/voice/mic_capture.py)
- `SileroVAD` in [src/voice/vad.py](src/voice/vad.py)
- `FasterWhisperSTT` in [src/voice/stt.py](src/voice/stt.py)

The orchestrator in [src/voice/voice_input.py](src/voice/voice_input.py) combines them into a single push-to-talk flow. If you change any of those parts, keep `VoiceInputResult` stable so the pipeline can keep using it.

## Pipeline Changes

The pipeline should remain the top-level integration point. If you add a new capability that affects the end-to-end query path, update [src/pipeline.py](src/pipeline.py) rather than leaking orchestration into the UI or registry layers.

The existing `PipelineResult` and `IngestionResult` dataclasses are the primary contract for UI and tests. Preserve those shapes when possible.

## Testing Guidance

Tests are organized under `tests/unit/` and `tests/integration/`.

Use unit tests to validate the local contract of a parser, chunker, embedder, or backend. Use integration tests to validate that multiple layers cooperate correctly.

Good test targets include:

- metadata preservation through parse/chunk/retrieve
- config-driven selection logic
- fallback behavior when dependencies are missing
- serialization and deserialization of data models
- UI helpers that transform state into display data

Prefer tests that assert behavior instead of implementation details.

## Style and Maintenance

- Keep modules small and import boundaries clear.
- Add docstrings for public classes and functions.
- Use typed return values for new public APIs.
- Avoid hidden global state.
- Keep optional dependencies behind import guards so the codebase still imports in constrained environments.

If you need to make a broad API change, update the package exports in the relevant `__init__.py` files and reflect the change in [API_REFERENCE.md](API_REFERENCE.md).
