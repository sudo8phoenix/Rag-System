# API Reference

This reference covers the public classes and functions that make up the current runtime surface. Private helpers are intentionally omitted.

## Configuration

### [src/config/loader.py](src/config/loader.py)

- `ConfigLoadError`: raised when a config file is missing or invalid YAML.
- `ConfigValidationError`: raised when YAML content does not validate against the schema.
- `load_yaml_file(config_path) -> dict[str, Any]`: read a YAML file into a dictionary.
- `load_config(config_path="config/config.yaml") -> AppConfig`: load and validate the application config.

### [src/config/settings.py](src/config/settings.py)

- `VoiceConfig`: STT, VAD, and voice input settings.
- `FormatChunkingConfig`: per-format chunking overrides.
- `ChunkingConfig`: global chunking limits and strategy selection.
- `EmbeddingConfig`: embedding model and vector store selection.
- `RetrievalConfig`: dense, hybrid, and reranking settings.
- `LLMConfig`: provider, model, prompt, and generation settings.
- `TTSConfig`: synthesis engine, voice, rate, volume, and mute flag.
- `AgentConfig`: reserved agent orchestration settings.
- `UIConfig`: local UI settings.
- `AppConfig`: root config object containing every section.

Key method:

- `ChunkingConfig.effective_for_format(source_type) -> ChunkingConfig`: merge per-format overrides into the global chunking config.

## Core Data Models

### [src/models/document.py](src/models/document.py)

- `Document(text, filename, source_type, original_metadata={})`
- `Document.to_dict() -> dict[str, Any]`

### [src/models/chunk.py](src/models/chunk.py)

- `Chunk(text, chunk_id, source_doc, chunk_index, strategy_used, metadata={})`
- `Chunk.to_dict() -> dict[str, Any]`

## Parsing

### [src/parsers/base.py](src/parsers/base.py)

- `ParserError`: raised when a parser cannot extract content.
- `BaseParser`: abstract parser contract.
- `BaseParser.parse(file_path) -> Document`: parse one file into a normalized document.

### [src/parsers/registry.py](src/parsers/registry.py)

- `ParserRegistry`: extension-based parser dispatcher.
- `ParserRegistry.supported_extensions -> tuple[str, ...]`
- `ParserRegistry.get_parser(file_path) -> BaseParser`
- `ParserRegistry.parse_file(file_path) -> Document`
- `get_parser_for_path(file_path) -> BaseParser`
- `parse_file(file_path) -> Document`

Supported parser classes exported by the package include:

- `TxtParser`
- `MarkdownParser`
- `DocxParser`
- `DocParser`
- `PdfParser`
- `CsvParser`
- `JsonParser`
- `JsonlParser`
- `HtmlParser`
- `XmlParser`
- `EpubParser`
- `PptxParser`
- `PptParser`
- `XlsxParser`
- `XlsParser`
- `OdtParser`

## Chunking

### [src/chunkers/base.py](src/chunkers/base.py)

- `ChunkingError`: raised when a chunker cannot process a document.
- `BaseChunker`: abstract chunking contract.
- `BaseChunker.chunk(document, config) -> list[Chunk]`

### [src/chunkers/registry.py](src/chunkers/registry.py)

- `ChunkingRegistry`: config-driven chunker dispatcher.
- `ChunkingRegistry.supported_strategies -> tuple[str, ...]`
- `ChunkingRegistry.get_chunker(document) -> BaseChunker`
- `ChunkingRegistry.chunk_document(document) -> list[Chunk]`
- `get_chunker_for_document(document, config=None) -> BaseChunker`
- `chunk_document(document, config=None) -> list[Chunk]`

Exported chunkers include:

- `LineBasedChunker`
- `CharacterBasedChunker`
- `ParagraphBasedChunker`
- `HeadingHierarchyChunker`
- `RowBasedChunker`
- `ArrayItemChunker`
- `SlideBasedChunker`
- `TagBasedChunker`
- `ChapterBasedChunker`
- `SemanticBasedChunker`
- `TokenBasedChunker`

## Embeddings and Retrieval

### [src/embeddings/base.py](src/embeddings/base.py)

- `EmbeddingError`: base error for embedding and retrieval operations.
- `EmbeddingDependencyError`: raised when optional dependencies are missing.
- `SearchResult(chunk, score, distance, metadata)`
- `SearchResult.to_dict() -> dict[str, Any]`
- `BaseEmbedder`: abstract text embedder.
- `BaseEmbedder.embed_texts(texts) -> list[list[float]]`
- `BaseEmbedder.embed_text(text) -> list[float]`
- `BaseEmbedder.embed_chunks(chunks) -> list[list[float]]`
- `BaseVectorStore`: abstract vector store contract.
- `StoredChunkRecord(chunk, vector, metadata)`
- `normalize_vector(vector) -> list[float]`
- `cosine_similarity(left, right) -> float`

### [src/embeddings/embedder.py](src/embeddings/embedder.py)

- `DeterministicTextEmbedder`: pure-Python fallback embedder.
- `SentenceTransformerEmbedder(model_name, device="cpu")`
- `OllamaEmbedder(model_name, base_url="http://127.0.0.1:11434", timeout_seconds=60.0)`
- `create_embedder(config=None, allow_fallback=True) -> BaseEmbedder`

### [src/embeddings/vectorstore.py](src/embeddings/vectorstore.py)

- `LocalVectorStore`: in-memory persistent vector store.
- `FaissVectorStore`: FAISS-compatible wrapper.
- `ChromaVectorStore`: Chroma-compatible wrapper.
- `QdrantVectorStore`: Qdrant-compatible wrapper.

Common methods on vector stores:

- `add(chunk, vector, metadata=None) -> None`
- `add_many(chunks, vectors, metadata=None) -> None`
- `search(query_vector, top_k=5, filters=None) -> list[SearchResult]`
- `save(path) -> None`
- `load(path) -> BaseVectorStore`
- `clear() -> None`
- `count() -> int`

### [src/embeddings/retriever.py](src/embeddings/retriever.py)

- `SemanticRetriever(embedder, vector_store)`
- `SemanticRetriever.index_chunks(chunks, metadata=None) -> None`
- `SemanticRetriever.search(query, top_k=5, filters=None) -> list[SearchResult]`
- `BM25Retriever(chunks=None)`
- `BM25Retriever.index_chunks(chunks) -> None`
- `BM25Retriever.add_chunks(chunks) -> None`
- `BM25Retriever.clear() -> None`
- `BM25Retriever.search(query, top_k=5, filters=None) -> list[SearchResult]`
- `HybridRetriever(dense_retriever, bm25_retriever, bm25_weight=0.3)`
- `HybridRetriever.search(query, top_k=5, filters=None) -> list[SearchResult]`
- `CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", min_score=None, model=None)`
- `CrossEncoderReranker.rerank(query, candidates, top_k=5) -> list[SearchResult]`

### [src/embeddings/orchestrator.py](src/embeddings/orchestrator.py)

- `create_vector_store(config=None) -> BaseVectorStore`
- `EmbeddingOrchestrator(...)`
- `EmbeddingOrchestrator.from_config(config) -> EmbeddingOrchestrator`
- `EmbeddingOrchestrator.index_chunks(chunks, metadata=None) -> None`
- `EmbeddingOrchestrator.search(query, top_k=None, filters=None) -> list[SearchResult]`
- `EmbeddingOrchestrator.save(path) -> None`
- `EmbeddingOrchestrator.load(path, embedding_config=None, retrieval_config=None) -> EmbeddingOrchestrator`

## LLM

### [src/llm/base.py](src/llm/base.py)

- `LLMError`: base LLM error.
- `LLMConnectionError`: raised when an endpoint cannot be reached.
- `LLMProviderError`: raised when a provider returns an error.
- `LLMStreamToken(token, done=False)`
- `LLMResponse(text, model, prompt, done_reason=None, raw=None)`

### [src/llm/prompting.py](src/llm/prompting.py)

- `format_context(context_items) -> str`
- `build_user_prompt(query, context_items=None) -> str`

### [src/llm/groq_wrapper.py](src/llm/groq_wrapper.py)

- `GroqStatus(api_reachable, model_available, available_models)`
- `GroqLLM(config=None, timeout_seconds=30.0)`
- `GroqLLM.from_app_config(config) -> GroqLLM`
- `GroqLLM.generate(query, context_items=None, system_prompt=None) -> LLMResponse`
- `GroqLLM.generate_stream(query, context_items=None, system_prompt=None) -> Iterator[LLMStreamToken]`
- `GroqLLM.list_models() -> list[str]`
- `GroqLLM.check_status() -> GroqStatus`

### [src/llm/ollama_wrapper.py](src/llm/ollama_wrapper.py)

- `OllamaStatus(api_reachable, model_available, available_models)`
- `OllamaLLM(config=None, timeout_seconds=120.0)`
- `OllamaLLM.from_app_config(config) -> OllamaLLM`
- `OllamaLLM.list_models() -> list[str]`
- `OllamaLLM.pull_model(model_name=None) -> None`
- `OllamaLLM.check_status() -> OllamaStatus`
- `OllamaLLM.verify_ready(auto_pull=False) -> OllamaStatus`
- `OllamaLLM.generate(query, context_items=None, system_prompt=None) -> LLMResponse`
- `OllamaLLM.generate_stream(query, context_items=None, system_prompt=None) -> Iterator[LLMStreamToken]`

## Voice Input

### [src/voice/mic_capture.py](src/voice/mic_capture.py)

- `MicCaptureDependencyError`
- `AudioFrame(samples, sample_rate)`
- `MicrophoneCapture(sample_rate=16000, channels=1, chunk_size=1024, dtype="float32", ...)`
- `MicrophoneCapture.list_input_devices() -> list[dict[str, Any]]`
- `MicrophoneCapture.record(duration_seconds) -> AudioFrame`
- `MicrophoneCapture.stream_chunks(duration_seconds) -> Iterator[AudioFrame]`

### [src/voice/vad.py](src/voice/vad.py)

- `VADDependencyError`
- `VADSegment(start, end, confidence=None)`
- `SileroVAD(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=30, ...)`
- `SileroVAD.detect_speech(audio, sample_rate) -> list[VADSegment]`
- `SileroVAD.has_speech(audio, sample_rate) -> bool`

### [src/voice/stt.py](src/voice/stt.py)

- `STTDependencyError`
- `STTSegment(text, start, end, avg_logprob=None)`
- `STTResult(text, segments, language=None, confidence=None)`
- `FasterWhisperSTT(model_size="base", device="cpu", compute_type="int8", beam_size=5, ...)`
- `FasterWhisperSTT.transcribe(audio_source, language=None, vad_filter=True) -> STTResult`

### [src/voice/voice_input.py](src/voice/voice_input.py)

- `VoiceInputResult(text, confidence, speech_detected)`
- `VoiceInput(mic_capture=None, vad=None, stt=None)`
- `VoiceInput.capture_and_transcribe(duration_seconds=5.0, push_to_talk=True, language=None) -> VoiceInputResult`

## Text to Speech

### [src/tts/base.py](src/tts/base.py)

- `TTSDependencyError`
- `TTSConfigurationError`
- `TTSBackendError`
- `TTSPlaybackError`
- `TTSResult(text, audio_path, engine, played)`
- `BaseTTSBackend`
- `BaseTTSBackend.synthesize_to_file(text, output_path=None) -> Path`
- `BaseTTSBackend.speak(text, output_path=None, block=True) -> TTSResult`

### [src/tts/pyttsx3_tts.py](src/tts/pyttsx3_tts.py)

- `Pyttsx3TTS`

### [src/tts/gtts_tts.py](src/tts/gtts_tts.py)

- `GTTSTTS`

### [src/tts/kokoro_tts.py](src/tts/kokoro_tts.py)

- `KokoroTTS`

### [src/tts/orchestrator.py](src/tts/orchestrator.py)

- `TTSOrchestrator(config=None, backend_factories=None, output_dir=None)`
- `TTSOrchestrator.from_app_config(config, **kwargs) -> TTSOrchestrator`
- `TTSOrchestrator.speak(text, output_path=None, block=True) -> TTSResult`

## Pipeline

### [src/pipeline.py](src/pipeline.py)

- `IngestionResult(documents=[], chunks=[], errors=[])`
- `PipelineResult(...)`
- `PipelineOrchestrator(...)`
- `PipelineOrchestrator.from_config(config) -> PipelineOrchestrator`
- `PipelineOrchestrator.ingest_documents(source_paths) -> IngestionResult`
- `PipelineOrchestrator.answer(query=None, *, source_paths=None, use_voice=False, voice_duration_seconds=5.0, voice_language=None, top_k=None, system_prompt=None, output_path=None, block=True, ingest_sources=True, raise_on_error=False) -> PipelineResult`

## UI

### [src/ui/gradio_app.py](src/ui/gradio_app.py)

- `create_gradio_app(config_path="config/config.yaml", orchestrator=None)`
- `launch_gradio_app(...)` is the convenience launcher for the Gradio interface.

The UI module also contains helper functions for configuration editing, response formatting, retrieval rendering, and conversation history display. Those helpers are internal to the UI layer and are not part of the stable integration surface.
