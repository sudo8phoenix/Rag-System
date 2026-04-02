"""Application configuration models validated with Pydantic."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VoiceConfig(BaseModel):
    """Speech input configuration."""

    stt_engine: Literal["faster-whisper", "whisper", "speechrecognition"] = (
        "faster-whisper"
    )
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = "base"
    vad_enabled: bool = True
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    language: str = Field(default="en", min_length=1)
    input_mode: Literal["voice", "text"] = "voice"


class FormatChunkingConfig(BaseModel):
    """Per-format chunking overrides with optional format-specific extras."""

    model_config = ConfigDict(extra="allow")

    strategy: str | None = None
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    chunk_unit: Literal["characters", "tokens"] | None = None


class ChunkingConfig(BaseModel):
    """Chunking strategy and limits."""

    strategy: str = "paragraph"
    chunk_size: int = Field(default=500, gt=0)
    chunk_overlap: int = Field(default=50, ge=0)
    chunk_unit: Literal["characters", "tokens"] = "characters"
    min_chunk_size: int = Field(default=30, ge=1)
    max_chunk_size: int = Field(default=2048, gt=0)
    respect_sentence_boundaries: bool = True
    prepend_metadata: bool = True
    per_format: dict[str, FormatChunkingConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_size_bounds(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.min_chunk_size > self.max_chunk_size:
            raise ValueError("min_chunk_size must be less than or equal to max_chunk_size")
        return self


class EmbeddingConfig(BaseModel):
    """Embedding and vector store configuration."""

    model: str = "BAAI/bge-m3"
    device: Literal["cpu", "cuda"] = "cpu"
    vector_store: Literal["faiss", "chroma", "qdrant"] = "chroma"
    chroma_path: str = "./data/chroma_db"


class RetrievalConfig(BaseModel):
    """Retriever tuning parameters."""

    top_k: int = Field(default=5, gt=0)
    hybrid_search: bool = False
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    rerank: bool = True


class LLMConfig(BaseModel):
    """Language model provider and generation settings."""

    provider: Literal["ollama", "openai", "anthropic", "litellm"] = "ollama"
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    api_key: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    system_prompt: str = (
        "You are a helpful assistant. Answer only from the provided context.\n"
        "If you don't know, say so clearly."
    )


class TTSConfig(BaseModel):
    """Text-to-speech engine settings."""

    engine: Literal["pyttsx3", "gtts", "kokoro", "bark", "elevenlabs"] = "pyttsx3"
    voice: str = "male"
    rate: float = Field(default=1.0, gt=0.0)
    volume: float = Field(default=1.0, ge=0.0, le=1.0)
    mute: bool = False


class AgentConfig(BaseModel):
    """Agent orchestration and tool configuration."""

    enabled: bool = True
    max_iterations: int = Field(default=6, gt=0)
    memory_turns: int = Field(default=10, gt=0)
    tools: list[str] = Field(
        default_factory=lambda: [
            "pdf_loader",
            "excel_reader",
            "json_parser",
            "vector_search",
            "calculator",
            "datetime",
        ]
    )


class UIConfig(BaseModel):
    """Web and local UI behavior settings."""

    mode: Literal["gradio", "streamlit", "cli"] = "gradio"
    host: str = "127.0.0.1"
    port: int = Field(default=7860, ge=1, le=65535)
    show_sources: bool = True
    show_agent_trace: bool = False


class AppConfig(BaseModel):
    """Root application config model."""

    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
