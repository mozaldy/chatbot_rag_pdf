from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    API_TITLE: str = "Local RAG API"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_TIMEOUT: int = 8
    COLLECTION_NAME: str = "pdf_rag"
    DOCUMENT_LIST_MAX_POINTS: int = 10000
    DOCUMENT_LIST_PAGE_SIZE: int = 256

    # Models
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    # Chunking Configuration
    CHUNK_SIZE: int = 768  # Tokens per chunk (increased for code-heavy docs)
    CHUNK_OVERLAP: int = 200  # Overlap between chunks for continuity
    CHUNKING_SCHEMA_VERSION: int = 2
    ENABLE_TABLE_ANCHORS: bool = True
    TABLE_ANCHOR_MAX_PER_TABLE: int = 6

    # PDF Processing
    ENABLE_OCR: bool = True  # Set to True if you have scanned docs or mixed content
    ENABLE_PICTURE_DESCRIPTION: bool = True  # SmolVLM-256M describes charts/pictures as text
    PICTURE_DESCRIPTION_PROVIDER: str = "local"  # Options: local, gemini_api
    PICTURE_DESCRIPTION_MODEL: str = "gemini-2.0-flash"
    PICTURE_DESCRIPTION_API_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    PICTURE_DESCRIPTION_TIMEOUT: float = 30.0
    PICTURE_DESCRIPTION_CONCURRENCY: int = 1
    PICTURE_DESCRIPTION_ONLY_CHARTS: bool = True
    PICTURE_DESCRIPTION_PROMPT: str = (
        "Describe this chart or figure for document retrieval. "
        "Include title, axes, legend, trend direction, and notable numeric values if visible."
    )
    ENABLE_TABLE_VISUAL_INTERPRETATION: bool = True
    TABLE_VISUAL_INTERPRETATION_PROVIDER: str = "gemini_api"  # Options: gemini_api
    TABLE_VISUAL_INTERPRETATION_MODEL: str = "gemini-2.5-flash-lite"
    TABLE_VISUAL_INTERPRETATION_API_URL: str = (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    TABLE_VISUAL_INTERPRETATION_TIMEOUT: float = 45.0
    TABLE_VISUAL_INTERPRETATION_CONCURRENCY: int = 2
    TABLE_VISUAL_INTERPRETATION_MAX_CHARS: int = 1600
    TABLE_VISUAL_ROUTING_MODE: str = "auto_signal"  # Options: auto_signal, always_on
    TABLE_VISUAL_DETECTOR_BIAS: str = "precision"  # Options: precision, balanced, recall
    TABLE_VISUAL_SIGNAL_THRESHOLD: float = 0.75
    TABLE_VISUAL_REQUIRE_STRONG_SIGNALS: bool = True
    TABLE_VISUAL_RETRY_ENABLED: bool = True
    TABLE_VISUAL_MAX_RETRIES: int = 5
    TABLE_VISUAL_BACKOFF_SECONDS: float = 2.0
    TABLE_VISUAL_BACKOFF_MULTIPLIER: float = 2.0
    TABLE_VISUAL_INTERPRETATION_PROMPT: str = (
        "You are reading a table image from a Standard Operating Procedure (SOP). "
        "Identify visual process flow semantics that are not captured by plain table cells, "
        "including arrows, connectors, decision symbols, and handoffs across roles. "
        "Return concise markdown bullets only. Do not hallucinate."
    )
    ENABLE_CHART_EXTRACTION: bool = False  # GraniteVision (~6GB) too large for 6GB GPU
    MAX_TABLE_CHUNK_TOKENS: int = 1500  # Legacy fallback, no longer used for row splitting in schema v2
    TABLE_FORMER_MODE: str = "accurate"  # "accurate" or "fast"

    # Retrieval Configuration
    # alpha: 0.0 = pure keyword/sparse, 1.0 = pure semantic/dense
    # Lower values favor exact text matching, higher values favor meaning
    HYBRID_ALPHA: float = 0.3  # Favor keyword matching for exact text retrieval
    DENSE_TOP_K: int = 24
    SPARSE_TOP_K: int = 24
    FUSION_MODE: str = "weighted_rrf"  # Options: weighted_rrf, rrf
    FUSION_RRF_K: int = 60
    RERANKER_TYPE: str = "cross_encoder"  # Options: lexical, cross_encoder
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_USE_FP16: bool = True
    RERANKER_BATCH_SIZE: int = 16
    RERANKER_MAX_LENGTH: int = 512
    RERANK_CANDIDATES_K: int = 24
    RERANK_TOP_K: int = 8
    RERANK_MIN_SCORE: float | None = 0.001
    MAX_CONTEXT_CHARS_PER_CHUNK: int = 1800
    TABLE_PARENT_EXPANSION_ENABLED: bool = True
    TABLE_PARENT_ALWAYS_FULL_CONTEXT: bool = True
    MAX_SOURCES: int = 5  # Maximum number of unique sources to display
    CONTEXT_DIVERSITY_ENABLED: bool = True
    MAX_SOURCES_PER_DOC: int = 2
    REQUIRE_VALID_CITATIONS: bool = True
    RETRIEVAL_DEBUG_LOGS: bool = False

    # Query Rewrite
    QUERY_REWRITE_ENABLED: bool = True
    QUERY_REWRITE_MODE: str = "rule"  # Options: rule, llm
    QUERY_REWRITE_WEIGHT: float = 0.6  # Weight for rewritten query during multi-query fusion
    QUERY_REWRITE_MAX_TERMS: int = 12
    QUERY_REWRITE_MAX_CHARS: int = 200

    # Conversation-aware Retrieval
    CONVERSATION_HISTORY_MAX_MESSAGES: int = 8
    CONVERSATION_STANDALONE_QUERY_ENABLED: bool = True
    CONVERSATION_STANDALONE_MAX_CHARS: int = 280

    UPSERT_BATCH_SIZE: int = 20

    # LLM Configuration (Main/Inference)
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "gemma3:4b"

    # LLM Configuration (Fast/Enrichment)
    FAST_LLM_PROVIDER: str = "ollama"
    FAST_LLM_MODEL: str = "gemma3:4b"

    # Provider Specifics
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_TIMEOUT: float = 120.0

    # Cloud Providers (Optional)
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()
