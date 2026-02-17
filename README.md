# Local RAG APP (FastAPI + LlamaIndex + Qdrant)

A memory-efficient RAG application built for Fedora Linux and limited VRAM (RTX 3050 6GB). It uses `Docling` for PDF parsing, `Ollama` (gemma3:4b) for global summarization and inference, and `Qdrant` for hybrid search (Dense + Sparse).

## Prerequisites

1.  **Docker & Docker Compose**: For running the Qdrant vector database.
2.  **Ollama**: Installed and running locally.
    *   Pull the model: `ollama pull gemma3:4b`
    *   Serve it: `ollama serve`
3.  **Python 3.10+** (Tested on 3.13.11)

## Installation

1.  **Create a Virtual Environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Start Qdrant**:
    ```bash
    docker-compose up -d
    ```
    This spins up Qdrant on `localhost:6333` with data persisted to `./qdrant_data`.

2.  **Start the API Server**:
    ```bash
    python3 main.py
    ```
    Or manually with uvicorn:
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    
    *Note: On first run, it will download the BAAI/bge-m3 embedding model.*


### 3. Run the Streamlit Interface
To use the graphical interface:
```bash
streamlit run streamlit_app.py
```
This will open the web UI at `http://localhost:8501`.
After ingestion, Streamlit now shows the full structure-aware markdown and chunk diagnostics (chunk kind, table parent/anchor lineage, visual status) so you can verify ingestion quality without running chat prompts.

### 4. Run Quality Evaluation
```bash
python3 evaluate_rag.py
```
Optional custom case file and output report:
```bash
python3 evaluate_rag.py --cases ./my_eval_cases.json --out ./rag_eval_report.json
```

## Gemini for Chart/Figure Ingestion (Low VRAM)

If your GPU cannot run local Granite Vision chart extraction, keep native chart extraction disabled and use Gemini for picture/chart descriptions instead:

```env
ENABLE_CHART_EXTRACTION=false
ENABLE_PICTURE_DESCRIPTION=true
PICTURE_DESCRIPTION_PROVIDER=gemini_api
PICTURE_DESCRIPTION_MODEL=gemini-2.0-flash
PICTURE_DESCRIPTION_API_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
PICTURE_DESCRIPTION_ONLY_CHARTS=true
GOOGLE_API_KEY=AIza...
```

Notes:
- `ENABLE_CHART_EXTRACTION=true` in current Docling uses a local Granite Vision chart model (`chart2csv`) and may exceed 6GB VRAM.
- `PICTURE_DESCRIPTION_PROVIDER=gemini_api` enriches chart/figure understanding as text for retrieval, but it does not run Docling's native chart-to-table extraction path.

### SOP Tables with Embedded Flow Diagrams

For SOP documents where arrows/decision symbols are drawn inside table cells, enable table visual interpretation:

```env
CHUNKING_SCHEMA_VERSION=2
ENABLE_TABLE_ANCHORS=true
TABLE_ANCHOR_MAX_PER_TABLE=6

ENABLE_TABLE_VISUAL_INTERPRETATION=true
TABLE_VISUAL_INTERPRETATION_PROVIDER=gemini_api
TABLE_VISUAL_INTERPRETATION_MODEL=gemini-2.5-flash-lite
TABLE_VISUAL_INTERPRETATION_API_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
TABLE_VISUAL_ROUTING_MODE=auto_signal
TABLE_VISUAL_DETECTOR_BIAS=precision
TABLE_VISUAL_SIGNAL_THRESHOLD=0.75
TABLE_VISUAL_REQUIRE_STRONG_SIGNALS=true
TABLE_VISUAL_RETRY_ENABLED=true
TABLE_VISUAL_MAX_RETRIES=5
TABLE_VISUAL_BACKOFF_SECONDS=2.0
TABLE_VISUAL_BACKOFF_MULTIPLIER=2.0

TABLE_PARENT_EXPANSION_ENABLED=true
TABLE_PARENT_ALWAYS_FULL_CONTEXT=true
```

Behavior in schema v2:
- Each table is indexed as one atomic `table_parent` chunk (never row-split).
- Additional `table_anchor` chunks are created for retrieval recall.
- Docling table parsing remains the default path for all tables.
- Gemini vision is only called for tables selected by the visual router (flow-like signals), so plain tables stay on Docling output.
- Query retrieval expands anchor hits back to their full table parent before answer generation.
- If Gemini visual calls are temporarily unavailable (e.g. 503), ingestion marks interpretation as pending, stores table image references for retry, and still indexes the table.

## API Usage

### 1. Ingest a PDF
Upload a PDF document to be parsed, summarized, and indexed.

**Endpoint:** `POST /api/ingest`
```bash
curl -X POST "http://localhost:8000/api/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

Ingest multiple PDFs in one request (repeat the `file` field):
```bash
curl -X POST "http://localhost:8000/api/ingest" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/doc1.pdf" \
     -F "file=@/path/to/doc2.pdf" \
     -F "file=@/path/to/doc3.pdf"
```

To include the full normalized markdown in the ingestion response (useful for UI/debug preview):
```bash
curl -X POST "http://localhost:8000/api/ingest?include_markdown=true" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
```

### 2. Chat / Query
Ask a question about the uploaded documents. The system uses Hybrid retrieval (Keyword + Semantic) and re-ranks results using the LLM.

**Endpoint:** `POST /api/chat`
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{ "messages": "What is the global summary of the document?" }'
```

You can also send multi-turn conversation history:
```bash
curl -X POST "http://localhost:8000/api/chat" \
     -H "accept: application/json" \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [
         {"role": "user", "content": "Tell me about OCR support."},
         {"role": "assistant", "content": "It supports OCR extraction from scanned pages."},
         {"role": "user", "content": "What are the main limitations?"}
       ]
     }'
```

## System Architecture

*   **Ingestion Pipeline**: 
    PDF -> Docling (Markdown + structure map) -> markdown normalization -> deterministic `doc_id/chunk_id` generation -> duplicate skip / same-filename replacement -> schema-v2 chunking (`text`, `figure`, atomic `table_parent`, semantic `table_anchor`) -> BAAI/bge-m3 (Dense+Sparse Embedding) -> Qdrant.
*   **Retrieval Pipeline**: 
    User Query -> Hybrid Retrieval (dense+sparse) -> lexical-aware reranking -> table-anchor parent expansion (two-stage retrieval) -> source-labeled context (`[S1]`, `[S2]`) -> grounded answer generation with inline citations.
*   **Memory Management**: 
    Strict Singleton pattern usage for `HuggingFaceEmbedding` and `Ollama` clients to prevent VRAM overflow.
