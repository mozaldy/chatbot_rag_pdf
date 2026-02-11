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

### 4. Run Quality Evaluation
```bash
python3 evaluate_rag.py
```
Optional custom case file and output report:
```bash
python3 evaluate_rag.py --cases ./my_eval_cases.json --out ./rag_eval_report.json
```

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
    PDF -> Docling (Markdown) -> Markdown normalization (dehyphenation/boilerplate cleanup) -> deterministic `doc_id/chunk_id` generation -> duplicate skip / same-filename replacement -> chunking -> BAAI/bge-m3 (Dense+Sparse Embedding) -> Qdrant.
*   **Retrieval Pipeline**: 
    User Query -> Hybrid Retrieval (dense+sparse) -> lexical-aware reranking -> source-labeled context (`[S1]`, `[S2]`) -> grounded answer generation with inline citations.
*   **Memory Management**: 
    Strict Singleton pattern usage for `HuggingFaceEmbedding` and `Ollama` clients to prevent VRAM overflow.
