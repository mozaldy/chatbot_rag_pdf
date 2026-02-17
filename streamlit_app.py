import os

import requests
import streamlit as st

# Configuration
def _timeout_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/api")
CONNECT_TIMEOUT = _timeout_from_env("CONNECT_TIMEOUT_SECONDS", 5.0)
REQUEST_TIMEOUT = _timeout_from_env("REQUEST_TIMEOUT_SECONDS", 60.0)
INGEST_REQUEST_TIMEOUT = _timeout_from_env("INGEST_REQUEST_TIMEOUT_SECONDS", 600.0)


def _request_timeout(read_timeout: float) -> tuple[float, float]:
    return CONNECT_TIMEOUT, read_timeout


def _api_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = payload.get("detail")
            if detail:
                return str(detail)
    except Exception:
        pass
    return response.text


def _invalidate_document_cache() -> None:
    st.session_state["documents_snapshot"] = None
    st.session_state["documents_snapshot_error"] = None
    st.session_state["documents_snapshot_max_points"] = None


def _fetch_documents(max_points: int) -> tuple[dict | None, str | None]:
    try:
        response = requests.get(
            f"{API_URL}/documents",
            params={"max_points": max_points},
            timeout=_request_timeout(REQUEST_TIMEOUT),
        )
        if response.status_code == 200:
            return response.json(), None
        return None, _api_error_message(response)
    except Exception as exc:
        return None, str(exc)


def _extract_markdown_previews(payload: dict) -> list[dict]:
    results = payload.get("results", []) if isinstance(payload, dict) else []
    previews: list[dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        markdown = result.get("ingested_markdown")
        if isinstance(markdown, str) and markdown.strip():
            previews.append(
                {
                    "filename": result.get("filename", "unknown.pdf"),
                    "doc_id": result.get("doc_id"),
                    "chunks": result.get("chunks"),
                    "chunking_schema_version": result.get("chunking_schema_version"),
                    "table_parent_count": result.get("table_parent_count", 0),
                    "table_anchor_count": result.get("table_anchor_count", 0),
                    "table_visual_done_count": result.get("table_visual_done_count", 0),
                    "table_visual_pending_count": result.get("table_visual_pending_count", 0),
                    "table_visual_failed_count": result.get("table_visual_failed_count", 0),
                    "table_visual_selected_count": result.get("table_visual_selected_count", 0),
                    "table_visual_skipped_count": result.get("table_visual_skipped_count", 0),
                    "chunk_diagnostics": result.get("chunk_diagnostics", []),
                    "markdown": markdown,
                }
            )
    return previews


def _summarize_ingestion_payload(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return payload
    summarized = dict(payload)
    raw_results = summarized.get("results")
    if not isinstance(raw_results, list):
        return summarized
    compact_results = []
    for item in raw_results:
        if not isinstance(item, dict):
            compact_results.append(item)
            continue
        compact_item = dict(item)
        markdown = compact_item.get("ingested_markdown")
        if isinstance(markdown, str):
            compact_item["ingested_markdown"] = f"<{len(markdown)} chars>"
        compact_results.append(compact_item)
    summarized["results"] = compact_results
    return summarized

st.set_page_config(page_title="Local RAG Chat", page_icon="🤖", layout="wide")

st.title("🤖 Local RAG (PDF Query)")

if "latest_ingestion_markdown_previews" not in st.session_state:
    st.session_state["latest_ingestion_markdown_previews"] = []

# Sidebar for Ingestion
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        if st.button("Ingest Documents"):
            with st.spinner("Ingesting and Indexing... (This may take a while)"):
                try:
                    files = [
                        ("file", (uploaded_file.name, uploaded_file, "application/pdf"))
                        for uploaded_file in uploaded_files
                    ]
                    response = requests.post(
                        f"{API_URL}/ingest",
                        params={"include_markdown": "true"},
                        files=files,
                        timeout=_request_timeout(INGEST_REQUEST_TIMEOUT),
                    )
                    
                    if response.status_code == 200:
                        payload = response.json()
                        succeeded = int(payload.get("succeeded", 0))
                        failed = int(payload.get("failed", 0))
                        total_files = int(payload.get("total_files", len(uploaded_files)))

                        if succeeded > 0:
                            _invalidate_document_cache()
                        st.session_state["latest_ingestion_markdown_previews"] = _extract_markdown_previews(payload)
                        if failed == 0:
                            st.success(f"✅ Ingestion completed: {succeeded}/{total_files} files succeeded.")
                        elif succeeded == 0:
                            st.error(f"❌ Ingestion failed for all files ({failed}/{total_files}).")
                        else:
                            st.warning(
                                f"⚠️ Partial ingestion: {succeeded}/{total_files} succeeded, {failed} failed."
                            )

                        st.json(_summarize_ingestion_payload(payload))
                    else:
                        st.error(f"❌ Error: {_api_error_message(response)}")
                except requests.exceptions.ReadTimeout:
                    st.error(
                        "❌ Ingestion request timed out. "
                        f"Set `INGEST_REQUEST_TIMEOUT_SECONDS` higher than {INGEST_REQUEST_TIMEOUT:g} "
                        "for large PDFs."
                    )
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")

    st.markdown("---")
    st.header("Database Management")
    if st.button("⚠️ Reset Database", type="primary", help="Deletes all vectors and resets the collection"):
        with st.spinner("Resetting database..."):
            try:
                response = requests.delete(
                    f"{API_URL}/reset",
                    timeout=_request_timeout(REQUEST_TIMEOUT),
                )
                if response.status_code == 200:
                    _invalidate_document_cache()
                    st.success("✅ Database Reset Successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Error: {_api_error_message(response)}")
            except Exception as e:
                st.error(f"❌ Connection Error: {e}")

    st.markdown("---")
    st.header("Documents")
    max_points = int(
        st.number_input(
            "List Scan Limit",
            min_value=100,
            max_value=100000,
            value=5000,
            step=100,
            help="Maximum Qdrant points scanned when building the document list.",
        )
    )
    if st.button("Refresh Document List"):
        _invalidate_document_cache()

    cached_max_points = st.session_state.get("documents_snapshot_max_points")
    if (
        st.session_state.get("documents_snapshot") is None
        or st.session_state.get("documents_snapshot_error") is not None
        or cached_max_points != max_points
    ):
        data, error = _fetch_documents(max_points=max_points)
        st.session_state["documents_snapshot"] = data
        st.session_state["documents_snapshot_error"] = error
        st.session_state["documents_snapshot_max_points"] = max_points

    document_error = st.session_state.get("documents_snapshot_error")
    document_data = st.session_state.get("documents_snapshot")

    if document_error:
        st.error(f"❌ Failed to load documents: {document_error}")
    elif document_data is not None:
        documents = document_data.get("documents", [])
        st.caption(
            f"Documents: {document_data.get('total_documents', 0)} | "
            f"Chunks: {document_data.get('total_chunks', 0)} | "
            f"Scanned: {document_data.get('scanned_points', 0)}"
        )
        if document_data.get("truncated"):
            st.warning("Document list is truncated. Increase List Scan Limit to see more.")

        if documents:
            table_rows = [
                {
                    "filename": doc.get("filename", "unknown"),
                    "doc_id": doc.get("doc_id", "unknown"),
                    "chunks": doc.get("chunks", 0),
                    "max_chunk_index": doc.get("max_chunk_index"),
                    "table_parent_chunks": doc.get("table_parent_chunks", 0),
                    "table_anchor_chunks": doc.get("table_anchor_chunks", 0),
                    "schema_versions": ", ".join(str(v) for v in (doc.get("schema_versions") or [])) or "n/a",
                }
                for doc in documents
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            option_map = {
                f"{doc['filename']} | {doc['doc_id']} | chunks={doc['chunks']}": doc["doc_id"]
                for doc in documents
            }

            unique_filenames = sorted({doc["filename"] for doc in documents})
            selected_filename = st.selectbox(
                "Select Filename",
                options=unique_filenames,
                key="filename_delete_select",
            )
            if st.button("Delete By Filename"):
                try:
                    response = requests.delete(
                        f"{API_URL}/documents/by-filename",
                        params={"filename": selected_filename},
                        timeout=_request_timeout(REQUEST_TIMEOUT),
                    )
                    if response.status_code == 200:
                        st.success(f"✅ Deleted chunks for filename '{selected_filename}'")
                        _invalidate_document_cache()
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {_api_error_message(response)}")
                except Exception as exc:
                    st.error(f"❌ Connection Error: {exc}")
        else:
            st.info("No indexed documents yet. Ingest a PDF to populate this list.")

preview_items = st.session_state.get("latest_ingestion_markdown_previews", [])
if preview_items:
    st.subheader("Latest Ingestion Markdown")
    st.caption(
        "These are the full structure-aware markdown outputs used for indexing. "
        "Use this to verify ingestion quality and chunking behavior without prompting the LLM."
    )
    if st.button("Clear Markdown Preview"):
        st.session_state["latest_ingestion_markdown_previews"] = []
        st.rerun()
    for idx, item in enumerate(preview_items):
        with st.expander(
            f"{item['filename']} | doc_id={item.get('doc_id') or 'n/a'} | chunks={item.get('chunks')}",
            expanded=False,
        ):
            st.caption(
                " | ".join(
                    [
                        f"schema_v={item.get('chunking_schema_version') or 'n/a'}",
                        f"table_parent={item.get('table_parent_count', 0)}",
                        f"table_anchor={item.get('table_anchor_count', 0)}",
                        f"table_visual_done={item.get('table_visual_done_count', 0)}",
                        f"pending={item.get('table_visual_pending_count', 0)}",
                        f"failed={item.get('table_visual_failed_count', 0)}",
                        f"selected={item.get('table_visual_selected_count', 0)}",
                        f"skipped={item.get('table_visual_skipped_count', 0)}",
                    ]
                )
            )

            chunk_diagnostics = item.get("chunk_diagnostics") or []
            if chunk_diagnostics:
                only_table_parents = st.checkbox(
                    "Show only table_parent chunks",
                    value=False,
                    key=f"table_parent_only_{idx}",
                )
                only_selected_for_visual = st.checkbox(
                    "Show only Gemini-selected tables",
                    value=False,
                    key=f"table_visual_selected_only_{idx}",
                )
                visible_rows = chunk_diagnostics
                if only_table_parents:
                    visible_rows = [
                        row for row in visible_rows if row.get("chunk_kind") == "table_parent"
                    ]
                if only_selected_for_visual:
                    visible_rows = [
                        row for row in visible_rows if bool(row.get("table_visual_selected", False))
                    ]
                st.dataframe(visible_rows, use_container_width=True, hide_index=True)

            st.code(item["markdown"], language="markdown")

# Main Chat Interface
st.subheader("Chat with your Data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources if they exist in the message
        if "sources" in message:
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            for source in message["sources"]:
                with st.expander(
                    f"{source['id']} • {source['filename']} "
                    f"(chunk {source['chunk_index']}, kind: {source.get('chunk_kind') or source.get('content_type') or 'n/a'}, "
                    f"page: {source.get('page_label') or 'n/a'}, "
                    f"section: {source.get('section_title') or 'n/a'}, score: {source['score']})"
                ):
                    st.markdown(source['text'])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        
        try:
            with st.spinner("Thinking..."):
                chat_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages
                    if msg.get("role") in {"user", "assistant"} and msg.get("content")
                ]
                payload = {"messages": chat_messages}
                response = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=_request_timeout(REQUEST_TIMEOUT),
                )
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", "No response")
                    sources = data.get("sources", [])
                    
                    # Display the answer
                    message_placeholder.markdown(answer)
                    
                    # Display interactive sources
                    if sources:
                        with sources_placeholder.container():
                            st.markdown("---")
                            st.markdown("**📚 Sources:**")
                            for source in sources:
                                with st.expander(
                                    f"{source['id']} • {source['filename']} "
                                    f"(chunk {source['chunk_index']}, kind: {source.get('chunk_kind') or source.get('content_type') or 'n/a'}, "
                                    f"page: {source.get('page_label') or 'n/a'}, "
                                    f"section: {source.get('section_title') or 'n/a'}, score: {source['score']})",
                                    expanded=False
                                ):
                                    st.markdown(source['text'])
                    
                    # Store in session with sources for history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    error_msg = f"Error: {_api_error_message(response)}"
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
        except Exception as e:
            error_msg = f"Connection Error: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
