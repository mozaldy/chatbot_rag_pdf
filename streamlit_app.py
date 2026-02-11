import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000/api"
REQUEST_TIMEOUT = 60


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
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 200:
            return response.json(), None
        return None, _api_error_message(response)
    except Exception as exc:
        return None, str(exc)

st.set_page_config(page_title="Local RAG Chat", page_icon="🤖", layout="wide")

st.title("🤖 Local RAG (PDF Query)")

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
                    response = requests.post(f"{API_URL}/ingest", files=files, timeout=REQUEST_TIMEOUT)
                    
                    if response.status_code == 200:
                        payload = response.json()
                        succeeded = int(payload.get("succeeded", 0))
                        failed = int(payload.get("failed", 0))
                        total_files = int(payload.get("total_files", len(uploaded_files)))

                        if succeeded > 0:
                            _invalidate_document_cache()
                        if failed == 0:
                            st.success(f"✅ Ingestion completed: {succeeded}/{total_files} files succeeded.")
                        elif succeeded == 0:
                            st.error(f"❌ Ingestion failed for all files ({failed}/{total_files}).")
                        else:
                            st.warning(
                                f"⚠️ Partial ingestion: {succeeded}/{total_files} succeeded, {failed} failed."
                            )

                        st.json(payload)
                    else:
                        st.error(f"❌ Error: {_api_error_message(response)}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")

    st.markdown("---")
    st.header("Database Management")
    if st.button("⚠️ Reset Database", type="primary", help="Deletes all vectors and resets the collection"):
        with st.spinner("Resetting database..."):
            try:
                response = requests.delete(f"{API_URL}/reset", timeout=REQUEST_TIMEOUT)
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
                        timeout=REQUEST_TIMEOUT,
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
                    f"(chunk {source['chunk_index']}, page: {source.get('page_label') or 'n/a'}, "
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
                response = requests.post(f"{API_URL}/chat", json=payload, timeout=REQUEST_TIMEOUT)
                
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
                                    f"(chunk {source['chunk_index']}, page: {source.get('page_label') or 'n/a'}, "
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
