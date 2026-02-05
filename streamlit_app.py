import streamlit as st
import requests
import json

# Configuration
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Local RAG Chat", page_icon="🤖", layout="wide")

st.title("🤖 Local RAG (PDF Query)")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Ingesting and Indexing... (This may take a while)"):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                    response = requests.post(f"{API_URL}/ingest", files=files)
                    
                    if response.status_code == 200:
                        st.success("✅ Ingestion Successful!")
                        st.json(response.json())
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection Error: {e}")

# Main Chat Interface
st.subheader("Chat with your Data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with st.spinner("Thinking..."):
                payload = {"messages": prompt}
                response = requests.post(f"{API_URL}/chat", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("response", "No response")
                    sources = data.get("sources", [])
                    
                    full_response = answer
                    if sources:
                        full_response += "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in set(sources)])
                    
                    message_placeholder.markdown(full_response)
                else:
                    full_response = f"Error: {response.text}"
                    message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"Connection Error: {e}"
            message_placeholder.error(full_response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
