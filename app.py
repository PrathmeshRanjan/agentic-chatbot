import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from chatbot import workflow, get_all_threads, get_thread_history, ingest_rag_document
import uuid
import tempfile
import os

# Generate unique thread ID
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the conversation list
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("🤖 Agentic Chatbot")

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

# Per-thread message history: thread_id -> list of (role, content)
if 'chat_histories' not in st.session_state:
    # Restore display history for all persisted threads from the checkpointer
    st.session_state['chat_histories'] = {
        tid: get_thread_history(tid) for tid in st.session_state['chat_threads']
    }

if 'thread_id' not in st.session_state:
    if st.session_state['chat_threads']:
        st.session_state['thread_id'] = st.session_state['chat_threads'][-1]
    else:
        initial_id = generate_thread_id()
        st.session_state['thread_id'] = initial_id
        add_thread(initial_id)
        st.session_state['chat_histories'][initial_id] = []

# Sidebar
st.sidebar.title("My Conversations")

if st.sidebar.button("＋ New Chat", use_container_width=True):
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    st.session_state['chat_histories'][new_id] = []
    st.rerun()

st.sidebar.divider()

for tid in reversed(st.session_state['chat_threads']):
    history = st.session_state['chat_histories'].get(tid, [])
    # Use first user message as the conversation label
    label = next((c for r, c in history if r == "user"), "New Chat")
    if len(label) > 30:
        label = label[:30] + "..."
    is_active = tid == st.session_state['thread_id']
    if st.sidebar.button(label, key=tid, use_container_width=True,
                         type="primary" if is_active else "secondary"):
        st.session_state['thread_id'] = tid
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("📄 Documents")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
if uploaded_pdf is not None:
    if st.session_state.get("ingested_pdf") != uploaded_pdf.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            tmp_path = tmp.name
        try:
            with st.sidebar.spinner("Ingesting PDF..."):
                ingest_rag_document(tmp_path)
            st.session_state["ingested_pdf"] = uploaded_pdf.name
        except Exception as e:
            st.sidebar.error(f"Ingestion failed: {e}")
        finally:
            os.unlink(tmp_path)
    st.sidebar.success(f"Active: **{uploaded_pdf.name}**")

# Render current thread's messages
current_history = st.session_state['chat_histories'].get(st.session_state['thread_id'], [])
for role, content in current_history:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    thread_id = st.session_state['thread_id']
    st.session_state['chat_histories'][thread_id].append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {
            "thread_id": thread_id,
            "pdf_name": st.session_state.get("ingested_pdf")
        }
    }

    initial_state = {"messages": [HumanMessage(content=user_input)]}

    with st.chat_message("assistant"):
        response_placeholder = None
        full_response = ""
        active_tool_calls = {}  # tool_call_id -> st.status widget

        for message_chunk, _ in workflow.stream(
            initial_state, config=config, stream_mode="messages"
        ):
            if isinstance(message_chunk, ToolMessage):
                tc_id = message_chunk.tool_call_id
                if tc_id in active_tool_calls:
                    result = message_chunk.content
                    with active_tool_calls[tc_id]:
                        st.caption(result[:500] + ("..." if len(result) > 500 else ""))
                    active_tool_calls[tc_id].update(state="complete", expanded=False)

            elif isinstance(message_chunk, AIMessageChunk):
                for tc in message_chunk.tool_call_chunks:
                    tc_id = tc.get("id")
                    if tc_id and tc_id not in active_tool_calls:
                        name = tc.get("name", "tool")
                        active_tool_calls[tc_id] = st.status(f"Using tool: **{name}**", expanded=True)
                if message_chunk.content:
                    if response_placeholder is None:
                        response_placeholder = st.empty()
                    full_response += message_chunk.content
                    response_placeholder.markdown(full_response + "▌")

        if response_placeholder is not None:
            response_placeholder.markdown(full_response)

    if full_response:
        st.session_state['chat_histories'][thread_id].append(("assistant", full_response))
