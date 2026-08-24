import streamlit as st
from langchain_core.messages import HumanMessage
from chatbot import workflow
import uuid

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
    st.session_state['chat_threads'] = []

# Per-thread message history: thread_id -> list of (role, content)
if 'chat_histories' not in st.session_state:
    st.session_state['chat_histories'] = {}

if 'thread_id' not in st.session_state:
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

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"messages": [HumanMessage(content=user_input)]}

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        for message_chunk, _ in workflow.stream(
            initial_state, config=config, stream_mode="messages"
        ):
            if message_chunk.content:
                full_response += message_chunk.content
                response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state['chat_histories'][thread_id].append(("assistant", full_response))
