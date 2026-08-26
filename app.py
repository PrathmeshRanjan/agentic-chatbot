import streamlit as st
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from langgraph.types import Command
from chatbot import workflow, get_all_threads, get_thread_history, ingest_rag_document
import uuid
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Light Pastel UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 920px;
    }

    /* Hero Header: Premium Pastel Gradient */
    .hero-card {
        background: linear-gradient(135deg, #fdfbf7 0%, #f4f6fb 45%, #f5f3ff 100%);
        border: 1px solid rgba(226, 232, 240, 0.85);
        border-radius: 20px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 30px -6px rgba(148, 163, 184, 0.12), 0 4px 12px -2px rgba(148, 163, 184, 0.06);
    }
    
    .hero-title {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
        background: linear-gradient(135deg, #1e293b 0%, #4338ca 60%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 0.94rem;
        color: #64748b;
        margin: 0;
        line-height: 1.5;
    }

    /* Pastel Capability Badges */
    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 1rem;
    }

    .badge-item {
        font-size: 0.76rem;
        font-weight: 600;
        padding: 0.28rem 0.65rem;
        border-radius: 30px;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        transition: transform 0.15s ease;
    }
    .badge-item:hover {
        transform: translateY(-1px);
    }

    .badge-mistral { background: #faf5ff; color: #7e22ce; border: 1px solid #e9d5ff; }
    .badge-search  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    .badge-rag     { background: #eff6ff; color: #1e40af; border: 1px solid #bfdbfe; }
    .badge-stock   { background: #ecfeff; color: #155e75; border: 1px solid #a5f3fc; }
    .badge-hitl    { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .badge-weather { background: #fefce8; color: #854d0e; border: 1px solid #fef08a; }
    .badge-calc    { background: #fff1f2; color: #9f1239; border: 1px solid #fecdd3; }

    /* Modern HITL Card */
    .hitl-container {
        background: linear-gradient(135deg, #fffdf5 0%, #fef3c7 100%);
        border: 1.5px solid #fcd34d;
        border-radius: 18px;
        padding: 1.3rem 1.6rem;
        margin: 1.2rem 0;
        box-shadow: 0 10px 25px -4px rgba(245, 158, 11, 0.15);
    }

    .hitl-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #92400e;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.4rem;
    }

    .hitl-body {
        font-size: 0.95rem;
        font-weight: 500;
        color: #78350f;
        margin-bottom: 0.9rem;
        line-height: 1.5;
    }

    /* Starter Suggestions */
    .starter-header {
        font-size: 0.92rem;
        font-weight: 600;
        color: #475569;
        margin: 1.2rem 0 0.6rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fbfcfe 0%, #f8fafc 100%);
        border-right: 1px solid #edf2f7;
    }

    /* Streamlit Chat Messages Soft Styling */
    .stChatMessage {
        border-radius: 18px !important;
        margin-bottom: 0.9rem !important;
        padding: 1rem 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Generate unique thread ID
def generate_thread_id():
    return str(uuid.uuid4())

# Add a new thread ID to the conversation list
def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def get_tool_icon(tool_name: str) -> str:
    """Return an intuitive emoji icon for each agent tool."""
    name_lower = tool_name.lower()
    if "search" in name_lower or "tavily" in name_lower:
        return "🔍"
    elif "calculator" in name_lower or "calc" in name_lower:
        return "🧮"
    elif "stock_price" in name_lower or "price" in name_lower:
        return "📈"
    elif "purchase" in name_lower or "buy" in name_lower:
        return "🛡️"
    elif "weather" in name_lower:
        return "⛅"
    elif "rag" in name_lower or "doc" in name_lower:
        return "📄"
    return "⚙️"

def stream_graph_response(stream_input, config, thread_id):
    """Execute graph streaming, rendering real-time tokens, status containers, and final responses."""
    with st.chat_message("assistant", avatar="✨"):
        response_placeholder = None
        full_response = ""
        active_tool_calls = {}  # tool_call_id -> st.status widget

        for message_chunk, _ in workflow.stream(
            stream_input, config=config, stream_mode="messages"
        ):
            # 1. Handle tool execution results
            if isinstance(message_chunk, ToolMessage):
                tc_id = message_chunk.tool_call_id
                result = str(message_chunk.content)
                if tc_id in active_tool_calls:
                    with active_tool_calls[tc_id]:
                        st.caption(result[:600] + ("..." if len(result) > 600 else ""))
                    active_tool_calls[tc_id].update(
                        label=f"Completed: **{active_tool_calls[tc_id]._label.split('**')[1]}**",
                        state="complete",
                        expanded=False
                    )
                else:
                    tool_name = getattr(message_chunk, "name", "tool") or "tool"
                    icon = get_tool_icon(tool_name)
                    tool_status = st.status(f"{icon} Finished: **{tool_name}**", state="complete", expanded=False)
                    with tool_status:
                        st.caption(result[:600] + ("..." if len(result) > 600 else ""))

            # 2. Handle LLM message tokens and tool call requests
            elif isinstance(message_chunk, AIMessageChunk):
                for tc in message_chunk.tool_call_chunks:
                    tc_id = tc.get("id")
                    if tc_id and tc_id not in active_tool_calls:
                        name = tc.get("name", "tool")
                        icon = get_tool_icon(name)
                        active_tool_calls[tc_id] = st.status(f"{icon} Using tool: **{name}**...", expanded=True)
                        active_tool_calls[tc_id]._label = f"{icon} **{name}**"

                if message_chunk.content:
                    if response_placeholder is None:
                        response_placeholder = st.empty()
                    full_response += message_chunk.content
                    response_placeholder.markdown(full_response + "▌")

        if response_placeholder is not None:
            response_placeholder.markdown(full_response)

    if full_response:
        st.session_state['chat_histories'][thread_id].append(("assistant", full_response))

# Session State Initialization
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = get_all_threads()

if 'chat_histories' not in st.session_state:
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

# Sidebar: Conversations & Knowledge Base
with st.sidebar:
    st.markdown("### 💬 Conversations")
    
    if st.button("＋ New Chat", use_container_width=True, type="primary"):
        new_id = generate_thread_id()
        st.session_state['thread_id'] = new_id
        add_thread(new_id)
        st.session_state['chat_histories'][new_id] = []
        st.rerun()

    st.markdown("<div style='margin-top: 0.4rem;'></div>", unsafe_allow_html=True)

    # Thread List
    for tid in reversed(st.session_state['chat_threads']):
        history = st.session_state['chat_histories'].get(tid, [])
        label = next((c for r, c in history if r == "user"), "New Chat")
        if len(label) > 26:
            label = label[:26] + "..."
        is_active = tid == st.session_state['thread_id']
        
        btn_label = f"💬 {label}" if not is_active else f"✨ {label}"
        if st.button(
            btn_label,
            key=tid,
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state['thread_id'] = tid
            st.rerun()

    st.divider()

    # Document RAG Ingestion Section
    st.markdown("### 📄 Knowledge Base")
    uploaded_pdf = st.file_uploader(
        "Upload PDF for RAG retrieval",
        type=["pdf"],
        help="Upload a PDF document. The agent will index and retrieve information from it via FAISS."
    )
    
    if uploaded_pdf is not None:
        if st.session_state.get("ingested_pdf") != uploaded_pdf.name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Indexing document into FAISS..."):
                    ingest_rag_document(tmp_path)
                st.session_state["ingested_pdf"] = uploaded_pdf.name
                st.success(f"Indexed: **{uploaded_pdf.name}**")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                os.unlink(tmp_path)
        else:
            st.info(f"Active: **{uploaded_pdf.name}**", icon="📄")

# Main View Hero Card
st.markdown("""
<div class="hero-card">
    <div class="hero-title">✨ Agentic AI Assistant</div>
    <div class="hero-subtitle">Multi-tool ReAct intelligence with real-time vector search and human-supervised safeguards.</div>
    <div class="badge-bar">
        <span class="badge-item badge-mistral">🧠 Mistral LLM</span>
        <span class="badge-item badge-search">🔍 Tavily Search</span>
        <span class="badge-item badge-rag">📄 FAISS Vector RAG</span>
        <span class="badge-item badge-stock">📈 Market Quotes</span>
        <span class="badge-item badge-hitl">🛡️ HITL Trading Guard</span>
        <span class="badge-item badge-weather">⛅ Weather API</span>
        <span class="badge-item badge-calc">🧮 Safe Math</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Render Chat History
current_history = st.session_state['chat_histories'].get(st.session_state['thread_id'], [])

# Welcome Quick Starters for Empty Conversations
if not current_history:
    st.markdown("<div class='starter-header'>💡 Try asking or testing an agent action:</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    prompt_to_trigger = None
    with col1:
        if st.button("📈 Check latest AAPL stock quote", use_container_width=True):
            prompt_to_trigger = "What is the latest stock price of AAPL?"
        if st.button("🛡️ Buy 10 shares of TSLA (Trigger HITL)", use_container_width=True):
            prompt_to_trigger = "Purchase 10 shares of TSLA"
    with col2:
        if st.button("⛅ Current weather in Tokyo, Japan", use_container_width=True):
            prompt_to_trigger = "What is the current weather in Tokyo?"
        if st.button("🧮 Compute 25 * 40 + math.sqrt(1024)", use_container_width=True):
            prompt_to_trigger = "Calculate 25 * 40 + math.sqrt(1024)"

    if prompt_to_trigger:
        thread_id = st.session_state['thread_id']
        st.session_state['chat_histories'][thread_id].append(("user", prompt_to_trigger))
        config = {
            "configurable": {
                "thread_id": thread_id,
                "pdf_name": st.session_state.get("ingested_pdf")
            }
        }
        initial_state = {"messages": [HumanMessage(content=prompt_to_trigger)]}
        stream_graph_response(initial_state, config, thread_id)
        st.rerun()

else:
    for role, content in current_history:
        avatar = "👤" if role == "user" else "✨"
        with st.chat_message(role, avatar=avatar):
            st.markdown(content)

# Runnable configuration for current thread
config = {
    "configurable": {
        "thread_id": st.session_state['thread_id'],
        "pdf_name": st.session_state.get("ingested_pdf")
    }
}

# Inspect if thread is paused at a Human-in-the-Loop (HITL) interrupt
thread_state = workflow.get_state(config)
pending_interrupt_prompt = None

if thread_state.tasks:
    for task in thread_state.tasks:
        if task.interrupts:
            pending_interrupt_prompt = task.interrupts[0].value
            break

# Render Modern HITL Approval Container if interrupt is active
if pending_interrupt_prompt:
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown(f"""
        <div class="hitl-container">
            <div class="hitl-header">🛡️ Action Authorization Required</div>
            <div class="hitl-body">{pending_interrupt_prompt}</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Approve Action", key=f"approve_{st.session_state['thread_id']}", type="primary", use_container_width=True):
                st.session_state['chat_histories'][st.session_state['thread_id']].append(("user", "Approved (yes)"))
                stream_graph_response(Command(resume="yes"), config, st.session_state['thread_id'])
                st.rerun()
        with col2:
            if st.button("❌ Reject Action", key=f"reject_{st.session_state['thread_id']}", type="secondary", use_container_width=True):
                st.session_state['chat_histories'][st.session_state['thread_id']].append(("user", "Rejected (no)"))
                stream_graph_response(Command(resume="no"), config, st.session_state['thread_id'])
                st.rerun()

# Chat Input (disabled during pending HITL action)
user_input = st.chat_input(
    "Type your message..." if not pending_interrupt_prompt else "Action pending authorization above...",
    disabled=bool(pending_interrupt_prompt)
)

if user_input:
    thread_id = st.session_state['thread_id']
    st.session_state['chat_histories'][thread_id].append(("user", user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    initial_state = {"messages": [HumanMessage(content=user_input)]}
    stream_graph_response(initial_state, config, thread_id)

    # Check if execution paused on an interrupt and trigger rerun to present HITL buttons
    state_after = workflow.get_state(config)
    if state_after.tasks and any(task.interrupts for task in state_after.tasks):
        st.rerun()


