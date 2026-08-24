import streamlit as st
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

st.set_page_config(page_title="Agentic Chatbot", page_icon="🤖")
st.title("🤖 Agentic Chatbot")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def build_workflow():
    model = init_chat_model("mistralai:mistral-small-latest")

    def chat_node(state: ChatState):
        response = model.invoke(state["messages"])
        return {"messages": [response]}

    checkpoint = InMemorySaver()
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)
    return graph.compile(checkpointer=checkpoint)


workflow = build_workflow()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, content) tuples

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit-session-1"

# Render existing messages
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
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

    st.session_state.chat_history.append(("assistant", full_response))
