from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

model = init_chat_model("mistralai:mistral-small-latest")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {'messages': [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

workflow = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)

def get_thread_history(thread_id):
    data = checkpoint.get({'configurable': {'thread_id': thread_id}})
    if not data:
        return []
    history = []
    for msg in data.get('channel_values', {}).get('messages', []):
        if msg.type == 'human':
            history.append(('user', msg.content))
        elif msg.type == 'ai':
            history.append(('assistant', msg.content))
    return history

# thread_id = 1
# config = {'configurable': {'thread_id': thread_id}}

# while True:
#     user_message = input('Enter your message: ')
#     if user_message.strip().lower() in ['exit', 'quit', 'bye']:
#         break
    
#     initial_state = {
#         'messages': [HumanMessage(content=user_message)]
#     }

#     # Non-streaming repsonse
#     # response = workflow.invoke(initial_state, config=config)
#     # print('AI: ', response['messages'][-1].content)

#     # Streaming response
#     for message_chunk, _ in workflow.stream(initial_state, config=config, stream_mode='messages'):
#         if message_chunk.content:
#             print(message_chunk.content, end=" ", flush=True)
    