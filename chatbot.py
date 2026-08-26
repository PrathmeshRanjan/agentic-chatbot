from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.types import interrupt, Command
from typing import Any
import sqlite3
import requests
import os
import math

load_dotenv()

model = init_chat_model("mistralai:mistral-small-latest")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)    

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
            folder_path=DB_PATH,
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return retriever

search_tool = TavilySearch(
    max_results=5,
    topic='general',
    search_depth='advanced'
)

@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retrieve PDF content.
    """

    documents = get_retriever().invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10 * 5
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}"
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return (
            "Weather API key is missing. "
            "Set the OPENWEATHER_API_KEY environment variable."
        )

    try:
        # Step 1: Convert the location name into latitude and longitude
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"

        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": api_key,
        }

        geo_response = requests.get(
            geocoding_url,
            params=geocoding_params,
            timeout=10,
        )
        geo_response.raise_for_status()

        locations: list[dict[str, Any]] = geo_response.json()

        if not locations:
            return f"Could not find the location: {location}"

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        # Step 2: Get current weather using latitude and longitude
        weather_url = "https://api.openweathermap.org/data/2.5/weather"

        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=10,
        )
        weather_response.raise_for_status()

        weather_data = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        pressure = weather_data["main"]["pressure"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data.get("wind", {}).get("speed", "N/A")
        visibility_meters = weather_data.get("visibility")

        visibility_km = (
            round(visibility_meters / 1000, 1)
            if visibility_meters is not None
            else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)

        if country:
            location_parts.append(country)

        display_location = ", ".join(location_parts)

        return (
            f"Current weather in {display_location}:\n"
            f"- Condition: {description.title()}\n"
            f"- Temperature: {temperature}°C\n"
            f"- Feels like: {feels_like}°C\n"
            f"- Humidity: {humidity}%\n"
            f"- Pressure: {pressure} hPa\n"
            f"- Wind speed: {wind_speed} m/s\n"
            f"- Visibility: {visibility_km} km"
        )

    except requests.Timeout:
        return "The weather service request timed out. Please try again."

    except requests.HTTPError as error:
        status_code = error.response.status_code if error.response else "unknown"

        if status_code == 401:
            return "The OpenWeather API key is invalid or inactive."

        return f"Weather API returned an HTTP error: {status_code}"

    except requests.RequestException as error:
        return f"Could not connect to the weather service: {error}"

    except (KeyError, TypeError, ValueError) as error:
        return f"Unexpected weather API response: {error}"

tools = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool, purchase_stock]

model_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that can answer directly or call an appropriate tool."""

    pdf_name = config.get("configurable", {}).get("pdf_name")
    pdf_context = (
        f"A PDF document named '{pdf_name}' has been uploaded and indexed. "
        "Always call `rag_tool` when the user asks about this document, 'the PDF', "
        "'the file', or any factual question that might be answered by it."
    ) if pdf_name else (
        "No PDF is currently uploaded. If the user asks about a document or PDF, "
        "tell them to upload one using the sidebar."
    )

    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"

            f"{pdf_context}\n\n"

            "Other tool usage instructions:\n"
            "- Use `search_tool` for current events, recent information, or information "
            "that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex "
            "expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "- Use `get_current_weather` when the user asks about current weather for a location.\n\n"

            "Answer general questions directly when no tool is required. "
            "Do not invent information — if a tool is available for the query, use it. "
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state["messages"] # Unpacking the nesting to a flat list
    ]

    response = model_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools) # Executes tool calls

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)

checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'chat_node')
# If the LLM asked for a tool, go to ToolNode; else no
graph.add_conditional_edges('chat_node', tools_condition) 
graph.add_edge('tools', 'chat_node')

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
            if msg.content:
                history.append(('assistant', msg.content))
    return history

# =====================================================================
# CLI Execution Examples with Human-in-the-Loop (HITL) Support
# =====================================================================
#
# --- Example 1: Streaming CLI with HITL ---
#
# thread_id = "cli_session_1"
# config = {"configurable": {"thread_id": thread_id}}
#
# while True:
#     user_message = input("\nUser: ").strip()
#     if user_message.lower() in ["exit", "quit", "bye"]:
#         print("Exiting conversation.")
#         break
#     if not user_message:
#         continue
#
#     initial_state = {"messages": [HumanMessage(content=user_message)]}
#
#     print("AI: ", end="", flush=True)
#     for message_chunk, _ in workflow.stream(
#         initial_state, config=config, stream_mode="messages"
#     ):
#         if message_chunk.content:
#             print(message_chunk.content, end="", flush=True)
#
#     # Check if the graph paused on a Human-in-the-Loop interrupt
#     state = workflow.get_state(config)
#     while state.tasks and any(task.interrupts for task in state.tasks):
#         for task in state.tasks:
#             for intr in task.interrupts:
#                 print(f"\n\n[HITL ACTION REQUIRED]: {intr.value}")
#                 decision = input("Enter decision (yes/no): ").strip()
#
#                 print("AI: ", end="", flush=True)
#                 # Resume execution using Command(resume=decision)
#                 for message_chunk, _ in workflow.stream(
#                     Command(resume=decision), config=config, stream_mode="messages"
#                 ):
#                     if message_chunk.content:
#                         print(message_chunk.content, end="", flush=True)
#
#         state = workflow.get_state(config)
#     print()
#
#
# --- Example 2: Non-Streaming CLI with HITL ---
#
# thread_id = "cli_session_2"
# config = {"configurable": {"thread_id": thread_id}}
#
# while True:
#     user_message = input("\nUser: ").strip()
#     if user_message.lower() in ["exit", "quit", "bye"]:
#         break
#     if not user_message:
#         continue
#
#     initial_state = {"messages": [HumanMessage(content=user_message)]}
#     response = workflow.invoke(initial_state, config=config)
#
#     # Print any assistant message generated before interrupt/completion
#     if response.get("messages") and response["messages"][-1].content:
#         print("AI:", response["messages"][-1].content)
#
#     # Check for HITL interrupts
#     state = workflow.get_state(config)
#     while state.tasks and any(task.interrupts for task in state.tasks):
#         for task in state.tasks:
#             for intr in task.interrupts:
#                 print(f"\n[HITL ACTION REQUIRED]: {intr.value}")
#                 decision = input("Enter decision (yes/no): ").strip()
#                 # Resume graph execution with the decision
#                 response = workflow.invoke(Command(resume=decision), config=config)
#                 if response.get("messages") and response["messages"][-1].content:
#                     print("AI:", response["messages"][-1].content)
#         state = workflow.get_state(config)