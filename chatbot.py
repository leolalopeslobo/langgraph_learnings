from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langchain.schema import BaseMessage, AIMessage, HumanMessage

import requests

# --- Define state ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# --- Ollama LLM wrapper ---
class OllamaLLM:
    def __init__(self, model):
        self.model = model
        self.url = "http://localhost:11434/api/chat"  # Correct endpoint

    def invoke(self, messages: list[BaseMessage]):
        converted_messages = []
        for msg in messages:
            role = {
                "human": "user",
                "ai": "assistant"
            }.get(msg.type, msg.type)
            converted_messages.append({
                "role": role,
                "content": msg.content
            })

        data = {
            "model": self.model,
            "messages": converted_messages,
            "stream": False 
        }

        resp = requests.post(self.url, json=data)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

# --- Instantiate model ---
llm = OllamaLLM(model="deepseek-r1:1.5b")

# --- Node function ---
def chatbot(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=response)]}

# --- Build graph ---
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

# # --- Optional graph visualization ---
# from IPython.display import Image, display
# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     pass

# --- Streaming runner ---
def stream_graph_updates(user_input: str):
    user_message = HumanMessage(content=user_input)
    for event in graph.stream({"messages": [user_message]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# --- Chat loop ---
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about LangGraph?"
        print("User:", user_input)
        stream_graph_updates(user_input)
        break
