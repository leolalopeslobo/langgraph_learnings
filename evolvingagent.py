# This agent is going to keep getting better as more and more concepts are going to be learnt

from langgraph.graph import MessagesState
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from langchain_community.chat_models import ChatOllama

from langchain_core.messages import AIMessage, HumanMessage


# Define state type
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def chatbot(state: MessagesState) -> MessagesState:
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


llm = ChatOllama(model="deepseek-r1:1.5b")


from langgraph.graph import StateGraph, START, END

builder = StateGraph(MessagesState) # This is telling LangGraph: “I’m building a state machine, and the state being passed between nodes is of type MessagesState.”
# Think of StateGraph like a flowchart or wiring diagram
builder.add_node("chatbot",chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

state: MessagesState = {
    "messages": [AIMessage(content="Hey! What's on your mind today?", name="Model")]
}

print(f"Chatbot: {state['messages'][-1].content}") #printing the content of the most recent message


while True:
    try:
        user_input = input("User: ")

        state["messages"].append(HumanMessage(content=user_input, name="User")) 

        state = graph.invoke(state)
        print(f"Chatbot: {state['messages'][-1].content}")

        # print(state)


    except KeyboardInterrupt:
        print("\nChatbot: Exiting chat.")
        break
