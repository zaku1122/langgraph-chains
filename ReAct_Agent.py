#Read , Act Agent
from typing import TypedDict, Annotated, Sequence
import langchain_core
from langchain_core.messages import BaseMessage, HumanMessage #The Foundational class for all the message types in LangChain
from langchain_core.messages import ToolMessage # Passess data back to the LLM after it calls a tool such as the content and the tool call id
from langchain_core.messages import SystemMessage #message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os

load_dotenv()

#Annotated - Provides additional context without affecting the type itself

 #Sequence(also type annotation) - To automatically handle the state upfates for sequence such as by adding new messages to achat history

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


@tool
def add(a: int, b:int):
    """this is add func that add 2 numbers together"""

    return a+b


@tool
def multiply(a: int, b:int):
    """Multiplication Function"""
    
    return a*b

@tool
def subtract(a: int, b:int):
    """this is add func that add 2 numbers together"""

    return a-b
tools = [add, subtract, multiply] 

model = ChatOpenAI(model = "gpt-4o").bind_tools(tools)


def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content = 
                                  "You are my AI assistant, Please answser my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages":[response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "our_agent")

app = graph.compile()


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = { "messages": [HumanMessage(content = "Add 40 + 12 and then multiply the result by 6")]}

print_stream(app.stream(inputs , stream_mode = "values"))
