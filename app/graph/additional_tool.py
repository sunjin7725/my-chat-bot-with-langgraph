from typing import List, TypedDict, Annotated
from langgraph.graph import START, StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, AIMessage
from langchain_tools import who_are_you_tool, get_remote_ip_tool, datetime_tool, python_repl, wikipidia
from utils import load_chat_model, graph_to_png

tools = [who_are_you_tool, get_remote_ip_tool, datetime_tool, python_repl, wikipidia]
tool_node = ToolNode(tools)

model = load_chat_model()
model_with_tools = model.bind_tools(tools)


def chat(state: MessagesState):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


workflow = StateGraph(MessagesState)

workflow.add_node("chat", chat)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "chat")
workflow.add_conditional_edges("chat", tools_condition)
workflow.add_edge("tools", "chat")
workflow.add_edge("chat", END)

app = workflow.compile()
# graph_to_png(app, "tools_graph.png")
