from typing import List, Annotated, TypedDict, Literal
from pydantic import BaseModel, Field

from langgraph.graph import START, StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, RemoveMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from graph.web_search import app as web_search_graph
from graph.retrieval import app as retrieval_graph
from graph.additional_tool import app as tools_graph
from utils import load_chat_model, graph_to_png


class MainState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    documents: Annotated[List, "Document by web_search or retrieval"]
    summary: Annotated[str, "chat history summary"]
    tools_information: Annotated[List[str], "information provided from tools node"]


class Router(BaseModel):
    """Route a user query to the most relevance datasource."""

    datasource: Literal[
        "",
        "tools",
        "web_search",
        "vectorstore",
    ] = Field(
        ...,
        description="""
        Give a user question choose to route. 
        You can select only web_search if need.

        Choose in below. If you don't need anything, give an empty string:
        - tools: if you need custom tools that provide information below.
        - web_search: if you need to search in DuckDuckGo search engine.
        - vectorstore: if you need to search in the vectorstore that has information about Korean tax law.

        ### tools:
        - Wikipedia: To query information from Wikipedia for general knowledge.
        - Python REPL: To execute Python code snippets dynamically.
        - Get Remote IP: To retrieve the remote client's IP address.
        - Datetime Tool: To get the current date and time.
        - Who Are You Tool: To provide information about the AI service and its creator.
        
        Please specify your choice based on the options above.
        """,
    )


model = load_chat_model(model="gpt-4o-mini", stream=True)
routing_model = model.with_structured_output(Router)

chat_prompt = """
You are a helpful and friendly assistant designed for engaging conversations with users. 
Your primary mission is to answer questions using the following context and chat history. 
If the context or chat history is not provided, feel free to engage the user in a friendly manner and ask how you can assist them. 
You should aim to provide concise and relevant answers while maintaining a natural flow of dialogue.

If you do not have the necessary context or chat history to answer the user's question, you can still provide a general response based on your knowledge. 
However, remember that special knowledge may be required for certain questions, and you should acknowledge that if applicable.

If the context includes source information, please include the source in your response.

Always respond in the same language as the user's question.

Additional information:
    {tools_information}

context:
    {context}

chat_history:
    {chat_history}
"""
chat_prompt_template = ChatPromptTemplate([("system", chat_prompt), ("human", "{question}")])
chat_chain = chat_prompt_template | model | StrOutputParser()

routing_prompt = """
You are an expert at routing a user question to ["", "web_search", "vectorstore", "additional_tools"].
"""
routing_prompt_template = ChatPromptTemplate([("system", routing_prompt), ("human", "{question}")])
routing_chain = routing_prompt_template | routing_model


def summarize_history(state: MainState):
    summary = state.get("summary", "")
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above in Korean:"
        )
    else:
        summary_message = "Create a summary of the conversation above in Korean:"

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}


def chat(state: MainState):
    summary = state.get("summary", "")
    context = state.get("documents", "")
    tools_information = state.get("tools_information", [])
    response = chat_chain.invoke(
        {
            "question": state["messages"],
            "chat_history": summary,
            "context": context,
            "tools_information": tools_information,
        }
    )
    return {"messages": AIMessage(response)}


def web_search(state: MainState):
    messages = state["messages"]
    summary = state.get("summary")
    response = web_search_graph.invoke(
        {
            "messages": messages,
            "summary": summary,
        }
    )
    return {"context": response}


def retrieval(state: MainState):
    messages = state["messages"]
    summary = state.get("summary")
    response = retrieval_graph.invoke(
        {
            "messages": messages,
            "summary": summary,
        }
    )
    return {"context": response}


def tools(state: MainState):
    messages = state["messages"]
    response = tools_graph.invoke({"messages": messages})
    tools_information = []
    for message in response["messages"]:
        if isinstance(message, AIMessage) and hasattr(message, "tool_calls"):
            tool_call = message.tool_calls[0]
            tools_information.append({"type": "tool_call", "name": tool_call["name"], "args": tool_call["args"]})
        if isinstance(message, ToolMessage):
            tools_information.append({"type": "tool_result", "name": message.name, "content": message.content})
    return {"tools_information": tools_information}


def routing_question(state: MainState):
    question = state["messages"][-1].content
    source = routing_chain.invoke({"question": question})
    if source.datasource == "":
        return "chat"
    elif source.datasource == "web_search":
        return "web_search"
    elif source.datasource == "vectorstore":
        return "vectorstore"
    elif source.datasource == "tools":
        return "tools"


def need_summarize_history(state: MainState):
    if len(state["messages"]) > 6:
        return "summarize_history"
    return END


workflow = StateGraph(MainState)
workflow.add_node("chat", chat)
workflow.add_node("summarize_history", summarize_history)
workflow.add_node("web_search", web_search)
workflow.add_node("vectorstore", retrieval)
workflow.add_node("tools", tools)

workflow.add_conditional_edges(
    START,
    routing_question,
    {"chat": "chat", "web_search": "web_search", "vectorstore": "vectorstore", "tools": "tools"},
)
workflow.add_edge("web_search", "chat")
workflow.add_edge("vectorstore", "chat")
workflow.add_edge("tools", "chat")
workflow.add_conditional_edges("chat", need_summarize_history, {"summarize_history": "summarize_history", END: END})
workflow.add_edge("summarize_history", END)

memory = MemorySaver()
app = workflow.compile(memory)
# graph_to_png(app, "main_graph.png")
