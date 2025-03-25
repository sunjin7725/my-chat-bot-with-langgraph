from typing import List, Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import load_chat_model, graph_to_png
from rag.pgvector.vectorstore import PostgresVectorstore

vectorstore = PostgresVectorstore().create_chain()
chain = vectorstore.chain
retrieval = vectorstore.retriever


class RetrievalState(TypedDict):
    messages: Annotated[List[BaseMessage], "User message list"]
    search_query: Annotated[List[str], "Web search query list"]
    summary: Annotated[str, "Message histories summary"]
    contents: Annotated[List[str], "Finded content"]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


model = load_chat_model(temperature=0)
structed_model_grader = model.with_structured_output(GradeDocuments)


transform_query_prompt = """
    You are a question re-writer that converts an input question to a better version that is optimized for vectorstore retrieval.
    In the vectorstore, there is a Korean tax law. So you have to make a query Korean.
    Look at the input and try to reason about the underlying semantic intent/meaning.
    Please help improve the following question to provide a better search query for the vectorstore to answer the given question. Please don't make the same query before transformed.

    Before transformed query:
        {search_query}

    Consider the following message history to understand the context better:
        {summary}
"""
transform_query_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", transform_query_prompt),
        (
            "user",
            """
        Here is the initial question:
            {question}
        Formulate an imporved question.
     """,
        ),
    ]
)
transform_query_chain = transform_query_prompt_template | model | StrOutputParser()

grader_system_prompt = """
    You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        (
            "user",
            """
        Retrieved documents: 
            {document}
        
        User question:
            {question}
     """,
        ),
    ]
)
grade_chain = grade_prompt | structed_model_grader


def transform_query(state: RetrievalState):
    messages = state["messages"]
    summary = state.get("summary", "")
    search_query = state.get("search_query", [])
    new_question = transform_query_chain.invoke(
        {"question": messages, "search_query": search_query, "summary": summary}
    )
    if search_query:
        search_query.append(new_question)
    else:
        search_query = [new_question]
    return {"messages": messages, "summary": summary, "search_query": search_query}


def retrieve(state: RetrievalState):
    last_search_query = state["search_query"][-1]
    search_result = retrieval.invoke(last_search_query)
    return {"search_query": state["search_query"], "contents": search_result}


def grade_documents(state: RetrievalState):
    last_search_query = state["search_query"][-1]
    contents = state["contents"]

    filtered_docs = []
    for d in contents:
        score = grade_chain.invoke({"question": last_search_query, "document": contents})
        grade = score.binary_score

        if grade == "yes":
            filtered_docs.append(d)

    return {"contents": filtered_docs}


def is_filtered_documents_ok(state: RetrievalState):
    filtered_documents = state["contents"]
    if filtered_documents:
        return END
    else:
        return "transform_query"


workflow = StateGraph(RetrievalState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("transform_query", transform_query)
workflow.add_node("grade_documents", grade_documents)

workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents", is_filtered_documents_ok, {END: END, "transform_query": "transform_query"}
)
app = workflow.compile()
# graph_to_png(app, "retrieval_graph.png")
