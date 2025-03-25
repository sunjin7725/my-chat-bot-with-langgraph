from typing import List, Annotated, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage
from langgraph.graph import START, StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_tools import ddg_search
from utils import load_chat_model


class WebSearchState(TypedDict):
    messages: Annotated[List[BaseMessage], "User message list"]
    search_query: Annotated[List[str], "Web search query list"]
    summary: Annotated[str, "Message histories summary"]
    content: Annotated[str, "Finded content"]


class RelevantCheck(BaseModel):
    """Binary score for relevance check on searched content result."""

    binary_score: str = Field(description="Search content result are relevant to the question, 'yes' or 'no'")


model = load_chat_model(temperature=0)
structed_output_model_relevant = model.with_structured_output(RelevantCheck)

transform_query_prompt = """
    You are a query re-writer that converts an input question into a better version optimized for the search engine (DuckDuckGo). 
    Provide a better search query for the search engine to answer the given question. Please don't make the same query before transformed query.

    Before transformed query:
        {search_query}

    Consider the following message history to understand the context better:
        {summary}

    Using the provided message history and user's question, formulate an improved question that captures the user's intent more effectively.

    ### DuckDuckGo Search Operators:
    - **cats dogs**: Results about cats or dogs.
    - **cats -dogs**: Fewer dogs in results.
    - **cats +dogs**: More dogs in results.
    - **cats filetype:pdf**: PDFs about cats. Supported file types: pdf, doc(x), xls(x), ppt(x), html.
    - **dogs site:example.com**: Pages about dogs from example.com.
    - **cats -site:example.com**: Pages about cats, excluding example.com.
    - **intitle:dogs**: Page title includes the word "dogs".
    - **inurl:cats**: Page URL includes the word "cats".

    Make sure to consider these operators when formulating the improved question. 
    You should create the query by following the search methods outlined above to ensure it is optimized for DuckDuckGo. 
    **Ensure that the new query is not similar to the previous queries provided.**
    **Avoid using unsupported operators like "OR" or list type like "[...]" in your queries.**
"""
transform_query_prompt_template = ChatPromptTemplate.from_messages(
    [("system", transform_query_prompt), ("user", "{question}")]
)
transform_query_chain = transform_query_prompt_template | model | StrOutputParser()

is_relevant_prompt = """
    You are a grader assessing the relevance of a web search result to the user's question.
    Please evaluate the provided search result and determine if it is relevant to the user's question.
    Give a binary score 'yes' or 'no' score. "yes" if the search result is relevant, and "no" if it is not.

    Search Result:
        {content}
"""
is_relevant_prompt_template = ChatPromptTemplate.from_messages(
    [("system", is_relevant_prompt), ("user", "{question}")]
)
is_relevant_chain = is_relevant_prompt_template | structed_output_model_relevant


def transform_query(state: WebSearchState):
    messages = state["messages"]
    search_query = state.get("search_query", [])
    summary = state.get("history", "")
    new_question = transform_query_chain.invoke(
        {"question": messages, "summary": summary, "search_query": search_query}
    )
    if search_query:
        search_query.append(new_question)
    else:
        search_query = [new_question]
    print(search_query)
    return {"search_query": search_query}


def web_search(state: WebSearchState):
    last_search_query = state["search_query"][-1]
    search_result = ddg_search.invoke(last_search_query)
    return {"search_query": state["search_query"], "content": search_result}


def relevant_check(state: WebSearchState):
    print("==== [RELEVANT CHECK SEARCH RESULT WITH QUESTION] ====")
    last_search_query = state["search_query"][-1]
    content = state["content"]
    score = is_relevant_chain.invoke({"question": last_search_query, "content": content})
    grade = score.binary_score

    if grade == "yes":
        print("---RELEVANT_CHECK: OK!")
        return "end"
    else:
        print("---RELEVANT_CHECK: NOT OKAY!")
        return "transform_query"


workflow = StateGraph(WebSearchState)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

workflow.add_edge(START, "transform_query")
workflow.add_edge("transform_query", "web_search")
workflow.add_conditional_edges("web_search", relevant_check, {"end": END, "transform_query": "transform_query"})

app = workflow.compile()

###### graph test ################
# 그래프 png 생성
# from utils import graph_to_png
# graph_to_png(app, "./web_search_graph.png")

# 그래프 테스트 실행
# inputs = {"question": "오늘 주요 뉴스 5개"}
# config = {"thread_id": 1}
# for event in app.stream(input=inputs, config=config, stream_mode="values"):
#     print(f"question: {event['question']}")
#     if "content" in event:
#         print(f"result: {event['content']}")
