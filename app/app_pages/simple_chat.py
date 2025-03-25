import yaml
import uuid

import streamlit as st

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

from graph.main import app as main_graph
from langchain_tools import get_remote_ip

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "config" not in st.session_state:
    st.session_state.config = {
        "thread_id": st.session_state.session_id,
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = main_graph

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    else:
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input():
    with st.chat_message("user"):
        st.write(prompt)
        st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = st.session_state.graph.invoke(
            input={"messages": [HumanMessage(prompt)]},
            config=RunnableConfig({"callbacks": [st_callback], "configurable": st.session_state.config}),
        )
        response = response["messages"][-1].content
        st.markdown(response)
        st.session_state.messages.append(AIMessage(response))
