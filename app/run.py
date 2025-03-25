"""
This is index page for my app.
"""

import os
import yaml

import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

from settings import secret_path

with open(secret_path, "r", encoding="utf-8") as f:
    secret = yaml.safe_load(f)
langsmith_tracing = secret.get("langsmith").get("tracing")
langsmith_endpoint = secret.get("langsmith").get("endpoint")
langsmith_api_key = secret.get("langsmith").get("api_key")
langsmith_project = secret.get("langsmith").get("project")

os.environ["LANGSMITH_TRACING"] = langsmith_tracing
os.environ["LANGSMITH_ENDPOINT"] = langsmith_endpoint
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGSMITH_PROJECT"] = langsmith_project

st.set_page_config(layout="wide")

nav = get_nav_from_toml(".streamlit/pages_sections.toml")

pg = st.navigation(nav)

add_page_title(pg)

pg.run()
