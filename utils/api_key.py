# utils/api_key.py

import streamlit as st

def set_api_key():
    st.rerun()

def initialize_api_key():
    if "groq_api_key" not in st.session_state:
        st.session_state.groq_api_key = None

def get_api_key():
    return st.session_state.groq_api_key
