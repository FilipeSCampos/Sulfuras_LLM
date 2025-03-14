__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import plotly.express as px
import fitz
from docx import Document
import os
import asyncio
import json
import shutil
import time

CHROMA_DB_PATH = "./chromadb"
CHATS_FILE = "chats.json"

# Configurar event loop (para evitar warnings com asyncio)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

st.set_page_config(page_title="Sulfuras - Chatbot Inteligente", layout="wide")

# Sidebar: Configuração da API
groq_api_key = st.sidebar.text_input("Insira sua API Key", type="password")
if not groq_api_key:
    st.sidebar.warning("🔑 Insira sua API Key para continuar.")
    # Exibir imagem na parte principal
    col_texto, col_imagem = st.columns([2, 1])

    with col_texto:
        st.markdown("""
        Este projeto é um chatbot inteligente capaz de compreender documentos carregados (PDF, DOCX ou CSV) e responder perguntas contextuais.

        **Criado para TCC usando:**
        - 🖥️ Streamlit para interface gráfica.
        - 🤖 Groq (Llama) como modelo LLM.
        - 🧠 ChromaDB para armazenamento vetorial.
        - 📊 Análises visuais com Plotly.

        **Insira sua API Key no painel lateral para começar.**

        Desenvolvido por: Filipe S. Campos, Rafael Canuto, Tatiana H., Hermes e Vinicius.

        Orientador: M.e Weslley Rodrigues.
        """)
    with col_imagem:
        st.image("assets\sulfurs.webp", use_container_width=True)
    st.stop()

client = Groq(api_key=groq_api_key)
st.sidebar.success("🔑 API Key inserida com sucesso!")

# Gerenciamento de múltiplos chats
def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_chats():
    with open(CHATS_FILE, "w") as file:
        json.dump(st.session_state.chats, file)

# Inicializa os chats se não existirem
if "chats" not in st.session_state:
    st.session_state.chats = load_chats()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

# Criar um novo chat
def create_new_chat():
    chat_name = st.sidebar.text_input("Nome do novo chat")
    if st.sidebar.button("➕ Criar Chat") and chat_name:
        if chat_name not in st.session_state.chats:
            st.session_state.chats[chat_name] = []
            st.session_state.current_chat = chat_name
            save_chats()
            st.rerun()
        else:
            st.sidebar.warning("Esse nome já existe!")

# Excluir um chat
def delete_chat(chat_name):
    if chat_name in st.session_state.chats:
        del st.session_state.chats[chat_name]
        if st.session_state.current_chat == chat_name:
            st.session_state.current_chat = None
        save_chats()
        st.rerun()

# Seleção de chats existentes
st.sidebar.subheader("📌 Seus Chats")
for chat_name in list(st.session_state.chats.keys()):
    col1, col2 = st.sidebar.columns([0.8, 0.2])
    with col1:
        if st.sidebar.button(chat_name):
            st.session_state.current_chat = chat_name
            st.rerun()
    with col2:
        if st.sidebar.button("❌", key=f"del_{chat_name}"):
            delete_chat(chat_name)

create_new_chat()

# Função para processar o documento
def process_document(uploaded_file):
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())
        numeric_columns = df.select_dtypes(include="number").columns
        if numeric_columns.any():
            coluna = st.selectbox("Escolha coluna numérica para visualização", numeric_columns)
            fig = px.histogram(df, x=coluna, title=f"Histograma da coluna: {coluna}")
            st.plotly_chart(fig, use_container_width=True)
        text = df.to_string(index=False)
    else:
        text = ""
    return text

uploaded_file = st.sidebar.file_uploader("📂 Carregar documento", type=["pdf", "docx", "csv"])

# Funções para limpar e recriar o banco de dados
def delete_chromadb_collection():
    try:
        chroma_client.delete_collection(name="document_embeddings")
        st.success("Coleção 'document_embeddings' excluída com sucesso!")
    except Exception as e:
        st.error(f"Erro ao excluir a coleção: {e}")

def recreate_chromadb_collection():
    try:
        global collection
        collection = chroma_client.get_or_create_collection(
            name="document_embeddings",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        st.success("Coleção 'document_embeddings' recriada com sucesso!")
    except Exception as e:
        st.error(f"Erro ao recriar a coleção: {e}")

if st.sidebar.button("🗑️ Limpar banco de dados"):
    delete_chromadb_collection()
    recreate_chromadb_collection()

# Markdown inicial
if not st.session_state.current_chat:
    col_texto, col_imagem = st.columns([2, 1])
    with col_texto:
        st.markdown(
            """
            Este projeto é um chatbot inteligente capaz de compreender documentos carregados (PDF, DOCX ou CSV) e responder perguntas contextuais.
            
            **Criado para TCC usando:**
            - 🖥️ Streamlit para interface gráfica.
            - 🤖 Groq (Llama) como modelo LLM.
            - 🧠 ChromaDB para armazenamento vetorial.
            - 📊 Análises visuais com Plotly.
            
            **Insira sua API Key no painel lateral para começar.**
            
            Desenvolvido por: Filipe S. Campos, Rafael Canuto, Tatiana H., Hermes e Vinicius.
            
            Orientador: M.e Weslley Rodrigues.
            """
        )
    with col_imagem:
        st.image("assets/sulfurs.webp", use_container_width=True)
    st.stop()

# Exibir chat selecionado
if st.session_state.current_chat:
    st.title(f"💬 Chat: {st.session_state.current_chat}")
    messages = st.session_state.chats[st.session_state.current_chat]
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Faça sua pergunta..."):
        messages.append({"role": "user", "content": prompt})
        
        # Geração da resposta
        try:
            with st.spinner("Gerando resposta..."):
                resposta = client.chat.completions.create(
                    messages=[{"role": "system", "content": "Você é um assistente inteligente."}] + messages,
                    model="llama3-8b-8192",
                    temperature=0.5,
                    max_tokens=2048,
                ).choices[0].message.content
        except Exception as e:
            resposta = f"⚠️ Erro ao acessar o Groq: {str(e)}"
        
        messages.append({"role": "assistant", "content": resposta})
        st.session_state.chats[st.session_state.current_chat] = messages
        save_chats()
        st.rerun()
