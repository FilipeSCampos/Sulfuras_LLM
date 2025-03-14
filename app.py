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

# Função para atualizar a página após inserir a API key
def set_api_key():
    st.rerun()

# Inicializa ou recupera a API key do session_state
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = None

# Se a API key não foi definida, exibe o campo para inseri-la e uma tela inicial
if st.session_state.groq_api_key is None:
    st.sidebar.header("🔑 Configuração da API")
    input_key = st.sidebar.text_input("Insira sua API Key", type="password", key="input_groq_api_key")
    st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")

    if input_key:
        st.session_state.groq_api_key = input_key
        st.rerun()

    
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

# Se a API key já estiver definida, a variável é recuperada
groq_api_key = st.session_state.groq_api_key

# Inicializa o cliente Groq
client = Groq(api_key=groq_api_key)
# Exibe mensagem de sucesso na sidebar
#st.sidebar.success("🔑 API Key inserida com sucesso!")

# Inicializar cliente ChromaDB
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(
    name="document_embeddings",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# Gerenciamento de múltiplos chats
def save_chats():
    with open(CHATS_FILE, "w") as file:
        json.dump(st.session_state.chats, file)

def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r") as file:
            return json.load(file)
    return {}

if "chats" not in st.session_state:
    st.session_state.chats = load_chats()
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")

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

def delete_chat(chat_name):
    if chat_name in st.session_state.chats:
        del st.session_state.chats[chat_name]
        if st.session_state.current_chat == chat_name:
            st.session_state.current_chat = None
        save_chats()
        st.rerun()

st.sidebar.subheader("📌 Seus Chats")
# Exibe um botão grande para cada chat na sidebar, um embaixo do outro
for chat_name in st.session_state.chats.keys():
    if st.sidebar.button(chat_name, key=f"chat_{chat_name}"):
        st.session_state.current_chat = chat_name
        st.rerun()

# Se um chat estiver ativo, exibe um botão elegante para excluí-lo
if st.session_state.current_chat:
    excluir = st.sidebar.button(f"❌ Excluir Chat: {st.session_state.current_chat}")
    if excluir:
        delete_chat(st.session_state.current_chat)


create_new_chat()

# Função para limpar e recriar o banco de dados
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

# Função para processar documentos
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
        text = df.to_string(index=False)
    else:
        text = ""
    return text

uploaded_file = st.sidebar.file_uploader("📂 Carregar documento", type=["pdf", "docx", "csv"])

if uploaded_file:
    text = process_document(uploaded_file)
    if text:
        embeddings = SentenceTransformer("all-MiniLM-L6-v2").encode(text).tolist()
        collection.add(ids=[uploaded_file.name], documents=[text], embeddings=[embeddings])
        st.sidebar.success("Documento processado e armazenado!")
    else:
        st.sidebar.error("Não foi possível extrair texto do documento.")

if st.sidebar.button("📚 Ver documentos armazenados"):
    docs = collection.peek()
    if docs.get("ids"):
        st.sidebar.write("📌 Documentos armazenados:")
        for doc_id in docs["ids"]:
            st.sidebar.write(f"- {doc_id}")
    else:
        st.sidebar.write("Nenhum documento encontrado.")

if st.sidebar.button("🗑️ Limpar banco de dados"):
    delete_chromadb_collection()
    recreate_chromadb_collection()

# Exibir chat selecionado
if st.session_state.current_chat:
    st.title(f"💬 Chat: {st.session_state.current_chat}")
    messages = st.session_state.chats[st.session_state.current_chat]
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Faça sua pergunta..."):
        messages.append({"role": "user", "content": prompt})

        docs = collection.get()
        contextos = (
            "\n".join(
                f"{doc_id}: {doc[:500]}..." for doc_id, doc in zip(docs["ids"], docs["documents"])
            )
            if docs.get("documents")
            else "Nenhum documento carregado."
        )
        historico = "\n".join(f'{msg["role"].capitalize()}: {msg["content"]}' for msg in messages)

        prompt_final = f"""
        Você é Sulfuras, assistente inteligente criado por Filipe Sampaio. Responda com base no contexto fornecido.
        
        Contexto:
        {contextos}
        
        Histórico:
        {historico}
        
        Pergunta:
        {prompt}
        
        Resposta detalhada:
        """

        try:
            with st.spinner("Gerando resposta..."):
                resposta = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "Você é um assistente inteligente, profissional e divertido."},
                        {"role": "user", "content": prompt_final}
                    ],
                    model="llama3-8b-8192",
                    temperature=0.5,
                    max_tokens=2048,
                ).choices[0].message.content
        except Exception as e:
            resposta = f"⚠️ Ocorreu um erro ao acessar o Groq: {str(e)}"

        messages.append({"role": "assistant", "content": resposta})
        st.session_state.chats[st.session_state.current_chat] = messages
        save_chats()
        st.rerun()
