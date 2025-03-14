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
import shutil
import time

# Configurar event loop (para evitar warnings com asyncio)
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Configuração da página
st.set_page_config(page_title="Sulfuras - Chatbot Inteligente", layout="wide")

# Sidebar: Configuração da API
st.sidebar.header("🔑 Configuração da API")
groq_api_key = st.sidebar.text_input("Insira sua API Key", type="password")

# Inicializa o histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")

# Se não houver API Key, exibe a tela inicial
if not groq_api_key:
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

# Inicializa o cliente Groq
client = Groq(api_key=groq_api_key)
st.sidebar.success("🔑 API Key inserida com sucesso!")

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

# Carregar documento após login
uploaded_file = st.sidebar.file_uploader("📂 Carregar documento", type=["pdf", "docx", "csv"])

# Carregar o modelo de embeddings (cache para evitar recarregamento)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()

# Inicializar ChromaDB (usando armazenamento persistente)
@st.cache_resource
def get_chroma_client():
    path = "./chromadb"
    if not os.path.exists(path):
        os.makedirs(path)  # Recria a pasta se não existir
        time.sleep(1)  # Aguarda um momento para garantir que a pasta seja criada

    client = chromadb.PersistentClient(path=path)
    return client

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(
    name="document_embeddings",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

# Processar documento se houver upload
if uploaded_file:
    text = process_document(uploaded_file)
    if text:
        embeddings = embed_model.encode(text).tolist()
        collection.add(ids=[uploaded_file.name], documents=[text], embeddings=[embeddings])
        st.sidebar.success("Documento processado e armazenado!")
    else:
        st.sidebar.error("Não foi possível extrair texto do documento.")

# Função para resetar o banco e limpar o estado da sessão
def reset_chromadb():
    path = "./chromadb"
    try:
        if os.path.exists(path):
            shutil.rmtree(path)  # Remove a pasta do ChromaDB
        os.makedirs(path)  # Recria a pasta vazia
        
        # Limpar estado da sessão do Streamlit
        st.session_state.clear()
        st.session_state["messages"] = []  # Reinicializa histórico do chatbot
        
        st.success("Banco de dados resetado com sucesso!")
        st.rerun()  # Força a reexecução do app para aplicar as mudanças
    except Exception as e:
        st.error(f"Erro ao limpar banco de dados: {e}")

# Botão para limpar banco de dados
if st.sidebar.button("🗑️ Limpar banco de dados"):
    reset_chromadb()

# Exibir documentos armazenados
if st.sidebar.button("📚 Ver documentos armazenados"):
    docs = collection.peek()
    if docs.get("ids"):
        st.sidebar.write("📌 Documentos armazenados:")
        for doc_id in docs["ids"]:
            st.sidebar.write(f"- {doc_id}")
    else:
        st.sidebar.write("Nenhum documento encontrado.")

# Exibir histórico e interação com o chatbot
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input de mensagem para interação
if prompt := st.chat_input("Faça sua pergunta sobre o documento ou qualquer assunto:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    docs = collection.get()
    contextos = (
        "\n".join(
            f"{doc_id}: {doc[:500]}..." for doc_id, doc in zip(docs["ids"], docs["documents"])
        )
        if docs.get("documents")
        else "Nenhum documento carregado."
    )
    historico = "\n".join(f'{msg["role"].capitalize()}: {msg["content"]}' for msg in st.session_state.messages)

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

    st.session_state.messages.append({"role": "assistant", "content": resposta})
    st.rerun()  # Força a reexecução para atualizar a exibição

# Loop para exibir todas as mensagens (executado após o input e a reexecução)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
