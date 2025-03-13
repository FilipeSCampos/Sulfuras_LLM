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
import shutil

# Configuração Inicial
st.set_page_config(page_title="Sulfuras - Chatbot Inteligente", layout="wide")

# Sidebar para API Key do Groq
st.sidebar.header("🔑 Configuração da API")
groq_api_key = st.sidebar.text_input("Insira sua API Key", type="password")

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")

if not groq_api_key:
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
        st.image("assets/sulfurs.webp", use_container_width=True)
    st.stop()

client = Groq(api_key=groq_api_key)
st.sidebar.success("🔑 API Key inserida com sucesso!")

# Cliente ChromaDB robusto
def get_chroma_client():
    db_path = "/tmp/chromadb"
    if not os.path.exists(db_path):
        os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)

# Inicializa cliente
chroma_client = get_chroma_client()

# Modelo embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# Criação robusta e garantida da coleção
def get_collection():
    return chroma_client.get_or_create_collection(
        name="document_embeddings",
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    )

collection = get_collection()



# Carregar documento após logado
uploaded_file = st.sidebar.file_uploader("📂 Carregar documento", type=["pdf", "docx", "csv"])

# Processar documento
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join([page.get_text() for page in doc])
        doc.close()

    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(uploaded_file)
        text = "\n".join([p.text for p in doc.paragraphs])

    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        numeric_columns = df.select_dtypes(include='number').columns
        if numeric_columns.any():
            coluna = st.selectbox("Escolha coluna numérica para visualização", numeric_columns)
            fig = px.histogram(df, x=coluna, title=f"Histograma da coluna: {coluna}")
            st.plotly_chart(fig, use_container_width=True)

        text = df.to_string(index=False)

    embeddings = embed_model.encode(text).tolist()
    collection.add(ids=[uploaded_file.name], documents=[text], embeddings=[embeddings])
    st.sidebar.success(f"Documento '{uploaded_file.name}' processado e armazenado!")

# Ver documentos armazenados
if st.sidebar.button("📚 Ver documentos armazenados"):
    docs = collection.get()
    if docs["ids"]:
        st.sidebar.write("📌 Documentos armazenados:")
        for doc_id in docs["ids"]:
            st.sidebar.write(f"- {doc_id}")
    else:
        st.sidebar.write("Nenhum documento encontrado no banco.")

# Botão que limpa banco e força recriação imediata
if st.sidebar.button("🗑️ Limpar banco de dados"):
    db_path = "/tmp/chromadb"
    shutil.rmtree(db_path, ignore_errors=True)
    os.makedirs(db_path, exist_ok=True)
    st.cache_resource.clear()  # Limpa cache do Streamlit completamente
    chroma_client = get_chroma_client()  # Recria explicitamente cliente
    collection = get_collection()        # Recria explicitamente coleção
    st.sidebar.success("Banco de dados limpo com sucesso!")
    st.rerun()  # Recarrega a aplicação imediatamente após limpeza

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Faça sua pergunta sobre o documento ou qualquer assunto:"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    docs = collection.get()
    contextos = "\n".join([f"{doc_id}: {doc[:500]}..." for doc_id, doc in zip(docs["ids"], docs["documents"])]) if docs["documents"] else "Nenhum documento carregado."

    historico = "\n".join([f'{msg["role"].capitalize()}: {msg["content"]}' for msg in st.session_state.messages])

    prompt_final = f"""
    Você é Sulfuras assistente inteligente, profissional e divertido criado por Filipe Sampaio. Responda a pergunta abaixo com base no contexto fornecido.

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
                messages=[{"role": "system", "content": "Você é um assistente inteligente, profissional e divertido."},
                          {"role": "user", "content": prompt_final}],
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=2048
            ).choices[0].message.content

    except Exception as e:
        resposta = f"⚠️ Erro ao acessar o Groq: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": resposta})

    with st.chat_message("assistant"):
        st.markdown(resposta)

