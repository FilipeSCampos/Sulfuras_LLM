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
        st.image("assets/sulfurs.webp", use_container_width=True)
    st.stop()

client = Groq(api_key=groq_api_key)
st.sidebar.success("🔑 API Key inserida com sucesso!")



# Limpeza do banco ao recarregar a página
@st.cache_resource
def get_chroma_client():
    db_path = "./chromadb"

    # Limpa a pasta do ChromaDB manualmente na inicialização
    if os.path.exists(db_path):
        shutil.rmtree(db_path)

    client = chromadb.PersistentClient(path=db_path)
    return client

chroma_client = get_chroma_client()

# Modelo embeddings
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

# Coleção
collection = chroma_client.get_or_create_collection(
    name="document_embeddings",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
)

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
    st.sidebar.success("Documento processado e armazenado!")

# Ver documentos armazenados
# Sidebar - Visualização dos documentos armazenados
if st.sidebar.button("📚 Ver documentos armazenados"):
    docs = collection.get()
    if docs and "documents" in docs and docs["documents"]:
        st.sidebar.write("📌 Documentos no Banco Vetorial:")
        for doc_id, doc_text in zip(docs["ids"], docs["documents"]):
            st.sidebar.text_area(f"{doc_id}", value=doc_text, height=150, disabled=True)
    else:
        st.sidebar.write("Nenhum documento encontrado no banco vetorial.")

# Adicionando rodapé logo abaixo do botão
st.sidebar.markdown("""
---
Desenvolvido por: Filipe S. Campos, Rafael Canuto, Tatiana H., Hermes e Vinicius.  
Orientador: M.e Weslley Rodrigues.
""")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
# Chat input do usuário
if prompt := st.chat_input("Faça sua pergunta sobre o documento ou qualquer assunto:"):

    # 1. Mensagem do usuário adicionada ao histórico imediatamente
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Exibir imediatamente mensagem do usuário
    with st.chat_message("user"):
        st.markdown(prompt)

    if uploaded_file:
        query_embedding = embed_model.encode(prompt).tolist()
        results = collection.query(query_embeddings=[query_embedding], n_results=1)

        if results["documents"] and results["documents"][0]:
            contexto_documento = results["documents"][0][0]
        else:
            contexto_documento = "Sem contexto disponível."
    else:
        contexto_documento = "Sem documento carregado. Respondendo sem contexto específico."

    historico = "\n".join([f'{msg["role"].capitalize()}: {msg["content"]}' for msg in st.session_state.messages])

    prompt_final = f"""
    Você é Sulfuras assistente inteligente, profissional e divertido criado por Filipe Sampaio. Responda a pergunta abaixo com base no contexto fornecido.

    Contexto:
    {contexto_documento}

    Histórico:
    {historico}

    Pergunta:
    {prompt}

    Resposta detalhada:
    """

    # 3. Gerar resposta do chatbot
    try:
        with st.spinner("Gerando resposta..."):
            resposta = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente inteligente, profissional e divertido."},
                    {"role": "user", "content": prompt_final}
                ],
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=2048
            ).choices[0].message.content

    except Exception as e:
        resposta = f"⚠️ Ocorreu um erro ao acessar o Groq: {str(e)}"

    # 4. Adicionar resposta ao histórico
    st.session_state.messages.append({"role": "assistant", "content": resposta})

    # 5. Exibir imediatamente resposta do chatbot
    with st.chat_message("assistant"):
        st.markdown(resposta)
