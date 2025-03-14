import os
import json
import time
import hashlib
import sqlite3
import asyncio
import streamlit as st
import pandas as pd
from groq import Groq
from sentence_transformers import SentenceTransformer

# Importar configurações e utilitários
from config import MODEL_NAME
from utils.api_key import initialize_api_key, set_api_key
from utils.chroma_client import get_chroma_client, get_or_create_collection, delete_and_recreate_collection
from utils.chat_manager import load_chats, save_chats, create_new_chat, delete_chat
from utils.file_processing import process_document
from utils.loggins_db import initialize_db, create_user, validate_user

# O comando set_page_config deve ser o primeiro comando do Streamlit
st.set_page_config(page_title="Sulfuras Chatbot", layout="wide")

# Inicializa o banco de dados de login (caso ainda não exista)
initialize_db()
# Inicializa a API Key (não sobrescreve se já estiver definida)
initialize_api_key()

# --- Tela de Login / Cadastro Unificada com API Key ---
if not st.session_state.get("logged_in", False) or st.session_state.get("groq_api_key") is None:
    
    st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")
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
            
            **Informe sua API Key, Email e Senha para acessar o sistema.**
            
            Desenvolvido por: Filipe S. Campos, Rafael Canuto, Tatiana H., Hermes e Vinicius.
            
            Orientador: M.e Weslley Rodrigues.
            """
        )
    with col_imagem:
        st.image("assets/sulfurs.webp", use_container_width=True)
    
    with st.sidebar:
        st.header("Acesso ao Sulfuras")
        opcao = st.radio("Selecione a opção", ["Login", "Cadastro"])
        
        # Campo de API Key (obrigatório)
        api_key_input = st.text_input("Insira sua API Key", type="password", key="api_key_input")
        if not api_key_input:
            st.error("Por favor, insira sua API Key para continuar.")
            st.stop()
        
        if opcao == "Login":
            st.subheader("Login")
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Senha", type="password", key="login_password")
            if st.button("Entrar"):
                if not email or not password:
                    st.error("Preencha todos os campos.")
                else:
                    if validate_user(email, password):
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.groq_api_key = api_key_input
                        st.success("Login realizado com sucesso!")
                        st.rerun()
                    else:
                        st.error("Credenciais inválidas!")
        else:  # Cadastro
            st.subheader("Cadastro")
            email = st.text_input("Email", key="cadastro_email")
            password = st.text_input("Senha", type="password", key="cadastro_password")
            confirm_password = st.text_input("Confirmar Senha", type="password", key="cadastro_confirm_password")
            if st.button("Cadastrar"):
                if not email or not password or not confirm_password:
                    st.error("Preencha todos os campos.")
                elif password != confirm_password:
                    st.error("As senhas não coincidem!")
                else:
                    if create_user(email, password):
                        st.success("Cadastro realizado com sucesso!")
                        # Auto-login: define o estado de sessão automaticamente
                        st.session_state.logged_in = True
                        st.session_state.user_email = email
                        st.session_state.groq_api_key = api_key_input
                        st.rerun()
                    else:
                        st.error("Usuário já cadastrado!")
    
    st.stop()

# Após a autenticação e antes de exibir as demais funcionalidades, adicione:
if st.sidebar.button("Logout", key="logout_button"):
    # Lista de chaves relacionadas à sessão do usuário
    keys_to_clear = [
        "logged_in",
        "user_email",
        "groq_api_key",
        "chats",
        "current_chat",
        "chroma_client",
        "collection"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# --- Fluxo após a autenticação e configuração da API Key ---
# Configurar o event loop para evitar warnings do asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Agora, o usuário está autenticado e a API Key foi definida.
user_email = st.session_state.user_email
groq_api_key = st.session_state.groq_api_key
client = Groq(api_key=groq_api_key)

# Inicializa o cliente ChromaDB e a coleção exclusiva para o usuário
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = get_chroma_client()
    st.session_state.collection = get_or_create_collection(st.session_state.chroma_client, user_email)

# Gerenciamento de chats exclusivos para o usuário
if "chats" not in st.session_state:
    st.session_state.chats = load_chats(user_email)
if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")

# Sidebar: Gerenciamento de Chats e Upload de Arquivos
st.sidebar.subheader("📌 Seus Chats")
for chat_name in st.session_state.chats.keys():
    if st.sidebar.button(chat_name, key=f"chat_{chat_name}"):
        st.session_state.current_chat = chat_name
        st.rerun()

novo_chat = st.sidebar.text_input("Nome do novo chat")
if st.sidebar.button("➕ Criar Chat") and novo_chat:
    st.session_state.chats, criado = create_new_chat(st.session_state.chats, novo_chat)
    if criado:
        st.session_state.current_chat = novo_chat
        save_chats(st.session_state.chats, user_email)
        st.rerun()
    else:
        st.sidebar.warning("Esse nome já existe!")

# Opção para excluir o chat ativo
if st.session_state.current_chat:
    if st.sidebar.button(f"❌ Excluir Chat: {st.session_state.current_chat}"):
        st.session_state.chats = delete_chat(st.session_state.chats, st.session_state.current_chat)
        st.session_state.current_chat = None
        save_chats(st.session_state.chats, user_email)
        st.rerun()

from utils.logging_manager import log_interaction
user_email = st.session_state.user_email

# --- Upload de Arquivos ---
uploaded_file = st.sidebar.file_uploader("📂 Carregar documento", type=["pdf", "docx", "csv"])
if uploaded_file:
    text = process_document(uploaded_file)
    if text:
        embeddings = SentenceTransformer(MODEL_NAME).encode(text).tolist()
        st.session_state.collection.add(ids=[uploaded_file.name], documents=[text], embeddings=[embeddings])
        st.sidebar.success("Documento processado e armazenado!")
        # Log do arquivo enviado
        log_interaction(user_email, {
            "type": "file",
            "timestamp": str(pd.Timestamp.now()),
            "filename": uploaded_file.name,
            "details": "Arquivo processado com sucesso."
        })
    else:
        st.sidebar.error("Não foi possível extrair o texto do documento.")

# --- Interface do Chat ---
if st.session_state.current_chat:
    st.title(f"💬 Chat: {st.session_state.current_chat}")
    mensagens = st.session_state.chats[st.session_state.current_chat]
    for msg in mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Faça sua pergunta..."):
        # Log da pergunta do usuário
        log_interaction(user_email, {
            "type": "chat",
            "timestamp": str(pd.Timestamp.now()),
            "role": "user",
            "message": prompt
        })
        mensagens.append({"role": "user", "content": prompt})
        
        docs = st.session_state.collection.get()
        contextos = (
            "\n".join(f"{doc_id}: {doc[:500]}..." for doc_id, doc in zip(docs["ids"], docs["documents"]))
            if docs.get("documents")
            else "Nenhum documento carregado."
        )
        historico = "\n".join(f'{msg["role"].capitalize()}: {msg["content"]}' for msg in mensagens)
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
        
        # Log da resposta do bot
        log_interaction(user_email, {
            "type": "chat",
            "timestamp": str(pd.Timestamp.now()),
            "role": "assistant",
            "message": resposta
        })
        mensagens.append({"role": "assistant", "content": resposta})
        st.session_state.chats[st.session_state.current_chat] = mensagens
        save_chats(st.session_state.chats, user_email)
        st.rerun()
else:
    st.info("Selecione ou crie um chat na barra lateral.")

# Opções adicionais na Sidebar
if st.sidebar.button("📚 Ver documentos armazenados"):
    docs = st.session_state.collection.peek()
    if docs.get("ids"):
        st.sidebar.write("📌 Documentos armazenados:")
        for doc_id in docs["ids"]:
            st.sidebar.write(f"- {doc_id}")
    else:
        st.sidebar.write("Nenhum documento encontrado.")

if st.sidebar.button("🗑️ Limpar banco de dados"):
    collection, msg = delete_and_recreate_collection(st.session_state.chroma_client, user_email)
    st.session_state.collection = collection
    st.sidebar.info(msg)
