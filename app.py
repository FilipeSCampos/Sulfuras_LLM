# -------------------- IMPORTS PADRÃO --------------------
import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

# -------------------- IMPORTS LOCAIS (PROJETO) --------------------
from config import MODEL_NAME
from utils.api_key import initialize_api_key
from utils.chat_history import (
    carregar_conversas_por_chat,
    exportar_conversa_para_csv,
    salvar_conversa_por_chat,
)
from utils.chat_manager import create_new_chat, delete_chat, load_chats, save_chats
from utils.chroma_client import (
    delete_and_recreate_collection,
    get_chroma_client,
    get_or_create_collection,
)
from utils.logging_manager import log_interaction
from utils.logins_db import create_user, initialize_db, validate_user

# Redefine o módulo sqlite3 para usar pysqlite3
__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# -------------------- CONFIGURAÇÕES E CONSTANTES --------------------
SYSTEM_PROMPT = """Você é Sulfuras, um assistente inteligente criado por
Filipe Sampaio Campos, Rafael Canuto, Tatiana Hanada, Hermes e Vinicius,
sob orientação do M.e Weslley Rodrigues.

Você é profissional, prestativo e tem conhecimento técnico avançado.
Sempre responda com base no contexto fornecido pelos documentos carregados
e no histórico da conversa. Se não houver informações suficientes no
contexto, seja claro sobre isso e ofereça ajuda geral sobre o tópico."""

MAX_CONTEXT_LENGTH = 8000
MAX_TOKENS = 2048
TEMPERATURE = 0.5

# -------------------- CONFIGURAÇÃO STREAMLIT --------------------
st.set_page_config(
    page_title="Sulfuras Chatbot",
    page_icon="🔨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------- FUNÇÕES AUXILIARES --------------------
@st.cache_resource
def load_sentence_transformer():
    """Carrega o modelo SentenceTransformer com cache."""
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo de embeddings: {str(e)}")
        return None


def initialize_session_state():
    """Inicializa as variáveis de estado da sessão."""
    defaults = {
        "logged_in": False,
        "user_email": None,
        "groq_api_key": None,
        "chats": {},
        "current_chat": None,
        "chroma_client": None,
        "collection": None,
        "sentence_transformer": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def validate_api_key(api_key: str) -> bool:
    """Valida se a API key é válida fazendo uma requisição de teste."""
    if not api_key or len(api_key) < 10:
        return False

    try:
        client = Groq(api_key=api_key)
        # Teste simples para verificar se a API key funciona
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama3-8b-8192",
            max_tokens=1,
        )
        return True
    except Exception:
        return False


def truncate_context(context: str, max_length: int = MAX_CONTEXT_LENGTH) -> str:
    """Trunca o contexto para evitar exceder limites de token."""
    if len(context) <= max_length:
        return context
    return context[:max_length] + "...\n[Contexto truncado devido ao tamanho]"


def format_chat_history(messages: List[Dict], max_messages: int = 10) -> str:
    """Formata o histórico do chat limitando o número de mensagens."""
    recent_messages = (
        messages[-max_messages:] if len(messages) > max_messages else messages
    )
    return "\n".join(
        [
            f'{msg["role"].capitalize()}: '
            f'{msg["content"][:500]}'
            f'{"..." if len(msg["content"]) > 500 else ""}'
            for msg in recent_messages
        ]
    )


def save_feedback(feedback_data: Dict):
    """Salva feedback do usuário em arquivo JSON."""
    try:
        with open("feedback_log.json", "a", encoding="utf-8") as f:
            json_str = json.dumps(feedback_data, ensure_ascii=False, indent=2)
            f.write(json_str + "\n")
    except Exception as e:
        st.error(f"Erro ao salvar feedback: {str(e)}")


# -------------------- INICIALIZAÇÃO --------------------
initialize_session_state()
initialize_db()
initialize_api_key()


# -------------------- INTERFACE DE LOGIN/CADASTRO --------------------
def render_login_page():
    """Renderiza a página de login/cadastro."""
    st.title("🔨 Sulfuras: Chatbot Inteligente com Contexto")

    col_texto, col_imagem = st.columns([2, 1])

    with col_texto:
        st.markdown(
            """
            ### Bem-vindo ao Sulfuras! 🤖

            Este projeto é um chatbot inteligente capaz de compreender
            documentos carregados (PDF, DOCX ou CSV) e responder perguntas
            contextuais com alta precisão.

            **🚀 Tecnologias utilizadas:**
            - 🖥️ **Streamlit** - Interface gráfica moderna e responsiva
            - 🤖 **Groq (Llama)** - Modelo LLM de alta performance
            - 🧠 **ChromaDB** - Armazenamento vetorial eficiente
            - 📊 **Sentence Transformers** - Embeddings semânticos
            - 🔒 **Sistema de autenticação** - Controle de acesso seguro

            **👥 Desenvolvido por:**
            Filipe S. Campos, Rafael Canuto, Tatiana H., Hermes e Vinicius

            **🎓 Orientador:** M.e Weslley Rodrigues

            ---
            **Para começar, informe sua API Key do Groq, email e senha.**
            """
        )

    with col_imagem:
        try:
            st.image(
                "assets/sulfurs.webp", use_container_width=True, caption="Sulfuras AI"
            )
        except FileNotFoundError:
            st.info("🔨 Sulfuras AI\n\nImagem não encontrada")

    with st.sidebar:
        st.header("🔐 Acesso ao Sistema")
        opcao = st.radio("Selecione uma opção:", ["Login", "Cadastro"], horizontal=True)

        st.divider()

        # Input da API Key com validação
        api_key_input = st.text_input(
            "🔑 API Key do Groq",
            type="password",
            help="Obtenha sua API key em: https://console.groq.com/keys",
            key="api_key_input",
        )

        if api_key_input:
            with st.spinner("Validando API Key..."):
                if not validate_api_key(api_key_input):
                    st.error("❌ API Key inválida! Verifique e tente novamente.")
                    st.stop()
                else:
                    st.success("✅ API Key válida!")
        else:
            st.warning("⚠️ Por favor, insira sua API Key para continuar.")
            st.stop()

        st.divider()

        if opcao == "Login":
            render_login_form(api_key_input)
        else:
            render_register_form(api_key_input)


def render_login_form(api_key: str):
    """Renderiza o formulário de login."""
    st.subheader("🔓 Fazer Login")

    with st.form("login_form"):
        email = st.text_input("📧 Email", placeholder="seu@email.com")
        password = st.text_input("🔒 Senha", type="password")

        submitted = st.form_submit_button("🚀 Entrar", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("❌ Por favor, preencha todos os campos.")
            elif validate_user(email, password):
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.groq_api_key = api_key
                st.success("✅ Login realizado com sucesso!")
                st.rerun()
            else:
                st.error("❌ Email ou senha incorretos!")


def render_register_form(api_key: str):
    """Renderiza o formulário de cadastro."""
    st.subheader("📝 Criar Conta")

    with st.form("register_form"):
        email = st.text_input("📧 Email", placeholder="seu@email.com")
        password = st.text_input("🔒 Senha", type="password")
        confirm_password = st.text_input("🔒 Confirmar Senha", type="password")

        submitted = st.form_submit_button("🎉 Cadastrar", use_container_width=True)

        if submitted:
            if not all([email, password, confirm_password]):
                st.error("❌ Por favor, preencha todos os campos.")
            elif password != confirm_password:
                st.error("❌ As senhas não coincidem!")
            elif len(password) < 6:
                st.error("❌ A senha deve ter pelo menos 6 caracteres.")
            elif create_user(email, password):
                st.success("✅ Cadastro realizado com sucesso!")
                st.session_state.logged_in = True
                st.session_state.user_email = email
                st.session_state.groq_api_key = api_key
                st.rerun()
            else:
                st.error("❌ Este email já está cadastrado!")


# -------------------- INTERFACE PRINCIPAL --------------------
def render_main_app():
    """Renderiza a interface principal do aplicativo."""
    # Inicialização do cliente e coleção
    if not st.session_state.chroma_client:
        st.session_state.chroma_client = get_chroma_client()
        st.session_state.collection = get_or_create_collection(
            st.session_state.chroma_client, st.session_state.user_email
        )

    if not st.session_state.sentence_transformer:
        st.session_state.sentence_transformer = load_sentence_transformer()

    if not st.session_state.chats:
        st.session_state.chats = load_chats(st.session_state.user_email)

    # Header
    st.title("🔨 Sulfuras: Chatbot Inteligente")
    st.markdown(f"**👤 Usuário:** {st.session_state.user_email}")

    # Sidebar
    render_sidebar()

    # Chat interface
    if st.session_state.current_chat:
        render_chat_interface()
    else:
        render_welcome_screen()


def render_sidebar():
    """Renderiza a barra lateral com controles."""
    with st.sidebar:
        st.header("🎛️ Controles")

        # Logout
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key.startswith(
                    (
                        "logged_in",
                        "user_email",
                        "groq_api_key",
                        "chats",
                        "current_chat",
                        "chroma",
                    )
                ):
                    del st.session_state[key]
            st.rerun()

        st.divider()

        # Gerenciamento de chats
        st.subheader("💬 Gerenciar Chats")

        # Lista de chats existentes
        if st.session_state.chats:
            st.write("**Seus chats:**")
            for chat_name in st.session_state.chats.keys():
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(
                        f"💭 {chat_name}",
                        key=f"chat_{chat_name}",
                        use_container_width=True,
                    ):
                        st.session_state.current_chat = chat_name
                        st.rerun()
                with col2:
                    if st.button(
                        "🗑️", key=f"delete_{chat_name}", help=f"Excluir {chat_name}"
                    ):
                        st.session_state.chats = delete_chat(
                            st.session_state.chats, chat_name
                        )
                        if st.session_state.current_chat == chat_name:
                            st.session_state.current_chat = None
                        save_chats(st.session_state.chats, st.session_state.user_email)
                        st.rerun()

        # Criar novo chat
        with st.form("new_chat_form"):
            novo_chat = st.text_input(
                "📝 Nome do novo chat", placeholder="Ex: Análise de Vendas"
            )
            if st.form_submit_button("➕ Criar Chat", use_container_width=True):
                if novo_chat.strip():
                    st.session_state.chats, criado = create_new_chat(
                        st.session_state.chats, novo_chat.strip()
                    )
                    if criado:
                        st.session_state.current_chat = novo_chat.strip()
                        save_chats(st.session_state.chats, st.session_state.user_email)
                        st.rerun()
                    else:
                        st.error("❌ Nome já existe!")

        st.divider()

        # Upload de documentos
        st.subheader("📂 Carregar Documentos")
        uploaded_file = st.file_uploader(
            "Selecione um arquivo",
            type=["pdf", "docx", "csv", "txt", "html"],
            help="Formatos suportados: PDF, DOCX, CSV, TXT, HTML",
        )

        if uploaded_file and st.session_state.sentence_transformer:
            process_uploaded_file(uploaded_file)

        st.divider()

        # Ferramentas
        st.subheader("🛠️ Ferramentas")

        if st.button("📚 Ver Documentos", use_container_width=True):
            show_stored_documents()

        if st.button("🗑️ Limpar Base", use_container_width=True):
            clear_database()

        if st.session_state.current_chat:
            st.divider()
            st.subheader("🕓 Histórico e Exportação")

            if st.button("📜 Ver Histórico", use_container_width=True):
                historico = carregar_conversas_por_chat(st.session_state.current_chat)
                if historico:
                    st.success(f"✅ {len(historico)} interações salvas:")
                    for i, item in enumerate(historico):
                        st.markdown(f"**{i+1}. {item['timestamp']}**")
                        st.markdown(f"- **Usuário:** {item['usuario']}")
                        st.markdown(f"- **Assistente:** {item['assistente']}")
                        st.markdown("---")
                else:
                    st.info("Nenhum histórico encontrado para este chat.")

            if st.button("📤 Exportar para CSV", use_container_width=True):
                caminho = exportar_conversa_para_csv(st.session_state.current_chat)
                if caminho:
                    with open(caminho, "rb") as f:
                        st.download_button(
                            label="⬇️ Baixar CSV",
                            data=f,
                            file_name=os.path.basename(caminho),
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:
                    st.warning("⚠️ Nenhum histórico disponível para exportar.")


def process_uploaded_file(uploaded_file):
    """Processa o arquivo carregado."""
    try:
        with st.spinner("📖 Processando documento..."):
            from utils.file_processing import process_document
            from utils.file_processors.html_loader import load_html
            from utils.file_processors.txt_loader import load_txt

            file_extension = uploaded_file.name.split(".")[-1].lower()

            if file_extension == "txt":
                text = load_txt(uploaded_file)
            elif file_extension == "html":
                text = load_html(uploaded_file)
            else:
                text = process_document(uploaded_file)

            if text:
                # Gera embeddings
                embeddings = st.session_state.sentence_transformer.encode(text).tolist()

                # Armazena no ChromaDB
                st.session_state.collection.add(
                    ids=[f"{uploaded_file.name}_{datetime.now().isoformat()}"],
                    documents=[text],
                    embeddings=[embeddings],
                    metadatas=[
                        {
                            "filename": uploaded_file.name,
                            "upload_date": datetime.now().isoformat(),
                            "file_type": uploaded_file.type,
                            "file_size": uploaded_file.size,
                        }
                    ],
                )

                st.success(f"✅ **{uploaded_file.name}** processado com sucesso!")

                # Log da interação
                log_interaction(
                    st.session_state.user_email,
                    {
                        "type": "file_upload",
                        "timestamp": datetime.now().isoformat(),
                        "filename": uploaded_file.name,
                        "file_type": uploaded_file.type,
                        "file_size": uploaded_file.size,
                        "status": "success",
                    },
                )
            else:
                st.error("❌ Não foi possível extrair texto do documento.")

    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")


def show_stored_documents():
    """Mostra os documentos armazenados."""
    try:
        docs = st.session_state.collection.peek()
        if docs.get("ids"):
            st.success(f"📊 **{len(docs['ids'])} documentos** encontrados:")
            for i, (doc_id, metadata) in enumerate(
                zip(docs["ids"], docs.get("metadatas", []))
            ):
                if metadata:
                    st.write(
                        f"**{i+1}.** {metadata.get('filename', doc_id)} "
                        f"({metadata.get('file_type', 'N/A')}) - "
                        f"{metadata.get('upload_date', 'N/A')}"
                    )
                else:
                    st.write(f"**{i+1}.** {doc_id}")
        else:
            st.info("📝 Nenhum documento encontrado.")
    except Exception as e:
        st.error(f"❌ Erro ao listar documentos: {str(e)}")


def clear_database():
    """Limpa a base de dados."""
    try:
        collection, msg = delete_and_recreate_collection(
            st.session_state.chroma_client, st.session_state.user_email
        )
        st.session_state.collection = collection
        st.success(f"✅ {msg}")
    except Exception as e:
        st.error(f"❌ Erro ao limpar base: {str(e)}")


def render_welcome_screen():
    """Renderiza a tela de boas-vindas."""
    st.markdown(
        """
    ## 🎯 Como usar o Sulfuras

    1. **📂 Carregue documentos** - Use a barra lateral para carregar
       PDFs, DOCX ou CSVs
    2. **💬 Crie um chat** - Dê um nome descritivo para organizar
       suas conversas
    3. **❓ Faça perguntas** - O Sulfuras responderá com base nos
       documentos carregados
    4. **📊 Obtenha insights** - Análises inteligentes sobre seus dados

    ### 🚀 Recursos Disponíveis:
    - ✅ Análise contextual de documentos
    - ✅ Múltiplos chats organizados
    - ✅ Histórico de conversas
    - ✅ Sistema de feedback
    - ✅ Base de conhecimento persistente

    **👈 Comece criando um novo chat na barra lateral!**
    """
    )


def render_chat_interface():
    """Renderiza a interface do chat."""
    st.subheader(f"💬 Chat: **{st.session_state.current_chat}**")

    mensagens = st.session_state.chats[st.session_state.current_chat]

    # Container para mensagens
    chat_container = st.container()

    with chat_container:
        for i, msg in enumerate(mensagens):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

                # Botão de feedback para respostas do assistente
                if msg["role"] == "assistant" and i == len(mensagens) - 1:
                    render_feedback_section(i, msg)

    # Input de nova mensagem
    if prompt := st.chat_input("💭 Digite sua pergunta..."):
        handle_user_message(prompt, mensagens)


def render_feedback_section(msg_index: int, message: dict):
    """Renderiza a seção de feedback."""
    feedback_key = f"feedback_{st.session_state.current_chat}_{msg_index}"

    # Estilo CSS customizado para os botões
    st.markdown(
        """
    <style>
    .feedback-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        margin: 4px;
        border: 1px solid #ddd;
        border-radius: 6px;
        background-color: transparent;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .feedback-button:hover {
        background-color: #f0f0f0;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("👍 Resposta útil", key=f"{feedback_key}_pos"):
            save_feedback(
                {
                    "user": st.session_state.user_email,
                    "chat": st.session_state.current_chat,
                    "message_index": msg_index,
                    "feedback": "positive",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            st.success("✅ Obrigado pelo feedback!")

    with col2:
        if st.button("👎 Resposta não útil", key=f"{feedback_key}_neg"):
            save_feedback(
                {
                    "user": st.session_state.user_email,
                    "chat": st.session_state.current_chat,
                    "message_index": msg_index,
                    "feedback": "negative",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            st.success("✅ Obrigado pelo feedback!")


def handle_user_message(prompt: str, mensagens: List[Dict]):
    """Processa a mensagem do usuário."""
    # Adiciona mensagem do usuário
    mensagens.append({"role": "user", "content": prompt})

    # Log da interação
    log_interaction(
        st.session_state.user_email,
        {
            "type": "user_message",
            "timestamp": datetime.now().isoformat(),
            "chat": st.session_state.current_chat,
            "message": prompt,
        },
    )

    # Busca contexto relevante
    try:
        docs = st.session_state.collection.query(
            query_texts=[prompt], n_results=min(5, st.session_state.collection.count())
        )

        contextos = ""
        if docs.get("documents") and docs["documents"][0]:
            contextos = "\n\n".join(
                [
                    f"**Documento {i+1}:** "
                    f"{doc[:1000]}{'...' if len(doc) > 1000 else ''}"
                    for i, doc in enumerate(docs["documents"][0])
                ]
            )

        if not contextos:
            contextos = (
                "Nenhum documento relevante encontrado " "na base de conhecimento."
            )

    except Exception as e:
        contextos = f"Erro ao buscar contexto: {str(e)}"

    # Prepara o prompt final
    # Exclui a última mensagem (atual)
    historico = format_chat_history(mensagens[:-1])
    contextos = truncate_context(contextos)

    prompt_final = f"""{SYSTEM_PROMPT}

**CONTEXTO DOS DOCUMENTOS:**
{contextos}

**HISTÓRICO DA CONVERSA:**
{historico}

**PERGUNTA ATUAL:**
{prompt}

**INSTRUÇÕES:**
- Responda de forma detalhada e precisa
- Use o contexto fornecido como base principal
- Se não houver informações suficientes, seja transparente sobre isso
- Mantenha um tom profissional e prestativo
"""

    # Gera resposta
    try:
        with st.spinner("🤔 Gerando resposta..."):
            client = Groq(api_key=st.session_state.groq_api_key)
            resposta = (
                client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt_final},
                    ],
                    model="llama3-8b-8192",
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                .choices[0]
                .message.content
            )

    except Exception as e:
        resposta = (
            f"⚠️ **Erro ao gerar resposta:** {str(e)}\n\n"
            "Por favor, verifique sua conexão e tente novamente."
        )

    # Adiciona resposta do assistente
    mensagens.append({"role": "assistant", "content": resposta})

    # ✅ Salva o histórico desse chat individualmente
    salvar_conversa_por_chat(st.session_state.current_chat, prompt, resposta)

    # Log da resposta
    log_interaction(
        st.session_state.user_email,
        {
            "type": "assistant_response",
            "timestamp": datetime.now().isoformat(),
            "chat": st.session_state.current_chat,
            "response": (resposta[:500] + "..." if len(resposta) > 500 else resposta),
        },
    )

    # Salva o chat atualizado
    st.session_state.chats[st.session_state.current_chat] = mensagens
    save_chats(st.session_state.chats, st.session_state.user_email)

    # Atualiza a interface
    st.rerun()


# -------------------- CONFIGURAÇÃO DO ASYNCIO --------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# -------------------- EXECUÇÃO PRINCIPAL --------------------
def main():
    """Função principal da aplicação."""
    if not st.session_state.get("logged_in") or not st.session_state.get(
        "groq_api_key"
    ):
        render_login_page()
    else:
        render_main_app()


if __name__ == "__main__":
    main()
