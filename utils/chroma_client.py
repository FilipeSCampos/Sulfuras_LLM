# utils/chroma_client.py

import re
import chromadb
from chromadb.utils import embedding_functions
from config import CHROMA_DB_PATH, MODEL_NAME

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def sanitize_username(username):
    # Substitui qualquer caractere que não seja alfanumérico, underline ou hífen por underline
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '_', username)
    # Garante que não comece nem termine com underline (opcional)
    sanitized = sanitized.strip('_')
    # Se o nome ficar vazio, usa um nome padrão
    if not sanitized:
        sanitized = "user"
    return sanitized

def get_or_create_collection(client, user_email):
    # Extrai a parte do email antes do @ e sanitiza
    username = user_email.split('@')[0]
    sanitized_username = sanitize_username(username)
    collection_name = f"document_embeddings_{sanitized_username}"
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
    )

def delete_and_recreate_collection(client, user_email):
    try:
        username = user_email.split('@')[0]
        sanitized_username = sanitize_username(username)
        collection_name = f"document_embeddings_{sanitized_username}"
        client.delete_collection(name=collection_name)
        # Recria a coleção com o nome único
        collection = get_or_create_collection(client, user_email)
        return collection, "Coleção excluída e recriada com sucesso!"
    except Exception as e:
        return None, f"Erro ao manipular a coleção: {e}"
