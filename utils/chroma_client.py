import re

from chromadb.utils import embedding_functions

import chromadb
from config import CHROMA_DB_PATH, MODEL_NAME


def get_chroma_client():
    """Returns a ChromaDB persistent client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def sanitize_username(username):
    """
    Sanitizes the username by replacing non-alphanumeric characters,
    underscores, or hyphens with underscores.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", username)
    sanitized = sanitized.strip("_")
    if not sanitized:
        sanitized = "user"
    return sanitized


def get_or_create_collection(client, user_email):
    """
    Retrieves an existing ChromaDB collection or creates a new one
    unique to the user's email.
    """
    username = user_email.split("@")[0]
    sanitized_username = sanitize_username(username)
    collection_name = f"document_embeddings_{sanitized_username}"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )


def delete_and_recreate_collection(client, user_email):
    """
    Deletes the user's existing ChromaDB collection and
    recreates a new, empty one.
    """
    try:
        username = user_email.split("@")[0]
        sanitized_username = sanitize_username(username)
        collection_name = f"document_embeddings_{sanitized_username}"
        client.delete_collection(name=collection_name)
        # Recria a coleção com o nome único
        collection = get_or_create_collection(client, user_email)
        return collection, "Coleção excluída e recriada com sucesso!"
    except Exception as e:
        return None, f"Erro ao manipular a coleção: {e}"
