import json
import os
import re

# Tenta importar docling, mas não há suporte para criar documentos
# a partir de dicionários.
try:
    # Importado apenas para verificar disponibilidade
    import docling  # noqa: F401

    USE_DOCLING = True
except ImportError:
    USE_DOCLING = False


def sanitize_username(username):
    """Extrai e sanitiza a parte do email antes do '@'."""
    sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", username)
    return sanitized.strip("_") or "user"


def get_user_log_path(user_email):
    """Retorna o caminho do arquivo de log para um determinado usuário."""
    username = sanitize_username(user_email.split("@")[0])
    log_folder = os.path.join("logs", username)
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "log.json")
    return log_file


def load_logs(user_email):
    """Carrega os logs existentes para o usuário."""
    log_file = get_user_log_path(user_email)
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            return json.load(f)
    return []


def save_logs(user_email, logs):
    """Salva os logs para o usuário."""
    log_file = get_user_log_path(user_email)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)


def log_interaction(user_email, log_entry):
    """
    Registra uma nova entrada de log para o usuário.
    log_entry deve ser um dicionário, por exemplo:
      {
          "type": "chat",  # ou "file"
          "timestamp": "2023-03-14T12:34:56",
          "role": "user" ou "assistant",
          "message": "conteúdo da mensagem",
          "details": {}  # informações adicionais (opcional)
      }
    """
    logs = load_logs(user_email)
    logs.append(log_entry)
    save_logs(user_email, logs)

    # A integração com docling não é possível pois a biblioteca
    # não fornece um método para criar um documento a partir
    # de um dicionário arbitrário.
    if USE_DOCLING:
        # Poderíamos, em teoria, converter um arquivo JSON para
        # um documento docling usando outros métodos da biblioteca,
        # mas isso foge ao escopo deste exemplo.
        pass
