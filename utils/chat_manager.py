import json
import os
import re

def sanitize_username(username):
    """Sanitiza a parte do email (antes do '@') para uso em nomes de arquivos."""
    sanitized = re.sub(r'[^A-Za-z0-9_-]', '_', username)
    return sanitized.strip('_') or "user"

def get_logs_folder():
    """Retorna o caminho da pasta de logs secundários e cria-a se necessário."""
    folder = "logs_secundarios"
    os.makedirs(folder, exist_ok=True)
    return folder

def get_chat_filename(user_email):
    """Retorna o nome do arquivo de chats exclusivo para o usuário, armazenado na pasta logs_secundarios."""
    username = sanitize_username(user_email.split('@')[0])
    folder = get_logs_folder()
    return os.path.join(folder, f"chats_{username}.json")

def load_chats(user_email):
    """Carrega os chats do usuário a partir do arquivo exclusivo na pasta logs_secundarios."""
    file_name = get_chat_filename(user_email)
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats, user_email):
    """Salva os chats do usuário no arquivo exclusivo dentro da pasta logs_secundarios."""
    file_name = get_chat_filename(user_email)
    with open(file_name, "w") as f:
        json.dump(chats, f, indent=4)

def create_new_chat(chats, chat_name):
    """
    Cria um novo chat se não existir.
    Retorna uma tupla (chats, criado) onde 'criado' é True se o chat foi criado.
    """
    if chat_name in chats:
        return chats, False
    chats[chat_name] = []
    return chats, True

def delete_chat(chats, chat_name):
    """Exclui o chat especificado, se existir."""
    if chat_name in chats:
        del chats[chat_name]
    return chats
