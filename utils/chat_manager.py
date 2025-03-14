# utils/chat_manager.py

import json
import os

def load_chats(user_email):
    file_name = f"chats_{user_email}.json"
    if os.path.exists(file_name):
        with open(file_name, "r") as file:
            return json.load(file)
    return {}

def save_chats(chats, user_email):
    file_name = f"chats_{user_email}.json"
    with open(file_name, "w") as file:
        json.dump(chats, file)

def create_new_chat(chats, chat_name):
    if chat_name not in chats:
        chats[chat_name] = []
        return chats, True
    return chats, False

def delete_chat(chats, chat_name):
    if chat_name in chats:
        del chats[chat_name]
    return chats
