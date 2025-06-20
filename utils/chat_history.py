# utils/chat_history.py
import csv
import json
import os
from datetime import datetime
from typing import Dict, List

HIST_DIR = "chat_logs"


def salvar_conversa_por_chat(chat_name: str, usuario: str, assistente: str):
    """Salva o histórico de um chat específico em JSON."""
    os.makedirs(HIST_DIR, exist_ok=True)
    path = os.path.join(HIST_DIR, f"{chat_name}.json")

    nova_interacao = {
        "usuario": usuario,
        "assistente": assistente,
        "timestamp": datetime.now().isoformat(),
    }

    historico = carregar_conversas_por_chat(chat_name)
    historico.append(nova_interacao)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2)


def carregar_conversas_por_chat(chat_name: str) -> List[Dict]:
    """Carrega o histórico de um chat específico."""
    path = os.path.join(HIST_DIR, f"{chat_name}.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exportar_conversa_para_csv(chat_name: str) -> str:
    """Exporta o histórico de um chat para CSV e retorna o caminho do arquivo."""
    historico = carregar_conversas_por_chat(chat_name)
    if not historico:
        return ""

    os.makedirs(HIST_DIR, exist_ok=True)
    csv_path = os.path.join(HIST_DIR, f"{chat_name}.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, fieldnames=["timestamp", "usuario", "assistente"]
        )
        writer.writeheader()
        writer.writerows(historico)

    return csv_path
