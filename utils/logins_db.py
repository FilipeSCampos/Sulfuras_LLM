# utils/login_db.py

import hashlib
import sqlite3

DB_PATH = "login.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def initialize_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """
    )
    conn.commit()
    conn.close()


def create_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    try:
        cursor.execute(
            "INSERT INTO users (email, password) VALUES (?, ?)",
            (email, hashed_password),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # O usuário já existe
        return False
    finally:
        conn.close()


def validate_user(email, password):
    conn = get_connection()
    cursor = conn.cursor()
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute(
        "SELECT * FROM users WHERE email = ? AND password = ?",
        (email, hashed_password),
    )
    user = cursor.fetchone()
    conn.close()
    return user is not None
