import sqlite3
import os

def init_db():
    #create/connect to sql database
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    
    #create table if it doesn't exist
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL, clearance_lvl INTEGER NOT NULL DEFAULT 1, account_status TEXT NOT NULL DEFAULT 'Active'
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized as users.db")

    #create logs database for logging access attempts
    conn = sqlite3.connect("logs.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            time text,
            result TEXT,
            score REAL,
            method TEXT
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized as logs.db")

if __name__ == "__main__":
    init_db()