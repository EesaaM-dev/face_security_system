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

if __name__ == "__main__":
    init_db()