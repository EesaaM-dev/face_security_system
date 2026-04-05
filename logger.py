import sqlite3


#function to log access attempts to access_logs.db
def log_attempt(user_id, result, confidence, method):
    try:
        
        conn = sqlite3.connect("logs.db")
        c = conn.cursor()
        c.execute("INSERT INTO access_logs (user_id, time, result, confidence, method) VALUES (?, datetime('now'), ?, ?, ?)",
                (user_id, result, confidence, method))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Database error: {e}")


