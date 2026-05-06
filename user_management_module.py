import sqlite3
import os
import shutil

def delete_user(user_id):
#removes users from datasets and dataets_CNN and deletes from database
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if not c.fetchone():
            conn.close()
            return False, "User not found"
        c.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()
        lbph_path = os.path.join("datasets", user_id)
        cnn_path = os.path.join("datasets_CNN", user_id)
        if os.path.exists(lbph_path):
            shutil.rmtree(lbph_path)
        if os.path.exists(cnn_path):
            shutil.rmtree(cnn_path)
        return True, f"User '{user_id}' deleted successfully"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"

#set account status to active or revoked
def set_account_status(user_id, status):
    if status not in ("Active", "Revoked"):
        return False, "Status must be 'Active' or 'Revoked'"
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if not c.fetchone():
            conn.close()
            return False, "User not found"
        c.execute("UPDATE users SET account_status=? WHERE user_id=?", (status, user_id))
        conn.commit()
        conn.close()
        return True, f"User '{user_id}' status set to {status}"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def set_clearance_level(user_id, level):
    try:
        level = int(level)
        if level < 1 or level > 5:
            return False, "Clearance level must be between 1 and 5"
    except ValueError:
        return False, "Clearance level must be a number"
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT user_id FROM users WHERE user_id=?", (user_id,))
        if not c.fetchone():
            conn.close()
            return False, "User not found"
        c.execute("UPDATE users SET clearance_lvl=? WHERE user_id=?", (level, user_id))
        conn.commit()
        conn.close()
        return True, f"User '{user_id}' clearance level set to {level}"
    except sqlite3.Error as e:
        return False, f"Database error: {e}"


def get_all_users():
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT user_id, account_status, clearance_lvl FROM users")
        rows = c.fetchall()
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def get_recent_logs(limit=50):
    try:
        conn = sqlite3.connect("logs.db")
        c = conn.cursor()
        c.execute("""
            SELECT user_id, time, result, score, method 
            FROM access_logs 
            ORDER BY id DESC 
            LIMIT ?
        """, (limit,))
        rows = c.fetchall()
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def get_account_status(user_id):
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT account_status FROM users WHERE user_id=?", (user_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else "User not found"
    
    except sqlite3.Error as e:
        return None