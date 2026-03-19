import sqlite3

#remove user id embedding from database
def remove_embedding(user_id):
    conn = sqlite3.connect("embeddings_db.db")
    c = conn.cursor()
    c.execute("DELETE FROM embeddings_db WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()

#remove_embedding()

