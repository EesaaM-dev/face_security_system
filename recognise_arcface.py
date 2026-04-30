import os
import cv2
import numpy as np
import onnxruntime as ort
import sqlite3
from logger import log_attempt
import numpy as np
from user_management_module import get_account_status
import time

def arc_recognition(app):
    last_user = "unknown"
    log_cooldown = 2 #cooldown to prevent spamming logs
    fps_list = []
    logged_dict = {} #dict to handle logs for multiple users at once

    #loading onnx model 
    sess = ort.InferenceSession("w600k_r50.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    #dataset path and threshold
    dataset_path = "datasets_CNN"
    similarity_threshold = 0.6

    #haar for detecting faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess_face(face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112))
        face_norm = (face_resized - 127.5) / 128.0
        face_input = np.transpose(face_norm, (2,0,1))
        face_input = np.expand_dims(face_input, axis=0).astype(np.float32)
        return face_input
    #load from database it has embeddings for each user_id
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT user_id, embedding FROM users")
    rows = c.fetchall()
    embeddings_db = {}
    for user_id, avg_emb in rows:
        embeddings_db[user_id] = np.frombuffer(avg_emb, dtype=np.float32)
    conn.close()
    #connect to access logs database.
    #real time recognition 

    video = cv2.VideoCapture(0)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width", width)
    print("height", height)
    cv2.namedWindow("Arcface Face Recognition")
    cv2.resizeWindow("Arcface Face Recognition", 800,600)
    prev_time = time.time()
    fps =0
    cv2.setWindowProperty("Arcface Face Recognition", cv2.WND_PROP_TOPMOST, 1)
    app.running = True
    while app.running:
        ret, frame = video.read()
        if not ret:
            break
        curr_time = time.time()
        fps =1/(curr_time - prev_time)
        prev_time = curr_time
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x,y,w,h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_input = preprocess_face(face_crop)
            emb = sess.run([output_name], {input_name: face_input})[0].flatten()
            emb = emb / np.linalg.norm(emb)

            # Compare with per-image embeddings
            best_score = -1
            best_user = "unknown"
            for user, db_emb in embeddings_db.items():
                sim = np.dot(emb, db_emb)
                if sim > best_score:
                    best_score = sim
                    best_user = user
            if best_score < similarity_threshold:
                best_user = "unknown"
                result = "access_denied"
                display_name = "unknown"
                colour = (0,0,255) #red for unknown
                
            else:
                #get account status for best_user
                status = get_account_status(best_user)
                if status == "Revoked":
                    result = "access_denied"
                    colour = (0,0,255) #red for revoked
                    display_name = f"{best_user} (Revoked)"
                else:

                    result = "access_granted"
                    display_name = best_user
                    colour = (0,255,0) #green for known

               
            
            current_time = time.time()
            last_entry = logged_dict.get(best_user, 0)

            if best_user != last_user or current_time - last_entry > log_cooldown:
                #convert best_score to real data type for database
                best_score_db = float(best_score)
                log_attempt(best_user, result, best_score_db, "arcface") #log the attempt to database
                logged_dict[best_user] = current_time
                last_user = best_user
                
            text = f"{display_name} ({best_score:.2f})"
            cv2.rectangle(frame, (x,y), (x+w, y+h), colour, 2) #rectangle color based on recognition result
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
            cv2.putText(frame, result, (x,y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.imshow("Arcface Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


