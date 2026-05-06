import os
import cv2
import numpy as np
import onnxruntime as ort
import sqlite3

def generate_embeddings():
    
    #connect to the existing database else return error
    if not os.path.exists("users.db"):
        print("Error: users.db not found")
        return
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    #create table if it doesn't exist
    
    #load onnx model
    sess = ort.InferenceSession("w600k_r50.onnx")
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    #dataset path
    dataset_path = "datasets_CNN"  #dataset folder with subfolders containing images for each user


    #haar for face detection 

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    #preprocessing function for ONNX ArcFace
    def preprocess_face(face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (112, 112)) #resize to 112x112 as required by arc
        face_norm = (face_resized - 127.5) / 128.0  
        face_input = np.transpose(face_norm, (2,0,1))  #change the format to chw format as required by onnx model
        face_input = np.expand_dims(face_input, axis=0).astype(np.float32) #add batch dimension and convert to float32 (need to specify batch even if just one image)
        return face_input
    
    #generate average embeddings for each user in dataset

    user_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path,d))] 
    print("found user dirs:", user_dirs) #debug statement
    #loop through
    for user in user_dirs:
        user_path = os.path.join(dataset_path, user)
        user_embeddings = []

        for img_name in os.listdir(user_path):
            if not img_name.lower().endswith((".jpg",".png")):
                continue
            img_path = os.path.join(user_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                print(f"no face found in {img_name} skip")
                continue

            #pick largest face
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            face_crop = img[y:y+h, x:x+w]

            #compute embedding
            face_input = preprocess_face(face_crop)
            emb = sess.run([output_name], {input_name: face_input})[0].flatten()
            emb = emb / np.linalg.norm(emb)
            user_embeddings.append(emb)


        if user_embeddings:
            avg_emb = np.mean(user_embeddings, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)

            #check if user_id is already in database
            c.execute("SELECT user_id FROM users WHERE user_id=?", (user,))
            exists = c.fetchone()
            if exists:
                c.execute("UPDATE users SET embedding=? WHERE user_id=?", (avg_emb.tobytes(), user))
            else:
                c.execute("INSERT INTO users (user_id, embedding) VALUES (?, ?)", (user, avg_emb.tobytes()))
    
            conn.commit()
            
            print(f"Saved average embedding for {user}, {len(user_embeddings)} images")
    
    conn.close()
    print("Database saved as users")

if __name__ == "__main__":
    generate_embeddings()
