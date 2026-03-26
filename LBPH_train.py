import cv2
import os
import numpy as np
import pickle
import sqlite3
root_path = "C:/Users/thego/Documents/Year 3/openCV_tests/datasets/"
def lbph_train():
        
    faces = []
    ids = []
    label_dict = {}
    recogniser = cv2.face.LBPHFaceRecognizer.create()
    conn = sqlite3.connect("users.db")
    c= conn.cursor()

    #get user ids from database
    c.execute("SELECT user_id, id FROM users")
    mapping_dict = {row[0]: row[1] for row in c.fetchall()} #create a mapping of user_id to label

    for user_id in os.listdir(root_path):
        if user_id in mapping_dict:
            actual_id = mapping_dict[user_id]
            print("the actual ID for user" + user_id + "is" + str(actual_id))
            user_path = os.path.join(root_path, user_id)
            for img in os.listdir(user_path):
                face_img_rgb = cv2.imread(user_path + "/" + img) #need to read the image+
                face_img = cv2.cvtColor(face_img_rgb, cv2.COLOR_BGR2GRAY)
                faces.append(face_img)
                ids.append(actual_id)

    #print(ids)
    print(label_dict)
    print(ids)
    # print(faces)
    recogniser.train(faces, np.array(ids))
    recogniser.write("recogniser.yml")
    print("LBPH training complete")

