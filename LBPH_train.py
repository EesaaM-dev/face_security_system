import cv2
import os
import numpy as np
import pickle


def lbph_train():
        
    faces = []
    ids = []
    label_dict = {}
    recogniser = cv2.face.LBPHFaceRecognizer.create()
    user_id_lt = os.listdir("C:/Users/thego/Documents/Year 3/openCV_tests/datasets")
    print(user_id_lt)
    for label, user_id in enumerate(user_id_lt):
        if label not in label_dict:
            label_dict[label] = user_id

        user_path = "C:/Users/thego/Documents/Year 3/openCV_tests/datasets/" + user_id
        for img in os.listdir(user_path):
            face_img_rgb = cv2.imread(user_path + "/" + img) #need to read the image+
            face_img = cv2.cvtColor(face_img_rgb, cv2.COLOR_BGR2GRAY)
            faces.append(face_img)
            ids.append(label)

    #print(ids)
    print(label_dict)
    print(ids)
    # print(faces)
    recogniser.train(faces, np.array(ids))
    recogniser.write("recogniser.yml")
    print("LBPH training complete")

    with open("label_dict.pkl", "wb") as file:
        pickle.dump(label_dict, file)
        file.close()