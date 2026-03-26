import numpy as np
import cv2 as cv
import time
import os
import sqlite3

#load pretrained haar cascade classifier
def lbph_recognition(app):


    face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

    recogniser = cv.face.LBPHFaceRecognizer.create()
    recogniser.read("recogniser.yml")

    #start video capture from webcam
    video= cv.VideoCapture(0)
    try:
        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT id, user_id FROM users")
        # This creates the map: {1: 'Steve', 2: 'Mark'}
        label_dict = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        print("loaded users from database")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        label_dict = {}

    video = cv.VideoCapture(0)
    app.running = True

    app.running = True
    while app.running:
        #capture each frame from webcam
        ret, frame = video.read()
        if not ret:
            print("failed to grab feed")
            break

        #convert frame to grayscale
        gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        #draw rectangle around detected faces 
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2) #display rectangle around detected face
            label, confidence = recogniser.predict(gray[y:y+h, x: x+w])
            #similarity = max(0, 100 - confidence) / 100  # Convert confidence to similarity score (0 to 1)
            if confidence < 60:
                name = label_dict[label]
                #change bounding box color to green if confidence is above threshold
                cv.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
                #change text color to green if confidence is above threshold
            else:
                name = "unknown"
            

            text=f"{name} ({confidence:.2f})"
            if confidence < 60:
                cv.putText(frame, text, (x,y-40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv.putText(frame, text, (x,y-40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        #display resulting frame with detected faces
        cv.imshow('Face ', frame)
        #exit loop when user presses q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    #release webcam and close opencv windows
    video.release()
    cv.destroyAllWindows()
