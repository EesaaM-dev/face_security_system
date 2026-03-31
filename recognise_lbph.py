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
    #start video capture and recognition loop
    video = cv.VideoCapture(0)
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
            #cv.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2) #display rectangle around detected face
            label, confidence = recogniser.predict(gray[y:y+h, x: x+w])
            #similarity = max(0, 100 - confidence) / 100  # Convert confidence to similarity score (0 to 1)
            if confidence < 70:
                result = "access_granted"
                name = label_dict.get(label, "unknown") #get name from dict, default to unknown if not found
                colour = (0,255,0) #green for known
            else:
                name = "unknown"
                result = "access_denied"
                colour = (0,0,255) #red for unknown 
            

            text=f"{name} ({confidence:.2f})"
            
            cv.rectangle(frame, (x,y),(x+w,y+h),colour,2) #rectangle color based on recognition result
           
            cv.putText(frame, text, (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
       
            cv.putText(frame, result, (x,y+h+25), cv.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        #display resulting frame with detected faces
        cv.imshow('Face ', frame)
        #exit loop when user presses q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    #release webcam and close opencv windows
    video.release()
    cv.destroyAllWindows()
