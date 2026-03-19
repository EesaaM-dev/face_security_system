import numpy as np
import cv2 as cv
import time
import os
#load pretrained haar cascade classifier

def enrol_user(user_id, img_no = 150):
        
    face_detect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    id = user_id

    cnn_path = os.path.join("C:/Users/thego/Documents/Year 3/openCV_tests/datasets_CNN/", str(id))
    os.makedirs(cnn_path, exist_ok=True)

    new_folder = os.makedirs("C:/Users/thego/Documents/Year 3/openCV_tests/datasets/" + str(id), exist_ok=True)
    #start video capture from webcam
    video= cv.VideoCapture(0)
    cv.namedWindow('Face ', cv.WINDOW_NORMAL)
    cv.setWindowProperty('Face ', cv.WND_PROP_TOPMOST, 1)
    ct = 0
    last_recorded = time.time() #to control data collection rate
    while True:
        #capture each frame from webcam
        ret, frame = video.read()
        if not ret:
            print("failed to grab feed")
            break
        gray=cv.cvtColor(frame, cv.COLOR_BGR2GRAY) #convert frame to grayscale for face detection and lbph training

        #detect faces in the video feed
        faces = face_detect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        display_frame = frame.copy() #make a copy so that rectangle isnt drawn on actual frame
        #draw rectangle around detected faces 
        
        cv.putText(display_frame, f"count: {ct}/{img_no}", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        for (x, y, w, h) in faces:
            current = time.time()
            cv.rectangle(display_frame, (x, y), (x + w, y + h), (255,0,0), 2) #display rectangle around detected face
            print("OK")
            #capture images every 0.1 seconds 
            if current - last_recorded >=0.1: 
                #saving grayscale crops for lbph
                print("Captured image", ct)
                cv.imwrite('datasets/'  +'/' +str(id) + '/User_'+str(id) + "_" +"img" + str(ct) + ".jpg", gray[y: y+h, x:x + w]) #resize image to the bounding box area
               
                #saving rgb faces for the cnn
                cnn_crop = frame[y:y+h, x:x+w]              #crop color image
                cnn_crop = cv.resize(cnn_crop, (112,112))  #resize to expected input for cnn
                cv.imwrite(os.path.join(cnn_path, f"User_{id}_img{ct}.jpg"), cnn_crop)
                ct+=1
                last_recorded = current
        
        #display resulting frame with detected faces
        cv.imshow('Face ', display_frame)
        if ct ==img_no:
            print("Data collected for user", id)
            break
        #exit loop when user presses q
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    #release webcam and close opencv windows
    video.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    enrol_user(user_id)