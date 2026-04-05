import os
import tkinter as tk
from tkinter import messagebox, simpledialog
import threading
import cv2
from PIL import Image, ImageTk
from data_collection import enrol_user
from recognise_lbph import lbph_recognition
from onxx_embeddings_generation import generate_embeddings
from LBPH_train import lbph_train
from recognise_arcface import arc_recognition
from init_db import init_db

#initialise database on first run
if not os.path.exists("users.db") or not os.path.exists("access_logs.db"):
    init_db()
class Face_recognition:

    def __init__(self, root):
        self.root = root
        self.root.title("Face recognition system")
        self.root.geometry("400x400")
        self.running = False
        self.create_widgets()


    def create_widgets(self):
        #title
        tk.Label(self.root, text="Face recognition system", font=("Arial", 16, "bold")).pack(pady=20)

        #buttons
        tk.Button(self.root, text="Enrol user", width=25, command=self.enrol_user).pack(pady=5)
        tk.Button(self.root, text="LBPH train", width=25, command=self.train_lbph).pack(pady=5)
        tk.Button(self.root, text="Generate CNN embeddings", width=25, command=self.generate_embeddings).pack(pady=5)
        tk.Button(self.root, text="LBPH recognition", width=25, command=self.lbph_recognition).pack(pady=5)
        tk.Button(self.root, text="ArcFace recognition", width=25, command=self.arc_recognition).pack(pady=5)
        tk.Button(self.root, text="Exit", width=25, command=self.exit_program).pack(pady=5)
        #status label

        # self.status_label = tk.Label(self.root, text="Status: Ready", font=("Arial", 10))
        
        # self.status_label.pack(side=tk.BOTTOM, pady=10)

    #functions for buttons (new threads to prevent GUI freezing)
 
    def enrol_user(self):
        user_id = simpledialog.askstring("Enrol User", "Enter User ID:")
        if user_id and not os.path.exists(os.path.join("datasets", user_id)):
            threading.Thread(target=self.enrol_thread, args=(user_id,)).start() #start enrolment in new thread
        else:
            messagebox.showerror("Error", "User ID already exists or is invalid.")

    def enrol_thread(self, user_id):
        enrol_user(user_id)

    def train_lbph(self):
        threading.Thread(target=self.lbph_train_thread).start()

    def lbph_train_thread(self):
        lbph_train()
        

    def generate_embeddings(self):
        threading.Thread(target=self.generate_embeddings_thread).start()

    def generate_embeddings_thread(self):
        generate_embeddings()
    

    def arc_recognition(self):
        threading.Thread(target=self.arc_recognition_thread).start()

    def arc_recognition_thread(self):
        arc_recognition(self)
  

    def lbph_recognition(self):
       
        threading.Thread(target=self.lbph_recognition_thread).start()

    def lbph_recognition_thread(self):
        self.running = True
        lbph_recognition(self)
        

    def set_status(self, message):
        #setter for status label
        self.status_label.config(text=f"Status: {message}")
    def exit_program(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Face_recognition(root)
    #initialise database on first run

    print("GUI started")
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Program interrupted by user")
        root.destroy()



    