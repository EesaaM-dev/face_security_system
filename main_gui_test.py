import os
import tkinter as tk
from tkinter import messagebox, simpledialog
import threading
import cv2
from data_collection import enrol_user
from recognise_lbph import lbph_recognition
from CNN_embeddings_generation import generate_embeddings
from LBPH_train import lbph_train
from recognise_arcface import arc_recognition
from init_db import init_db
from user_management_module import delete_user, set_account_status, set_clearance_level, get_all_users

#initialise database on first run
if not os.path.exists("users.db") or not os.path.exists("logs.db"):
    init_db()



class Face_recognition:
    def __init__(self, root):
        self.root = root
        self.root.title("Face recognition system")
        self.root.geometry("400x500")
        self.running = False
        self.create_widgets()
    
    def create_widgets(self):
        tk.Label(self.root, text="Face recognition system", font=("Arial", 16, "bold")).pack(pady=20)
        tk.Button(self.root, text="Enrol user", width=25, command=self.enrol_user).pack(pady=5)
        tk.Button(self.root, text="LBPH train", width=25, command=self.train_lbph).pack(pady=5)
        tk.Button(self.root, text="Generate CNN embeddings", width=25, command=self.generate_embeddings).pack(pady=5)
        tk.Button(self.root, text="LBPH recognition", width=25, command=self.lbph_recognition).pack(pady=5)
        tk.Button(self.root, text="ArcFace recognition", width=25, command=self.arc_recognition).pack(pady=5)
        tk.Button(self.root, text="User Management", width=25, command=self.open_user_management).pack(pady=10)
        tk.Button(self.root, text="Exit", width=25, command=self.exit_program).pack(pady=5)


        
        #status label
        self.status_label = tk.Label(self.root, text="Status: Ready", font=("Arial", 10), fg="green")
        self.status_label.pack(side=tk.BOTTOM, pady=10)

    def set_status(self, message, colour="green"):
        #update status label from any thread safely
        self.root.after(0, lambda: self.status_label.config(text=f"Status: {message}", fg=colour))

    #functions for buttons (new threads to prevent GUI freezing)

    def enrol_user(self):
        user_id = simpledialog.askstring("Enrol User", "Enter User ID:")
        if user_id and not os.path.exists(os.path.join("datasets", user_id)):
            self.set_status("Enrolling user...", "orange")
            threading.Thread(target=self.enrol_thread, args=(user_id,)).start()
        else:
            messagebox.showerror("Error", "User ID already exists or is invalid.")

    def enrol_thread(self, user_id):
        enrol_user(user_id)
        self.set_status("Enrolment complete", "green")

    def train_lbph(self):
        self.set_status("Training LBPH model...", "orange")
        threading.Thread(target=self.lbph_train_thread).start()

    def lbph_train_thread(self):
        lbph_train()
        self.set_status("LBPH training complete", "green")

    def generate_embeddings(self):
        self.set_status("Generating embeddings...", "orange")
        threading.Thread(target=self.generate_embeddings_thread).start()

    def generate_embeddings_thread(self):
        generate_embeddings()
        self.set_status("Embeddings generated", "green")

    def arc_recognition(self):
        self.set_status("ArcFace recognition running...", "orange")
        threading.Thread(target=self.arc_recognition_thread).start()

    def arc_recognition_thread(self):
        arc_recognition(self)
        self.set_status("Ready", "green")

    def lbph_recognition(self):
        self.set_status("LBPH recognition running...", "orange")
        threading.Thread(target=self.lbph_recognition_thread).start()

    def lbph_recognition_thread(self):
        self.running = True
        lbph_recognition(self)
        self.set_status("Ready", "green")
    def view_users(self):
        users = get_all_users()
        if not users:
            messagebox.showinfo("Users", "No users enrolled.")
            return
 
        win = tk.Toplevel(self.root)
        win.title("Enrolled Users")
        win.geometry("400x300")
 
        tk.Label(win, text=f"{'User ID':<20}{'Status':<15}{'Clearance'}",
                font=("Courier", 11, "bold")).pack(pady=5)
 
        frame = tk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=10)
 
        scrollbar = tk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
 
        listbox = tk.Listbox(frame, font=("Courier", 11), yscrollcommand=scrollbar.set)
        for user_id, status, level in users:
            listbox.insert(tk.END, f"{user_id:<20}{status:<15}{level}")
        listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
    def open_user_management(self):
        #window to manage users
        win = tk.Toplevel(self.root)
        win.title("User Management")
        win.geometry("400x500")
        tk.Label(win, text="User Management", font=("Arial", 14, "bold")).pack(pady=100)
        tk.Button(win, text="View Users", width=25, command=self.view_users).pack(pady=5)
        tk.Button(win, text="Revoke / Activate User", width=25, command=self.toggle_status).pack(pady=5)
        tk.Button(win, text="Set Clearance Level", width=25, command=self.change_clearance).pack(pady=5)
        tk.Button(win, text="Delete User", width=25, command=self.delete_user).pack(pady=5)
        tk.Button(win, text="Exit", width=25, command=win.destroy).pack(pady=5)
 
    def toggle_status(self):
        user_id = simpledialog.askstring("Revoke/Activate User", "Enter User ID:")
        if not user_id:
            return
        status = simpledialog.askstring("Set Status", "Enter status (Active/Revoked):")
        if not status:
            return
        success, message = set_account_status(user_id, status.strip().capitalize())
        if success:
            messagebox.showinfo("Success", message)
            self.set_status(message, "green")
        else:
            messagebox.showerror("Error", message)
 
    def change_clearance(self):
        user_id = simpledialog.askstring("Set clearance level", "Enter User ID:")
        if not user_id:
            return
        level = simpledialog.askstring("Set clearance Level", "Enter clearance level (1-5):")
        if not level:
            return
        success, message = set_clearance_level(user_id, level)
        if success:
            messagebox.showinfo("Success", message)
            self.set_status(message, "green")
        else:
            messagebox.showerror("Error", message)
 
    def delete_user(self):
        user_id = simpledialog.askstring("Delete User", "Enter User ID to delete:")
        if not user_id:
            return
        confirm = messagebox.askyesno("Confirm delete",
                f"Are you sure you want to delete user '{user_id}'?\nThis will remove all their data.")
        if confirm:
            success, message = delete_user(user_id)
            if success:
                messagebox.showinfo("Success", message)
                self.set_status(message, "green")
            else:
                messagebox.showerror("Error", message)
 
    def exit_program(self):
        self.running = False
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = Face_recognition(root)
    print("GUI started")
    root.mainloop()