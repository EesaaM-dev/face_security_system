import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
import pickle

# Load your precomputed embeddings
with open("embeddings_db.pkl", "rb") as f:
    embeddings_db = pickle.load(f)
print("Embeddings loaded for:", list(embeddings_db.keys()))
# Load InsightFace model
print("Loading model...")
app = FaceAnalysis(
    name='buffalo_s',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(256, 256))
# Start webcam
video = cv2.VideoCapture(0)
cv2.namedWindow("Face Recognition Demo", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Face Recognition Demo", cv2.WND_PROP_TOPMOST, 1)
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Detect all faces in the frame
    faces = app.get(frame)

    if faces:
        for face in faces:
            face_emb = face.normed_embedding
            best_match = None
            best_score = -1

            # Compare each detected face with database
            for user, db_emb in embeddings_db.items():
                sim = np.dot(face_emb, db_emb)  # cosine similarity
                if sim > best_score:
                    best_score = sim
                    best_match = user
            # Threshold for recognition
            if best_score > 0.6:
                name = best_match
            else:
                name = "unknown"

            # Draw bounding box and label
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{name} ({best_score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show video
    cv2.imshow("Face Recognition Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()



