import cv2
import os
import numpy as np
from insightface.app import FaceAnalysis
import pickle

dataset_path = r"C:/Users/thego/Documents/Year 3/openCV_tests/datasets_CNN/"

user_id_lt = [
    d for d in os.listdir(dataset_path)
    if os.path.isdir(os.path.join(dataset_path, d))
]

print("Loading model...")
app = FaceAnalysis(
    name='buffalo_s',
    providers=['CPUExecutionProvider'],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(256, 256))  # big enough for your crops

embeddings_db = {}

for user in user_id_lt:
    print(f"Processing user: {user}")
    user_path = os.path.join(dataset_path, user)
    embeddings_lt = []

    for img_name in os.listdir(user_path):
        img_path = os.path.join(user_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"  Skipping {img_name} (cannot read)")
            continue

        faces = app.get(img)
        if not faces:
            print(f"  No face found in {img_name}, skipping")
            continue

        # pick the biggest face (just in case)
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        emb = face.normed_embedding  # 512-dim normalized embedding
        embeddings_lt.append(emb)

    if embeddings_lt:
        embeddings_db[user] = np.mean(embeddings_lt, axis=0)

print("Embeddings created for all users:")
for user, emb in embeddings_db.items():
    print(f"{user}: {emb.shape}, |emb|={np.linalg.norm(emb):.3f}")

with open("embeddings_db.pkl", "wb") as file:
    pickle.dump(embeddings_db, file)
    file.close()