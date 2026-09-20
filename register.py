import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Initialize InsightFace
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

DATASET_DIR = 'dataset'
ENCODINGS_FILE = 'encodings.pkl'


def load_existing_encodings():
    """Safely load encodings, returning {} if file is missing or corrupt."""
    if not os.path.exists(ENCODINGS_FILE):
        return {}
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except (EOFError, pickle.UnpicklingError, Exception) as e:
        print(f"[WARN] Could not read existing encodings ({e}). Starting fresh.")
        return {}


def generate_encodings():
    if not os.path.exists(DATASET_DIR) or not os.listdir(DATASET_DIR):
        print(f"[ERROR] '{DATASET_DIR}' is missing or empty.")
        print("Create the folder and add one subfolder per person, each with 3-5 photos.")
        return

    encodings = load_existing_encodings()

    new_count = 0
    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        if person_name in encodings:
            print(f"[SKIP] {person_name} already encoded.")
            continue

        person_embeddings = []
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(person_dir, filename)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"[WARN] Could not read {img_path}")
                    continue
                faces = app.get(img)
                if faces:
                    person_embeddings.append(faces[0].embedding)

        if person_embeddings:
            encodings[person_name] = np.mean(person_embeddings, axis=0)
            new_count += 1
            print(f"[OK] Encoded {person_name} ({len(person_embeddings)} photos)")
        else:
            print(f"[WARN] No face detected for {person_name}. Skipping.")

    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(encodings, f)
    print(f"[DONE] Saved {len(encodings)} encodings ({new_count} new).")


if __name__ == "__main__":
    generate_encodings()