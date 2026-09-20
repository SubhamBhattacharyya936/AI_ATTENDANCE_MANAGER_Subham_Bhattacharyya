import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# ---------- Load known encodings safely ----------
if not os.path.exists('encodings.pkl'):
    print("[ERROR] encodings.pkl not found. Run register.py first.")
    raise SystemExit(1)

with open('encodings.pkl', 'rb') as f:
    try:
        known_encodings = pickle.load(f)
    except (EOFError, pickle.UnpicklingError):
        print("[ERROR] encodings.pkl is empty or corrupt. Delete it and run register.py again.")
        raise SystemExit(1)

if not known_encodings:
    print("[ERROR] No encodings found. Run register.py after adding photos to dataset/.")
    raise SystemExit(1)

print(f"[INFO] Loaded {len(known_encodings)} known face(s): {list(known_encodings.keys())}")

# ---------- Initialize InsightFace ----------
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

THRESHOLD = 0.45  # Cosine similarity threshold; adjust if needed

# Track who has been marked today to reduce duplicate disk writes
_marked_today = set()

def mark_attendance(name):
    """Append attendance to today's CSV, ignoring duplicates."""
    global _marked_today
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    os.makedirs("attendance", exist_ok=True)
    file_path = f"attendance/{today}.csv"

    # Fast in-memory duplicate check
    if name in _marked_today:
        return

    # Also check the file in case it was written in a previous run today
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f.readlines()[1:]:
                if line.startswith(f"{name},"):
                    _marked_today.add(name)
                    return
    else:
        with open(file_path, 'w') as f:
            f.write("Name,Date,Time\n")

    with open(file_path, 'a') as f:
        f.write(f"{name},{today},{time_now}\n")
    _marked_today.add(name)
    print(f"[ATTENDANCE] {name} marked present at {time_now}")


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check that no other app is using it.")
        return

    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame. Exiting.")
            break

        faces = app.get(frame)
        for face in faces:
            emb = face.embedding
            best_name = "Unknown"
            highest_sim = -1.0

            for name, known_emb in known_encodings.items():
                sim = float(
                    np.dot(emb, known_emb)
                    / (np.linalg.norm(emb) * np.linalg.norm(known_emb))
                )
                if sim > highest_sim:
                    highest_sim = sim
                    best_name = name

            box = face.bbox.astype(int)
            is_match = highest_sim > THRESHOLD
            color = (0, 255, 0) if is_match else (0, 0, 255)
            label = f"{best_name} ({highest_sim:.2f})" if is_match else "Unknown"

            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            if is_match:
                mark_attendance(best_name)

        cv2.imshow('Attendance System', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()