Generate encodings:

powershell
python register.py
Start real-time attendance:

powershell
python attendance.py
View dashboard (optional):

powershell
streamlit run dashboard.py

powershell
python register_people.py





AI_ATTENDANCE_MANAGER/
├── venv/                     # Virtual environment
├── dataset/                  # Folder for student reference photos
│   ├── John_Doe/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Jane_Smith/
│       └── photo1.jpg
├── attendance/               # Auto-created attendance logs
├── encodings.pkl             # Auto-generated face encodings
├── register.py               # Generates encodings from dataset
├── attendance.py             # Real-time attendance via webcam
├── dashboard.py              # Streamlit dashboard for viewing records
└── requirements.txt

C:\Users\brand\OneDrive\Desktop\AI_ATTENDANCE_MANAGER\attendance.py==
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


C:\Users\brand\OneDrive\Desktop\AI_ATTENDANCE_MANAGER\register.py===
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






C:\Users\brand\OneDrive\Desktop\AI_ATTENDANCE_MANAGER\dashboard.py==import streamlit as st
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Attendance Dashboard", layout="wide")
st.title("📊 Attendance Dashboard")

today = datetime.now().strftime("%Y-%m-%d")
file_path = f"attendance/{today}.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.subheader(f"Attendance for {today}")
    st.dataframe(df, use_container_width=True)
    st.metric("Total Present", len(df))
else:
    st.info("No attendance recorded for today yet.")

# Show all historical records
st.subheader("All Records")
all_records = []
if os.path.exists("attendance"):
    for f in os.listdir("attendance"):
        if f.endswith(".csv"):
            all_records.append(pd.read_csv(os.path.join("attendance", f)))

if all_records:
    combined = pd.concat(all_records, ignore_index=True)
    st.dataframe(combined, use_container_width=True)
else:
    st.info("No historical records found.")



C:\Users\brand\OneDrive\Desktop\AI_ATTENDANCE_MANAGER\register_people.py==

"""
Interactive registration tool.

Two modes:
  1. Webcam  - capture N photos with your camera (SPACE to snap, Q to finish early)
  2. Import  - copy images from another folder on disk

Usage:
    python register_people.py
"""

import os
import shutil
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pkl"
NUM_PHOTOS = 5           # how many photos to capture in webcam mode
MIN_FACE_SIZE = 60       # ignore detections smaller than this (pixels)

# ---------- InsightFace (used only to verify a face is present) ----------
print("[INFO] Loading InsightFace model (first run may take a moment)...")
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))


# ---------- helpers ----------
def ask_name():
    """Ask for a valid person name."""
    while True:
        name = input("\nEnter the person's name (e.g. John_Doe): ").strip()
        if not name:
            print("Name cannot be empty.")
            continue
        # Sanitize: replace spaces with underscores, strip unsafe characters
        safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        if safe != name:
            print(f"[INFO] Using sanitized name: {safe}")
        return safe


def person_dir(name):
    path = os.path.join(DATASET_DIR, name)
    os.makedirs(path, exist_ok=True)
    return path


def has_face(image_bgr, min_size=MIN_FACE_SIZE):
    """Return True if at least one reasonably-sized face is detected."""
    faces = app.get(image_bgr)
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
            return True
    return False


def rebuild_encodings():
    """Recompute encodings.pkl from the entire dataset/ folder."""
    encodings = {}
    if not os.path.isdir(DATASET_DIR):
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(encodings, f)
        return encodings

    for person in os.listdir(DATASET_DIR):
        pdir = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(pdir):
            continue
        embs = []
        for fn in os.listdir(pdir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                img = cv2.imread(os.path.join(pdir, fn))
                if img is None:
                    continue
                faces = app.get(img)
                if faces:
                    embs.append(faces[0].embedding)
        if embs:
            encodings[person] = np.mean(embs, axis=0)
            print(f"  [OK] {person} ({len(embs)} photos)")

    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)
    print(f"[DONE] encodings.pkl rebuilt with {len(encodings)} person(s).")
    return encodings


# ---------- Mode 1: webcam capture ----------
def register_webcam(name):
    target = person_dir(name)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return 0

    saved = 0
    print(f"\n[WEBCAM] Capturing up to {NUM_PHOTOS} photos for '{name}'.")
    print("  - Look at the camera, keep your face inside the green box")
    print("  - Press SPACE to capture, Q to finish early")

    while saved < NUM_PHOTOS:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed.")
            break

        # Live preview with face boxes
        preview = frame.copy()
        faces = app.get(frame)
        face_ok = False
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            if (x2 - x1) >= MIN_FACE_SIZE and (y2 - y1) >= MIN_FACE_SIZE:
                face_ok = True
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 165, 255), 2)

        status = f"Captured {saved}/{NUM_PHOTOS}"
        hint = "SPACE = capture   Q = finish"
        cv2.putText(preview, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(preview, hint, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        if not face_ok:
            cv2.putText(preview, "No face / face too small",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Register - press SPACE / Q", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord(' '):
            if not face_ok:
                print("[WARN] No suitable face detected - photo skipped.")
                continue
            saved += 1
            out_path = os.path.join(target, f"{name}_{saved:02d}.jpg")
            cv2.imwrite(out_path, frame)
            print(f"  [SAVED] {out_path}")
            # small visual feedback flash
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (flash.shape[1], flash.shape[0]),
                          (0, 255, 0), 20)
            cv2.imshow("Register - press SPACE / Q", flash)
            cv2.waitKey(150)

    cap.release()
    cv2.destroyAllWindows()
    return saved


# ---------- Mode 2: import from disk ----------
def register_from_folder(name):
    src = input("Enter the full path to the folder containing the photos: ").strip().strip('"')
    if not os.path.isdir(src):
        print(f"[ERROR] Not a folder: {src}")
        return 0

    target = person_dir(name)
    copied = 0
    for fn in os.listdir(src):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        src_path = os.path.join(src, fn)
        img = cv2.imread(src_path)
        if img is None:
            print(f"  [SKIP] Cannot read {fn}")
            continue
        if not has_face(img):
            print(f"  [SKIP] No face in {fn}")
            continue
        copied += 1
        dst_path = os.path.join(target, f"{name}_{copied:02d}.jpg")
        shutil.copy2(src_path, dst_path)
        print(f"  [COPIED] {fn} -> {os.path.basename(dst_path)}")

    return copied


# ---------- main ----------
def main():
    print("=" * 55)
    print("  Face Attendance - Register a New Person")
    print("=" * 55)

    while True:
        print("\nChoose an option:")
        print("  1) Register via webcam")
        print("  2) Register by importing existing photos")
        print("  3) Rebuild encodings from dataset/ (no new photos)")
        print("  4) List registered people")
        print("  5) Quit")
        choice = input("Enter 1-5: ").strip()

        if choice == "1":
            name = ask_name()
            n = register_webcam(name)
            if n:
                print(f"\n[INFO] Captured {n} photo(s) for '{name}'.")
                rebuild_encodings()

        elif choice == "2":
            name = ask_name()
            n = register_from_folder(name)
            if n:
                print(f"\n[INFO] Imported {n} photo(s) for '{name}'.")
                rebuild_encodings()

        elif choice == "3":
            print("\n[INFO] Rebuilding encodings...")
            rebuild_encodings()

        elif choice == "4":
            if not os.path.isdir(DATASET_DIR):
                print("  (no dataset folder yet)")
                continue
            people = [p for p in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, p))]
            if not people:
                print("  (no registered people)")
            else:
                print("\n  Registered people:")
                for p in people:
                    count = len([f for f in os.listdir(os.path.join(DATASET_DIR, p))
                                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                    print(f"    - {p}  ({count} photos)")

        elif choice == "5":
            print("Bye.")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()




















Step 1: Create a Virtual Environment
Open PowerShell in your project folder and run:

powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
If you get an execution policy error, run:

powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Step 2: Install Dependencies
Create a file named requirements.txt:

text
insightface
opencv-python
onnxruntime
numpy
pandas
streamlit
Then install:

powershell
pip install -r requirements.txt
InsightFace will automatically download its models on first use.

Step 3: Project Structure
text
AI_ATTENDANCE_MANAGER/
├── venv/                     # Virtual environment
├── dataset/                  # Folder for student reference photos
│   ├── John_Doe/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Jane_Smith/
│       └── photo1.jpg
├── attendance/               # Auto-created attendance logs
├── encodings.pkl             # Auto-generated face encodings
├── register.py               # Generates encodings from dataset
├── attendance.py             # Real-time attendance via webcam
├── dashboard.py              # Streamlit dashboard for viewing records
└── requirements.txt






mkdir AI_ATTENDANCE_MANAGER
cd AI_ATTENDANCE_MANAGER

# Create virtual environment
python -m venv venv

# Create folders
mkdir dataset
mkdir attendance

# Create sample student folders
mkdir dataset\John_Doe
mkdir dataset\Jane_Smith

# Create Python files
New-Item -ItemType File -Path register.py
New-Item -ItemType File -Path attendance.py
New-Item -ItemType File -Path dashboard.py
New-Item -ItemType File -Path requirements.txt

# Create placeholder for encodings
New-Item -ItemType File -Path encodings.pkl

# Open project in VS Code
code .





write all file code





Step 7: Run the System
Add student photos to dataset/Person_Name/.

Generate encodings:

powershell
python register.py
Start real-time attendance:

powershell
python attendance.py
View dashboard (optional):

powershell
streamlit run dashboard.py