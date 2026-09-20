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