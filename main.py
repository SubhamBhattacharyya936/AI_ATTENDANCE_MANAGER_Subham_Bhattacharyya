"""
main.py — Student Attendance System (Tkinter GUI)

Single entry point that wraps every operation of the AI_ATTENDANCE_MANAGER project:

    1. Take Attendance      -> webcam + InsightFace recognition, logs to attendance/YYYY-MM-DD.csv
    2. Show Attendance      -> table of who was present, date + time in IST (12-hour format)
    3. Add New Student      -> register via webcam / import photos / rebuild encodings / list people
    4. Exit

Run:  python main.py
"""

import os
import pickle
import shutil
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

try:
    from PIL import Image, ImageTk
except ImportError:
    raise SystemExit("Pillow is required.  Install it with:  pip install pillow")

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise SystemExit("insightface is required.  Install it with:  pip install insightface onnxruntime")


# ──────────────────────────────────────────────────────────────
#  Configuration / paths
# ──────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR    = os.path.join(BASE_DIR, "dataset")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pkl")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "attendance")

THRESHOLD     = 0.45      # cosine similarity threshold
NUM_PHOTOS    = 5         # photos to capture per person in webcam mode
MIN_FACE_SIZE = 60        # px — ignore smaller detections

IST = timezone(timedelta(hours=5, minutes=30))   # Indian Standard Time

for _d in (DATASET_DIR, ATTENDANCE_DIR):
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
#  Theme
# ──────────────────────────────────────────────────────────────
BG      = "#0f172a"
CARD    = "#1e293b"
CARD2   = "#0b1220"
TEXT    = "#e2e8f0"
MUTED   = "#94a3b8"
BLUE    = "#3b82f6"
GREEN   = "#22c55e"
AMBER   = "#f59e0b"
RED     = "#ef4444"
PURPLE  = "#8b5cf6"
TEAL    = "#14b8a6"


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────
def now_ist():
    return datetime.now(IST)


def today_str():
    return now_ist().strftime("%Y-%m-%d")


def sanitize_name(name: str) -> str:
    name = (name or "").strip()
    safe = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)
    return safe or "Unknown"


def to_12hr(value) -> str:
    """Convert any stored time string to 12-hour format (IST display style)."""
    s = str(value).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"):
        try:
            return datetime.strptime(s, fmt).strftime("%I:%M:%S %p")
        except ValueError:
            continue
    return s


def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)   # Windows fallback
    return cap if cap.isOpened() else None


# ──────────────────────────────────────────────────────────────
#  Attendance storage
# ──────────────────────────────────────────────────────────────
COLS = ["Name", "Date", "Time"]


def attendance_file(date=None):
    return os.path.join(ATTENDANCE_DIR, f"{date or today_str()}.csv")


def load_attendance(date=None) -> pd.DataFrame:
    """Return today's (or a given date's) attendance as a DataFrame."""
    path = attendance_file(date)
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLS)
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame(columns=COLS)
    if df.empty:
        return pd.DataFrame(columns=COLS)
    for c in COLS:
        if c not in df.columns:
            df[c] = ""
    return df[COLS]


def mark_attendance(name: str) -> bool:
    """Append a row for `name` if not already present today. Returns True if newly marked."""
    df = load_attendance()
    if name in set(df["Name"].tolist()):
        return False

    path = attendance_file()
    new_file = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        if new_file:
            f.write("Name,Date,Time\n")
        f.write(f"{name},{today_str()},{now_ist().strftime('%I:%M:%S %p')}\n")
    return True


def load_encodings() -> dict:
    if not os.path.exists(ENCODINGS_FILE):
        return {}
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def rebuild_encodings(model) -> dict:
    """Recompute encodings.pkl from the whole dataset/ folder."""
    encodings = {}
    if os.path.isdir(DATASET_DIR):
        for person in sorted(os.listdir(DATASET_DIR)):
            pdir = os.path.join(DATASET_DIR, person)
            if not os.path.isdir(pdir):
                continue
            embs = []
            for fn in sorted(os.listdir(pdir)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(os.path.join(pdir, fn))
                    if img is None:
                        continue
                    faces = model.get(img)
                    if faces:
                        embs.append(faces[0].embedding)
            if embs:
                encodings[person] = np.mean(embs, axis=0)
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings, f)
    return encodings


# ══════════════════════════════════════════════════════════════
#  Main application
# ══════════════════════════════════════════════════════════════
class AttendanceApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Student Attendance System")
        self.configure(bg=BG)
        self.minsize(940, 660)
        self._center(1060, 720)

        # model / camera state
        self.model        = None
        self.cap          = None
        self.video_job    = None
        self.last_faces   = []
        self.frame_idx    = 0
        self.mode         = None          # "attendance" | "capture"
        self.known        = {}
        self.marked_session = set()

        # capture-mode state
        self.capture_name = ""
        self.capture_dir  = ""
        self.saved_count  = 0
        self.current_frame = None
        self.current_face_ok = False
        self._space_bind = None
        self._esc_bind   = None

        self.container = tk.Frame(self, bg=BG)
        self.container.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.show_home()

    # ────────────────────────────── window utils
    def _center(self, w, h):
        self.update_idletasks()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _center_toplevel(self, win):
        win.update_idletasks()
        w, h = win.winfo_width(), win.winfo_height()
        x = (win.winfo_screenwidth() - w) // 2
        y = (win.winfo_screenheight() - h) // 2
        win.geometry(f"+{x}+{y}")

    def clear(self):
        self.stop_camera()
        for w in self.container.winfo_children():
            w.destroy()

    def on_close(self):
        self.stop_camera()
        self.destroy()

    # ────────────────────────────── buttons
    def _big_button(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Segoe UI", 16, "bold"),
                         bg=color, fg="white",
                         activebackground=color, activeforeground="white",
                         relief="flat", bd=0, width=26, height=2,
                         cursor="hand2", highlightthickness=0)

    def _small_button(self, parent, text, cmd, color=BLUE):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Segoe UI", 10, "bold"),
                         bg=color, fg="white",
                         activebackground=color, activeforeground="white",
                         relief="flat", bd=0, padx=14, pady=6,
                         cursor="hand2", highlightthickness=0)

    # ══════════════════════════════════════════════════════════
    #  HOME
    # ══════════════════════════════════════════════════════════
    def show_home(self):
        self.clear()

        head = tk.Frame(self.container, bg=BG)
        head.pack(fill="x", pady=(45, 0))
        tk.Label(head, text="🎓", font=("Segoe UI Emoji", 46), bg=BG, fg=TEXT).pack()
        tk.Label(head, text="Student Attendance System",
                 font=("Segoe UI", 30, "bold"), bg=BG, fg=TEXT).pack(pady=(6, 0))

        self.clock_var = tk.StringVar()
        tk.Label(head, textvariable=self.clock_var,
                 font=("Segoe UI", 12), bg=BG, fg=MUTED).pack(pady=(8, 0))
        self._tick_clock()

        body = tk.Frame(self.container, bg=BG)
        body.pack(expand=True)

        self._big_button(body, "📷   Take Attendance",  self.take_attendance,  GREEN).pack(pady=9)
        self._big_button(body, "📋   Show Attendance",  self.show_attendance,  BLUE).pack(pady=9)
        self._big_button(body, "➕   Add New Student",  self.add_new_student,  PURPLE).pack(pady=9)
        self._big_button(body, "🚪   Exit",             self.on_close,         RED).pack(pady=9)

        n_students = len([p for p in os.listdir(DATASET_DIR)
                          if os.path.isdir(os.path.join(DATASET_DIR, p))]) if os.path.isdir(DATASET_DIR) else 0
        df_today = load_attendance()
        tk.Label(self.container,
                 text=f"Registered students: {n_students}     •     Present today: {len(df_today)}",
                 font=("Segoe UI", 11), bg=BG, fg=MUTED).pack(pady=(18, 22))

    def _tick_clock(self):
        if not self.winfo_exists():
            return
        try:
            self.clock_var.set(now_ist().strftime("%A, %d %B %Y   •   %I:%M:%S %p IST"))
        except Exception:
            return
        self.after(1000, self._tick_clock)

    # ══════════════════════════════════════════════════════════
    #  MODEL LOADING
    # ══════════════════════════════════════════════════════════
    def ensure_model(self) -> bool:
        if self.model is not None:
            return True

        dlg = tk.Toplevel(self)
        dlg.title("Please wait")
        dlg.configure(bg=CARD)
        dlg.resizable(False, False)
        dlg.transient(self)
        tk.Label(dlg, text="Loading face recognition model…\nThis may take a few seconds.",
                 font=("Segoe UI", 12), bg=CARD, fg=TEXT, padx=36, pady=28).pack()
        self._center_toplevel(dlg)
        dlg.grab_set()
        self.update_idletasks()

        try:
            model = FaceAnalysis(name="buffalo_l")
            model.prepare(ctx_id=0, det_size=(640, 640))
            self.model = model
        except Exception as e:
            dlg.grab_release()
            dlg.destroy()
            messagebox.showerror("Model Error", f"Could not load face model:\n\n{e}")
            return False

        dlg.grab_release()
        dlg.destroy()
        return True

    # ══════════════════════════════════════════════════════════
    #  CAMERA CONTROL
    # ══════════════════════════════════════════════════════════
    def stop_camera(self):
        if self.video_job:
            try:
                self.after_cancel(self.video_job)
            except Exception:
                pass
            self.video_job = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self._space_bind:
            try:
                self.unbind("<space>", self._space_bind)
            except Exception:
                pass
            self._space_bind = None
        if self._esc_bind:
            try:
                self.unbind("<Escape>", self._esc_bind)
            except Exception:
                pass
            self._esc_bind = None
        self.mode = None
        self.last_faces = []
        self.frame_idx = 0

    # ══════════════════════════════════════════════════════════
    #  1) TAKE ATTENDANCE
    # ══════════════════════════════════════════════════════════
    def take_attendance(self):
        if not self.ensure_model():
            return

        self.known = load_encodings()
        if not self.known:
            messagebox.showwarning(
                "No registered students",
                "encodings.pkl is empty.\n\n"
                "Go to  Add New Student  and register at least one person first."
            )
            return

        self.clear()

        head = tk.Frame(self.container, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 6))
        self._small_button(head, "← Back", self.show_home, CARD).pack(side="left")
        tk.Label(head, text="Take Attendance", font=("Segoe UI", 18, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", padx=16)
        tk.Label(head, text=f"{len(self.known)} registered", font=("Segoe UI", 10),
                 bg=BG, fg=MUTED).pack(side="right")

        body = tk.Frame(self.container, bg=BG)
        body.pack(expand=True, fill="both", padx=16, pady=(0, 14))

        self.video_label = tk.Label(body, bg="#000000", bd=0)
        self.video_label.pack(side="left")

        side = tk.Frame(body, bg=CARD, width=250)
        side.pack(side="left", fill="y", padx=(16, 0))
        side.pack_propagate(False)

        tk.Label(side, text="✅  Marked Today", font=("Segoe UI", 12, "bold"),
                 bg=CARD, fg=TEXT).pack(pady=(14, 8))

        self.marked_box = tk.Listbox(side, bg=CARD2, fg="#4ade80",
                                     font=("Consolas", 10), bd=0,
                                     highlightthickness=0, selectbackground=CARD)
        self.marked_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.status_var = tk.StringVar(value="Starting camera…")
        tk.Label(self.container, textvariable=self.status_var,
                 font=("Segoe UI", 11), bg=BG, fg=AMBER).pack(pady=(0, 10))

        # open camera
        self.cap = open_camera(0)
        if self.cap is None:
            messagebox.showerror("Camera Error",
                                 "Cannot open the webcam.\nClose any app that is using it and try again.")
            self.show_home()
            return

        self.mode = "attendance"
        self.marked_session = set(r for r in load_attendance()["Name"].tolist())
        self.refresh_marked_list()
        self.status_var.set("Camera running — press the Back button to stop.")

        self.video_job = self.after(50, self._video_loop)

    def refresh_marked_list(self):
        if not hasattr(self, "marked_box") or not self.marked_box.winfo_exists():
            return
        self.marked_box.delete(0, tk.END)
        df = load_attendance()
        for _, row in df.iterrows():
            self.marked_box.insert(tk.END, f"{row['Name']:<16} {to_12hr(row['Time'])}")

    # ══════════════════════════════════════════════════════════
    #  SHARED VIDEO LOOP
    # ══════════════════════════════════════════════════════════
    def _video_loop(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.video_job = self.after(40, self._video_loop)
            return

        frame = cv2.flip(frame, 1)
        self.current_frame = frame.copy()
        self.frame_idx += 1

        if self.frame_idx % 3 == 0:                      # detect every 3rd frame (speed)
            try:
                self.last_faces = self.model.get(frame)
            except Exception:
                self.last_faces = []

        self.current_face_ok = False

        for face in self.last_faces:
            box = face.bbox.astype(int)
            bw, bh = box[2] - box[0], box[3] - box[1]

            if self.mode == "capture":
                if bw >= MIN_FACE_SIZE and bh >= MIN_FACE_SIZE:
                    self.current_face_ok = True
                    color = (0, 220, 0)
                else:
                    color = (0, 165, 255)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            else:  # attendance mode — recognise
                emb = face.embedding
                best_name, best_sim = "Unknown", -1.0
                for person, known in self.known.items():
                    denom = float(np.linalg.norm(emb) * np.linalg.norm(known))
                    if denom == 0:
                        continue
                    sim = float(np.dot(emb, known) / denom)
                    if sim > best_sim:
                        best_sim, best_name = sim, person

                matched = best_sim > THRESHOLD
                color = (0, 220, 0) if matched else (0, 0, 255)
                label = f"{best_name}  {best_sim:.2f}" if matched else "Unknown"

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], max(box[1] - 10, 22)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

                if matched and best_name not in self.marked_session:
                    self.marked_session.add(best_name)
                    if mark_attendance(best_name):
                        self.refresh_marked_list()
                        self.status_var.set(
                            f"✅  {best_name} marked present at "
                            f"{now_ist().strftime('%I:%M:%S %p')} IST"
                        )
                        try:
                            self.bell()
                        except Exception:
                            pass

        # capture-mode overlay
        if self.mode == "capture":
            cv2.putText(frame, f"Captured {self.saved_count}/{NUM_PHOTOS}", (12, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "SPACE = capture      ESC = finish", (12, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            if not self.current_face_ok:
                cv2.putText(frame, "No face / face too small", (12, 94),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # render into the Tk label
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        w0, h0 = img.size
        tw = 700
        th = int(h0 * tw / w0)
        img = img.resize((tw, th), Image.BILINEAR)
        imgtk = ImageTk.PhotoImage(img)
        self.video_label.configure(image=imgtk)
        self.video_label.image = imgtk

        self.video_job = self.after(10, self._video_loop)

    # ══════════════════════════════════════════════════════════
    #  2) SHOW ATTENDANCE
    # ══════════════════════════════════════════════════════════
    def show_attendance(self):
        self.clear()

        head = tk.Frame(self.container, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 6))
        self._small_button(head, "← Back", self.show_home, CARD).pack(side="left")
        tk.Label(head, text="Attendance Records", font=("Segoe UI", 18, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", padx=16)
        self._small_button(head, "🔄  Refresh", self.show_attendance, TEAL).pack(side="right")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview",
                        background=CARD, fieldbackground=CARD, foreground=TEXT,
                        rowheight=28, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading",
                        background="#334155", foreground=TEXT,
                        font=("Segoe UI", 10, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", BLUE)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=TEXT,
                        padding=(18, 8), font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", BLUE)])

        nb = ttk.Notebook(self.container)
        nb.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        # ---------- Today tab ----------
        today_tab = tk.Frame(nb, bg=BG)
        nb.add(today_tab, text=f"  Today ({today_str()})  ")

        df_today = load_attendance().copy()
        if not df_today.empty:
            df_today["Time"] = df_today["Time"].apply(to_12hr)
            df_today.insert(0, "S.No", range(1, len(df_today) + 1))

        tk.Label(today_tab,
                 text=f"Total present today: {len(df_today)}",
                 font=("Segoe UI", 12, "bold"), bg=BG, fg=GREEN).pack(anchor="w", padx=14, pady=(12, 0))
        self._build_table(today_tab, df_today)

        # ---------- All records tab ----------
        all_tab = tk.Frame(nb, bg=BG)
        nb.add(all_tab, text="  All Records  ")

        frames = []
        for fn in sorted(os.listdir(ATTENDANCE_DIR)):
            if fn.lower().endswith(".csv"):
                d = load_attendance(fn[:-4])
                if not d.empty:
                    frames.append(d)

        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined["Time"] = combined["Time"].apply(to_12hr)
            combined = combined.sort_values(["Date", "Name"], ascending=[False, True])
            combined.insert(0, "S.No", range(1, len(combined) + 1))
        else:
            combined = pd.DataFrame(columns=["S.No"] + COLS)

        tk.Label(all_tab, text=f"Total records: {len(combined)}",
                 font=("Segoe UI", 12, "bold"), bg=BG, fg=BLUE).pack(anchor="w", padx=14, pady=(12, 0))
        self._build_table(all_tab, combined)

    def _build_table(self, parent, df: pd.DataFrame):
        wrap = tk.Frame(parent, bg=BG)
        wrap.pack(fill="both", expand=True, padx=14, pady=12)

        cols = list(df.columns)
        tree = ttk.Treeview(wrap, columns=cols, show="headings")

        for c in cols:
            tree.heading(c, text=c)
            if c == "S.No":
                tree.column(c, width=60, anchor="center", stretch=False)
            elif c == "Name":
                tree.column(c, width=260, anchor="w")
            else:
                tree.column(c, width=180, anchor="center")

        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        tree.tag_configure("odd", background="#172033")
        tree.tag_configure("even", background=CARD)

        for i, (_, row) in enumerate(df.iterrows()):
            tree.insert("", "end", values=list(row),
                        tags=("odd" if i % 2 else "even",))

        if df.empty:
            tree.insert("", "end", values=["—"] * len(cols))

    # ══════════════════════════════════════════════════════════
    #  3) ADD NEW STUDENT
    # ══════════════════════════════════════════════════════════
    def add_new_student(self):
        self.clear()

        head = tk.Frame(self.container, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 6))
        self._small_button(head, "← Back", self.show_home, CARD).pack(side="left")
        tk.Label(head, text="Add New Student", font=("Segoe UI", 18, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", padx=16)

        body = tk.Frame(self.container, bg=BG)
        body.pack(expand=True)

        tk.Label(body, text="Choose an option", font=("Segoe UI", 13),
                 bg=BG, fg=MUTED).pack(pady=(0, 22))

        self._big_button(body, "📷   Register via Webcam",
                         self.register_webcam, GREEN).pack(pady=9)
        self._big_button(body, "📁   Register by Importing Existing Photos",
                         self.register_from_folder, BLUE).pack(pady=9)
        self._big_button(body, "🔄   Rebuild Encodings from dataset/",
                         self.rebuild_encodings_ui, AMBER).pack(pady=9)
        self._big_button(body, "📋   List Registered People",
                         self.list_people, PURPLE).pack(pady=9)

    # ── 3a) webcam registration ─────────────────────────────
    def register_webcam(self):
        if not self.ensure_model():
            return

        name = simpledialog.askstring("New Student",
                                      "Enter the student's name\n(e.g. John_Doe):",
                                      parent=self)
        if not name or not name.strip():
            return
        name = sanitize_name(name)

        target = os.path.join(DATASET_DIR, name)
        os.makedirs(target, exist_ok=True)

        self.clear()
        self.capture_name = name
        self.capture_dir = target
        self.saved_count = 0

        head = tk.Frame(self.container, bg=BG)
        head.pack(fill="x", padx=16, pady=(14, 6))
        self._small_button(head, "← Cancel", self.add_new_student, CARD).pack(side="left")
        tk.Label(head, text=f"Registering:  {name}", font=("Segoe UI", 18, "bold"),
                 bg=BG, fg=TEXT).pack(side="left", padx=16)

        self.video_label = tk.Label(self.container, bg="#000000", bd=0)
        self.video_label.pack(pady=(0, 10))

        self.status_var = tk.StringVar(
            value=f"Look at the camera • SPACE = capture • ESC = finish   (0/{NUM_PHOTOS})"
        )
        tk.Label(self.container, textvariable=self.status_var,
                 font=("Segoe UI", 11), bg=BG, fg=AMBER).pack(pady=(0, 14))

        self.cap = open_camera(0)
        if self.cap is None:
            messagebox.showerror("Camera Error", "Cannot open the webcam.")
            self.add_new_student()
            return

        self.mode = "capture"
        self._space_bind = self.bind("<space>", lambda e: self.capture_photo())
        self._esc_bind = self.bind("<Escape>", lambda e: self.finish_capture())
        self.focus_force()
        self.video_job = self.after(50, self._video_loop)

    def capture_photo(self):
        if self.mode != "capture" or self.cap is None:
            return
        if not self.current_face_ok:
            self.status_var.set("⚠  No suitable face detected — move closer and try again")
            return

        self.saved_count += 1
        path = os.path.join(self.capture_dir, f"{self.capture_name}_{self.saved_count:02d}.jpg")
        cv2.imwrite(path, self.current_frame)
        self.status_var.set(
            f"✅  Saved {self.saved_count}/{NUM_PHOTOS}   •   "
            f"SPACE = capture   ESC = finish"
        )

        if self.saved_count >= NUM_PHOTOS:
            self.after(250, self.finish_capture)

    def finish_capture(self):
        if self.mode != "capture":
            return
        name = self.capture_name
        saved = self.saved_count
        self.stop_camera()

        if saved == 0:
            messagebox.showwarning("No Photos", "No photos were captured — nothing was saved.")
            self.add_new_student()
            return

        self._rebuild_with_progress()
        messagebox.showinfo(
            "Registration Complete",
            f"'{name}' registered with {saved} photo(s).\n\n"
            f"encodings.pkl has been rebuilt — the student can now be recognised."
        )
        self.add_new_student()

    # ── 3b) import photos ───────────────────────────────────
    def register_from_folder(self):
        if not self.ensure_model():
            return

        name = simpledialog.askstring("New Student",
                                      "Enter the student's name\n(e.g. Jane_Smith):",
                                      parent=self)
        if not name or not name.strip():
            return
        name = sanitize_name(name)

        folder = filedialog.askdirectory(title="Select the folder containing the photos")
        if not folder:
            return

        target = os.path.join(DATASET_DIR, name)
        os.makedirs(target, exist_ok=True)

        copied = skipped = 0
        for fn in sorted(os.listdir(folder)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src = os.path.join(folder, fn)
            img = cv2.imread(src)
            if img is None:
                skipped += 1
                continue

            faces = self.model.get(img)
            good = any((f.bbox[2] - f.bbox[0]) >= MIN_FACE_SIZE and
                       (f.bbox[3] - f.bbox[1]) >= MIN_FACE_SIZE for f in faces)
            if not good:
                skipped += 1
                continue

            copied += 1
            dst = os.path.join(target, f"{name}_{copied:02d}.jpg")
            shutil.copy2(src, dst)

        if copied == 0:
            messagebox.showwarning(
                "Nothing Imported",
                f"No usable photos found.\n\nSkipped: {skipped}\n\n"
                "Make sure each photo contains a clear, forward-facing face."
            )
            return

        self._rebuild_with_progress()
        messagebox.showinfo(
            "Import Complete",
            f"Imported {copied} photo(s) for '{name}'.\n"
            f"Skipped {skipped} file(s).\n\nencodings.pkl has been rebuilt."
        )
        self.add_new_student()

    # ── 3c) rebuild encodings ───────────────────────────────
    def rebuild_encodings_ui(self):
        if not self.ensure_model():
            return
        n = self._rebuild_with_progress()
        messagebox.showinfo("Done", f"encodings.pkl rebuilt successfully.\n\nStudents encoded: {n}")

    def _rebuild_with_progress(self):
        dlg = tk.Toplevel(self)
        dlg.title("Working…")
        dlg.configure(bg=CARD)
        dlg.resizable(False, False)
        dlg.transient(self)
        tk.Label(dlg, text="Rebuilding encodings from dataset/ …\nPlease wait.",
                 font=("Segoe UI", 12), bg=CARD, fg=TEXT, padx=36, pady=28).pack()
        self._center_toplevel(dlg)
        dlg.grab_set()
        self.update_idletasks()

        try:
            enc = rebuild_encodings(self.model)
        except Exception as e:
            dlg.grab_release()
            dlg.destroy()
            messagebox.showerror("Error", f"Failed to rebuild encodings:\n\n{e}")
            return 0

        dlg.grab_release()
        dlg.destroy()
        self.known = enc
        return len(enc)

    # ── 3d) list registered people ──────────────────────────
    def list_people(self):
        win = tk.Toplevel(self)
        win.title("Registered People")
        win.configure(bg=BG)
        win.geometry("520x460")
        win.transient(self)

        tk.Label(win, text="Registered Students", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=TEXT).pack(pady=(18, 4))
        tk.Label(win, text=f"Dataset folder:  {DATASET_DIR}",
                 font=("Segoe UI", 9), bg=BG, fg=MUTED).pack(pady=(0, 12))

        style = ttk.Style()
        style.configure("Ppl.Treeview", background=CARD, fieldbackground=CARD,
                        foreground=TEXT, rowheight=26, borderwidth=0)
        style.configure("Ppl.Treeview.Heading", background="#334155",
                        foreground=TEXT, font=("Segoe UI", 10, "bold"))

        wrap = tk.Frame(win, bg=BG)
        wrap.pack(fill="both", expand=True, padx=16, pady=(0, 16))

        cols = ("#", "Name", "Photos", "Encoded")
        tree = ttk.Treeview(wrap, columns=cols, show="headings", style="Ppl.Treeview")
        for c, w in zip(cols, (50, 240, 90, 90)):
            tree.heading(c, text=c)
            tree.column(c, width=w, anchor="center" if c != "Name" else "w")

        vsb = ttk.Scrollbar(wrap, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        encodings = load_encodings()
        people = sorted(p for p in os.listdir(DATASET_DIR)
                        if os.path.isdir(os.path.join(DATASET_DIR, p)))

        for i, p in enumerate(people, start=1):
            pdir = os.path.join(DATASET_DIR, p)
            n = len([f for f in os.listdir(pdir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            tree.insert("", "end", values=(i, p, n, "Yes" if p in encodings else "No"))

        if not people:
            tree.insert("", "end", values=("—", "No registered people", "—", "—"))

        tk.Button(win, text="Close", command=win.destroy,
                  font=("Segoe UI", 10, "bold"), bg=BLUE, fg="white",
                  relief="flat", bd=0, padx=22, pady=7,
                  cursor="hand2").pack(pady=(0, 16))

        self._center_toplevel(win)
        win.grab_set()


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = AttendanceApp()
    app.mainloop()