from __future__ import annotations
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import pyttsx3
import threading

from flask import Flask, render_template, request, redirect, session, flash, send_file

# ---------------- EYE ASPECT RATIO ----------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ---------------- AI MODELS ----------------
from insightface.app import FaceAnalysis
from ultralytics import YOLO

# ================= CONFIG =================
app = Flask(__name__)
app.secret_key = "smart_attendance_secret"
DB_PATH = "database.db"

SIMILARITY_THRESHOLD = 0.45
OCCLUDED_SIMILARITY_THRESHOLD = 0.38
OCCLUDED_MARGIN_THRESHOLD = 0.03
REGISTRATION_SAMPLES = 20
ATTENDANCE_FRAME_SKIP = 2
MASK_CHECK_INTERVAL = 6
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FACE_DET_SIZE = (960, 960)
ATTENDANCE_LOCK_PATH = "attendance_session.lock"
MIN_ATTENDANCE_FACE_RATIO = 0.018
LIVENESS_HISTORY = 8
LIVENESS_MIN_MOTION = 12.0
LIVENESS_MIN_YAW_RANGE = 7.0
LIVENESS_MIN_PITCH_RANGE = 5.0
LIVENESS_MIN_SHARPNESS = 28.0
IDENTITY_CONFIRM_FRAMES = 3
SCREEN_PROXY_BRIGHT_RATIO = 0.18
SCREEN_PROXY_RECT_AREA_RATIO = 1.45
# ================= VOICE =================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()
# ================= LOAD MODELS =================
print("Loading Face Model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(
    ctx_id=0 if cv2.cuda.getCudaEnabledDeviceCount()>0 else -1,
    det_size=FACE_DET_SIZE
)

print("Loading Mask Model...")
MASK_MODEL_PATH = "best.pt"
mask_model = YOLO(MASK_MODEL_PATH) if os.path.exists(MASK_MODEL_PATH) else None
if mask_model is None:
    print("Mask model not found at best.pt - using occlusion fallback matching")

# ================= DATABASE =================
def connect_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = connect_db()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS teachers(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS classes(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        teacher_id INTEGER,
        class_name TEXT,
        subject TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        class_id INTEGER,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS encodings(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        encoding BLOB
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS attendance(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        class_id INTEGER,
        lecture_no INTEGER,
        date TEXT,
        status TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS admins(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )""")

    conn.commit()
    conn.close()

create_tables()
# ================= AUTO LIGHTING =================
def auto_brightness(frame):
    # Balance glare and dark corners so faces across the room stay usable.
    frame_f = frame.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    # Compress very bright highlights before local contrast enhancement.
    highlight_mask = v > 220
    if np.any(highlight_mask):
        frame_f[highlight_mask] *= 0.82

    balanced = np.clip(frame_f * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # Pull average luminance toward the middle to help both bright and dim rooms.
    mean_l = float(np.mean(l))
    gamma = 1.0
    if mean_l > 175:
        gamma = 1.25
    elif mean_l < 105:
        gamma = 0.78

    l_norm = np.clip((l / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    merged = cv2.merge((l_norm, a, b))
    corrected = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Mild denoise plus sharpen helps distant faces.
    denoised = cv2.bilateralFilter(corrected, 5, 40, 40)
    sharpened = cv2.addWeighted(denoised, 1.15, cv2.GaussianBlur(denoised, (0, 0), 2.2), -0.15, 0)
    return sharpened

def normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def get_primary_face(image):
    faces = face_app.get(image)
    if not faces:
        return None
    return max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    )

def expand_bbox(frame, bbox, x_ratio=0.3, y_ratio=0.4):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    pad_x = int(bw * x_ratio)
    pad_y = int(bh * y_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return x1, y1, x2, y2

def extract_face_crop(frame, face):
    x1, y1, x2, y2 = expand_bbox(frame, face.bbox)
    crop = frame[y1:y2, x1:x2].copy()

    rel_kps = None
    if getattr(face, "kps", None) is not None:
        rel_kps = np.array(face.kps, dtype=np.float32).copy()
        rel_kps[:, 0] -= x1
        rel_kps[:, 1] -= y1

    return crop, rel_kps

def get_face_center(face):
    x1, y1, x2, y2 = face.bbox.astype(int)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

def get_face_pose(face):
    pose = getattr(face, "pose", None)
    if pose is None:
        return 0.0, 0.0
    pose = np.asarray(pose, dtype=np.float32).flatten()
    yaw = float(pose[0]) if pose.size > 0 else 0.0
    pitch = float(pose[1]) if pose.size > 1 else 0.0
    return yaw, pitch

def get_face_sharpness(frame, face):
    crop, _ = extract_face_crop(frame, face)
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def get_face_ratio(frame, face):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face.bbox.astype(int)
    face_area = max(1, x2 - x1) * max(1, y2 - y1)
    frame_area = max(1, h * w)
    return face_area / frame_area

def is_screen_proxy(frame, face):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(frame, face.bbox, x_ratio=0.9, y_ratio=0.9)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return False

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    bright_ratio = float(np.mean(gray > 210))

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    face_x1, face_y1, face_x2, face_y2 = face.bbox.astype(int)
    face_area = max(1, face_x2 - face_x1) * max(1, face_y2 - face_y1)
    face_center_x = (face_x1 + face_x2) / 2.0
    face_center_y = (face_y1 + face_y2) / 2.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < face_area * SCREEN_PROXY_RECT_AREA_RATIO:
            continue

        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) != 4:
            continue

        rx, ry, rw, rh = cv2.boundingRect(approx)
        if rw <= 0 or rh <= 0:
            continue

        aspect_ratio = rw / float(rh)
        if not (0.45 <= aspect_ratio <= 2.2):
            continue

        rect_x1 = x1 + rx
        rect_y1 = y1 + ry
        rect_x2 = rect_x1 + rw
        rect_y2 = rect_y1 + rh

        inside_rect = (
            rect_x1 <= face_center_x <= rect_x2 and
            rect_y1 <= face_center_y <= rect_y2
        )

        if inside_rect and bright_ratio >= SCREEN_PROXY_BRIGHT_RATIO:
            return True

    return False

def update_liveness_state(state_map, track_key, frame, face):
    entry = state_map.setdefault(track_key, {
        "centers": [],
        "yaws": [],
        "pitches": [],
        "sharpness": []
    })

    entry["centers"].append(get_face_center(face))
    yaw, pitch = get_face_pose(face)
    entry["yaws"].append(yaw)
    entry["pitches"].append(pitch)
    entry["sharpness"].append(get_face_sharpness(frame, face))

    for key in ("centers", "yaws", "pitches", "sharpness"):
        entry[key] = entry[key][-LIVENESS_HISTORY:]

    if len(entry["centers"]) < LIVENESS_HISTORY:
        return False

    motion = float(np.linalg.norm(entry["centers"][-1] - entry["centers"][0]))
    yaw_range = max(entry["yaws"]) - min(entry["yaws"])
    pitch_range = max(entry["pitches"]) - min(entry["pitches"])
    sharpness = float(np.mean(entry["sharpness"]))

    return (
        sharpness >= LIVENESS_MIN_SHARPNESS and (
            motion >= LIVENESS_MIN_MOTION or
            yaw_range >= LIVENESS_MIN_YAW_RANGE or
            pitch_range >= LIVENESS_MIN_PITCH_RANGE
        )
    )

def get_best_match_scores(known_embeddings, emb):
    similarities = np.dot(known_embeddings, emb)
    if similarities.size == 0:
        return -1.0, -1.0

    top_k = min(3, similarities.size)
    top_scores = np.sort(similarities)[-top_k:]
    mean_score = float(np.mean(top_scores))
    max_score = float(np.max(similarities))
    return mean_score, max_score

def add_synthetic_mask(face_crop, kps=None):
    masked = face_crop.copy()
    h, w = masked.shape[:2]
    if h == 0 or w == 0:
        return masked

    overlay = masked.copy()
    if kps is not None and len(kps) >= 5:
        left_eye, right_eye, nose, _, _ = kps
        nose_y = int(nose[1])
        top_y = max(int(min(left_eye[1], right_eye[1]) + 0.18 * h), nose_y - int(0.05 * h))
        polygon = np.array([
            [int(0.14 * w), top_y],
            [int(0.86 * w), top_y],
            [int(0.82 * w), int(0.92 * h)],
            [int(0.18 * w), int(0.92 * h)]
        ], dtype=np.int32)
    else:
        polygon = np.array([
            [int(0.12 * w), int(0.48 * h)],
            [int(0.88 * w), int(0.48 * h)],
            [int(0.82 * w), int(0.94 * h)],
            [int(0.18 * w), int(0.94 * h)]
        ], dtype=np.int32)

    cv2.fillConvexPoly(overlay, polygon, (220, 220, 220))
    cv2.polylines(overlay, [polygon], True, (120, 120, 120), 2)
    return cv2.addWeighted(overlay, 0.85, masked, 0.15, 0)

def add_synthetic_glasses(face_crop, kps=None):
    glasses = face_crop.copy()
    h, w = glasses.shape[:2]
    if h == 0 or w == 0:
        return glasses

    overlay = glasses.copy()
    if kps is not None and len(kps) >= 2:
        left_eye, right_eye = kps[0], kps[1]
        eye_span = max(18, int(abs(right_eye[0] - left_eye[0]) * 0.55))
        lens_w = eye_span
        lens_h = max(16, int(eye_span * 0.65))

        left_center = (int(left_eye[0]), int(left_eye[1]))
        right_center = (int(right_eye[0]), int(right_eye[1]))
    else:
        lens_w = int(w * 0.2)
        lens_h = int(h * 0.14)
        left_center = (int(w * 0.36), int(h * 0.38))
        right_center = (int(w * 0.64), int(h * 0.38))

    for center in (left_center, right_center):
        x1 = max(0, center[0] - lens_w // 2)
        y1 = max(0, center[1] - lens_h // 2)
        x2 = min(w, center[0] + lens_w // 2)
        y2 = min(h, center[1] + lens_h // 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (35, 35, 35), 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (70, 70, 70), -1)

    cv2.line(overlay, left_center, right_center, (35, 35, 35), 2)
    cv2.line(overlay, (0, left_center[1]), (left_center[0] - lens_w // 2, left_center[1]), (35, 35, 35), 2)
    cv2.line(overlay, (right_center[0] + lens_w // 2, right_center[1]), (w - 1, right_center[1]), (35, 35, 35), 2)
    return cv2.addWeighted(overlay, 0.55, glasses, 0.45, 0)

def generate_registration_embeddings(frame, face):
    embeddings = [normalize_embedding(face.embedding)]
    face_crop, rel_kps = extract_face_crop(frame, face)

    if face_crop.size == 0:
        return embeddings

    for augmenter in (add_synthetic_mask, add_synthetic_glasses):
        augmented_face = get_primary_face(augmenter(face_crop, rel_kps))
        if augmented_face is not None:
            embeddings.append(normalize_embedding(augmented_face.embedding))

    return embeddings

def detect_mask_status(frame, bbox):
    if mask_model is None:
        return False, "MaskCheckOff"

    x1, y1, x2, y2 = expand_bbox(frame, bbox, x_ratio=0.1, y_ratio=0.1)
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False, "MaskUnknown"

    try:
        results = mask_model.predict(face_crop, conf=0.35, verbose=False)
    except Exception:
        return False, "MaskUnknown"

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return False, "MaskUnknown"

    boxes = results[0].boxes
    best_index = int(np.argmax(boxes.conf.cpu().numpy()))
    class_id = int(boxes.cls[best_index].item())
    label = str(results[0].names.get(class_id, "NoMask"))

    lowered = label.lower()
    is_masked = "with_mask" in lowered or "incorrect" in lowered
    return is_masked, label

def get_class_lecture_count(cur, class_id):
    cur.execute("""
        SELECT COUNT(DISTINCT lecture_no)
        FROM attendance
        WHERE class_id = ?
    """, (class_id,))
    return cur.fetchone()[0] or 0

def write_attendance_lock(pid, class_id, lecture):
    with open(ATTENDANCE_LOCK_PATH, "w", encoding="utf-8") as f:
        f.write(f"{pid}|{class_id}|{lecture}")

def clear_attendance_lock(pid=None):
    if not os.path.exists(ATTENDANCE_LOCK_PATH):
        return

    if pid is None:
        try:
            os.remove(ATTENDANCE_LOCK_PATH)
        except OSError:
            pass
        return

    try:
        with open(ATTENDANCE_LOCK_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return

    lock_pid = content.split("|", 1)[0].strip() if content else ""
    if lock_pid == str(pid):
        try:
            os.remove(ATTENDANCE_LOCK_PATH)
        except OSError:
            pass

def is_process_running(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False

    if pid <= 0:
        return False

    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True
        )
        return str(pid) in result.stdout

    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def is_attendance_session_running():
    if not os.path.exists(ATTENDANCE_LOCK_PATH):
        return False

    try:
        with open(ATTENDANCE_LOCK_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError:
        return False

    if not content:
        clear_attendance_lock()
        return False

    pid = content.split("|", 1)[0].strip()
    if is_process_running(pid):
        return True

    clear_attendance_lock()
    return False
# ================= LECTURE =================
def get_next_lecture_no(class_id):
    conn = connect_db()
    cur = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    cur.execute("""
        SELECT MAX(lecture_no)
        FROM attendance
        WHERE class_id=? AND date=?
    """,(class_id,today))

    result = cur.fetchone()[0]
    conn.close()
    return 1 if result is None else result + 1

# ================= AUTH =================
def require_login():
    return "teacher_id" in session

@app.route("/", methods=["GET","POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = connect_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM teachers WHERE email=? AND password=?",(email,password))
        user = cur.fetchone()
        conn.close()

        if user:
            session["teacher_id"] = user["id"]
            return redirect("/dashboard")
        else:
            flash("Invalid Email or Password ❌")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()

    cur.execute("SELECT * FROM classes WHERE teacher_id=?",(session["teacher_id"],))
    classes = cur.fetchall()

    cur.execute("""
        SELECT COUNT(*)
        FROM students
        JOIN classes ON students.class_id=classes.id
        WHERE classes.teacher_id=?
    """,(session["teacher_id"],))
    total_students = cur.fetchone()[0]

    today = datetime.now().strftime("%Y-%m-%d")
    cur.execute("""
        SELECT COUNT(*)
        FROM attendance
        JOIN classes ON attendance.class_id=classes.id
        WHERE classes.teacher_id=? AND attendance.date=?
    """,(session["teacher_id"],today))
    today_attendance = cur.fetchone()[0]

    conn.close()

    return render_template("dashboard.html",
                           classes=classes,
                           total_students=total_students,
                           today_attendance=today_attendance)

# ================= CLASS PANEL =================
@app.route("/class/<int:class_id>")
def class_panel(class_id):
    if not require_login():
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.*, COUNT(e.id) AS sample_count
        FROM students s
        LEFT JOIN encodings e ON s.id = e.student_id
        WHERE s.class_id=?
        GROUP BY s.id
        ORDER BY s.name
    """,(class_id,))
    students = cur.fetchall()
    conn.close()

    return render_template(
        "class.html",
        students=students,
        class_id=class_id,
        registration_samples=REGISTRATION_SAMPLES
    )

# ================= CREATE CLASS =================
@app.route("/create_class", methods=["POST"])
def create_class():
    if not require_login():
        return redirect("/")

    name = request.form["class_name"]
    subject = request.form["subject"]

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO classes (teacher_id,class_name,subject) VALUES (?,?,?)",
                (session["teacher_id"],name,subject))
    conn.commit()
    conn.close()
    return redirect("/dashboard")

# ================= ADD STUDENT (Threaded, no GUI) =================
@app.route("/add_student/<int:class_id>", methods=["POST"])
def add_student(class_id):
    import time

    if not require_login():
        return redirect("/")

    name = request.form["name"]

    # 1️⃣ Insert student into DB
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO students (class_id,name) VALUES (?,?)", (class_id, name))
    student_id = cur.lastrowid
    conn.commit()
    conn.close()

    def capture_faces(student_id, name):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        time.sleep(1)

        samples = []
        captured_frames = 0
        liveness_state = {}
        while captured_frames < REGISTRATION_SAMPLES:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = auto_brightness(frame)
            face = get_primary_face(frame)
            if face is None:
                continue

            if not update_liveness_state(liveness_state, "registration", frame, face):
                continue

            samples.extend(generate_registration_embeddings(frame, face))
            captured_frames += 1

            # No cv2.imshow → no GUI

        cap.release()

        # Save embeddings to DB
        conn = connect_db()
        cur = conn.cursor()
        for emb in samples:
            cur.execute("INSERT INTO encodings (student_id, encoding) VALUES (?,?)",
                        (student_id, emb.astype(np.float32).tobytes()))
        conn.commit()
        conn.close()
        print(f"Student '{name}' registered ✅")
        threading.Thread(target=speak, args=(f"{name} registered successfully",)).start()

    # Start face capture in a thread
    threading.Thread(target=capture_faces, args=(student_id, name)).start()

    flash(f"Student '{name}' added successfully ✅ (Face registration in background)", "success")
    return redirect(f"/class/{class_id}")

# ================= TAKE ATTENDANCE (MULTI FACE FIXED) =================
def run_attendance_session(class_id, lecture, today):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT s.id, s.name, e.encoding
        FROM students s
        JOIN encodings e ON s.id = e.student_id
        WHERE s.class_id = ?
    """, (class_id,))
    rows = cur.fetchall()
    conn.close()

    if not rows:
        print("No students registered")
        return

    student_map = {}
    for sid, name, enc in rows:
        emb = np.frombuffer(enc, dtype=np.float32)
        if sid not in student_map:
            student_map[sid] = {"name": name, "enc": []}
        student_map[sid]["enc"].append(emb)

    student_ids = list(student_map.keys())
    for data in student_map.values():
        data["enc"] = np.array(
            [normalize_embedding(emb) for emb in data["enc"]],
            dtype=np.float32
        )

    pid = os.getpid()
    write_attendance_lock(pid, class_id, lecture)

    cap = None
    marked_students = set()
    frame_index = 0
    session_started = False
    liveness_state = {}
    identity_streak = {}
    tracked_faces = []

    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        if not cap.isOpened():
            print("Camera not opening")
            return

        session_started = True
        cv2.namedWindow("AI Attendance System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AI Attendance System", CAMERA_WIDTH, CAMERA_HEIGHT)

        print("Attendance Started")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = auto_brightness(frame)
            frame_index += 1

            if frame_index % ATTENDANCE_FRAME_SKIP == 0 or not tracked_faces:
                tracked_faces = []
                faces = face_app.get(frame)

                for face in faces:
                    emb = normalize_embedding(face.embedding)
                    mask_detected = False
                    screen_proxy = is_screen_proxy(frame, face)

                    if frame_index % MASK_CHECK_INTERVAL == 0:
                        mask_detected, _ = detect_mask_status(frame, face.bbox)

                    name = "Unknown"
                    sid = None
                    best_score = -1.0
                    best_occluded_score = -1.0
                    second_best_score = -1.0
                    second_best_occluded_score = -1.0
                    best_student_name = "Unknown"
                    best_student_id = None

                    for student_id, data in student_map.items():
                        score, occluded_score = get_best_match_scores(data["enc"], emb)
                        if score > best_score:
                            second_best_score = best_score
                            best_score = score
                            second_best_occluded_score = best_occluded_score
                            best_occluded_score = occluded_score
                            best_student_name = data["name"]
                            best_student_id = student_id
                        elif occluded_score > second_best_occluded_score:
                            second_best_occluded_score = occluded_score
                        elif score > second_best_score:
                            second_best_score = score

                    score_margin = best_score - second_best_score
                    occluded_margin = best_occluded_score - second_best_occluded_score

                    if best_score >= SIMILARITY_THRESHOLD:
                        name = best_student_name
                        sid = best_student_id
                    elif (
                        (mask_detected or mask_model is None)
                        and best_occluded_score >= OCCLUDED_SIMILARITY_THRESHOLD
                        and occluded_margin >= OCCLUDED_MARGIN_THRESHOLD
                    ):
                        name = best_student_name
                        sid = best_student_id

                    if sid is not None:
                        updated_streak = {}
                        for existing_sid, count in identity_streak.items():
                            if existing_sid == sid:
                                updated_streak[existing_sid] = count + 1
                            else:
                                updated_streak[existing_sid] = max(0, count - 1)
                        if sid not in updated_streak:
                            updated_streak[sid] = 1
                        identity_streak = updated_streak
                        identity_count = identity_streak[sid]
                    else:
                        identity_streak = {
                            existing_sid: max(0, count - 1)
                            for existing_sid, count in identity_streak.items()
                        }
                        identity_count = 0

                    track_key = sid if sid is not None else f"pending_{face.bbox.astype(int).tolist()}"
                    is_live = update_liveness_state(liveness_state, track_key, frame, face)
                    face_ratio = get_face_ratio(frame, face)

                    if (
                        sid
                        and is_live
                        and not screen_proxy
                        and face_ratio >= MIN_ATTENDANCE_FACE_RATIO
                        and identity_count >= IDENTITY_CONFIRM_FRAMES
                        and sid not in marked_students
                    ):
                        marked_students.add(sid)
                        db_conn = connect_db()
                        db_cur = db_conn.cursor()
                        db_cur.execute("""
                            INSERT INTO attendance
                            (student_id, class_id, lecture_no, date, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (sid, class_id, lecture, today, "Present"))
                        db_conn.commit()
                        db_conn.close()

                        print(name, "marked present")
                        threading.Thread(
                            target=speak,
                            args=(f"{name} marked present",),
                            daemon=True
                        ).start()

                    if screen_proxy:
                        name = "Unknown"
                        sid = None

                    tracked_faces.append({
                        "bbox": face.bbox.astype(int),
                        "name": name,
                        "score": round(best_occluded_score if (mask_detected or mask_model is None) and sid else best_score, 2)
                    })

            for tracked_face in tracked_faces:
                x1, y1, x2, y2 = tracked_face["bbox"]
                color = (0, 255, 0) if tracked_face["name"] != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{tracked_face['name']} ({tracked_face['score']})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2
                )

            cv2.putText(
                frame,
                f"Lecture: {lecture}  Press Q to stop",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            cv2.imshow("AI Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        if session_started:
            db_conn = connect_db()
            db_cur = db_conn.cursor()
            for sid in student_ids:
                if sid not in marked_students:
                    db_cur.execute("""
                        INSERT INTO attendance
                        (student_id, class_id, lecture_no, date, status)
                        VALUES (?, ?, ?, ?, ?)
                    """, (sid, class_id, lecture, today, "Absent"))
            db_conn.commit()
            db_conn.close()

        clear_attendance_lock(pid)
        print(f"Lecture {lecture} attendance completed")

@app.route("/take_attendance/<int:class_id>")
def take_attendance(class_id):
    if not require_login():
        return redirect("/")

    if is_attendance_session_running():
        flash("Attendance camera is already running. Close the current camera window first.", "warning")
        return redirect(f"/class/{class_id}")

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*)
        FROM students s
        JOIN encodings e ON s.id = e.student_id
        WHERE s.class_id = ?
    """, (class_id,))
    has_registered_students = cur.fetchone()[0] > 0
    conn.close()

    if not has_registered_students:
        flash("No students registered", "danger")
        return redirect(f"/class/{class_id}")

    lecture = get_next_lecture_no(class_id)
    today = datetime.now().strftime("%Y-%m-%d")

    launch_code = (
        "import app; "
        f"app.run_attendance_session({class_id}, {lecture}, {today!r})"
    )
    subprocess.Popen([sys.executable, "-c", launch_code], cwd=os.getcwd())

    flash("Attendance started. Camera should open now in a separate worker window; press Q in the camera window to finish.", "success")
    return redirect(f"/class/{class_id}")
# ================= GRAPH=================
@app.route("/graph/<int:class_id>")
def graph(class_id):

    if "teacher_id" not in session:
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()

    total_lectures = get_class_lecture_count(cur, class_id)

    cur.execute("""
        SELECT 
            students.id,
            students.name,
            COUNT(DISTINCT CASE
                WHEN attendance.status = 'Present' THEN attendance.lecture_no
            END) as present_count
        FROM students
        LEFT JOIN attendance
            ON students.id = attendance.student_id
            AND attendance.class_id = ?
        WHERE students.class_id = ?
        GROUP BY students.id, students.name
        ORDER BY students.name
    """, (class_id, class_id))

    data = cur.fetchall()
    conn.close()

    names       = [row["name"] for row in data]
    counts      = [row["present_count"] for row in data]
    percentages = [
        round((row["present_count"] / total_lectures) * 100, 1) if total_lectures else 0
        for row in data
    ]

    return render_template(
        "graph.html",
        names=names,
        counts=counts,
        percentages=percentages,
        total_lectures=total_lectures,
        class_id=class_id
    )

# ================= DEFAULTERS =================
@app.route("/defaulters/<int:class_id>")
def defaulters(class_id):

    if "teacher_id" not in session:
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()

    total_lectures = get_class_lecture_count(cur, class_id)

    cur.execute("""
        SELECT 
            students.id,
            students.name,
            COUNT(DISTINCT CASE
                WHEN attendance.status = 'Present' THEN attendance.lecture_no
            END) as present_count
        FROM students
        LEFT JOIN attendance
            ON students.id = attendance.student_id
            AND attendance.class_id = ?
        WHERE students.class_id = ?
        GROUP BY students.id, students.name
        ORDER BY students.name
    """, (class_id, class_id))

    rows = cur.fetchall()
    conn.close()

    defaulters_list = []
    for row in rows:
        present = row["present_count"] or 0
        percentage = ((present / total_lectures) * 100) if total_lectures else 0
        if total_lectures and percentage < 75:
            defaulters_list.append({
                "name": row["name"],
                "present": present,
                "absent": total_lectures - present,
                "total": total_lectures,
                "percentage": round(percentage, 2)
            })

    return render_template(
        "defaulters.html",
        defaulters=defaulters_list,
        class_id=class_id,
        total_lectures=total_lectures
    )
# ================= ADD MASK SAMPLES =================
@app.route("/add_mask/<int:student_id>", methods=["POST"])
def add_mask(student_id):
    if "teacher_id" not in session:
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT class_id FROM students WHERE id=?", (student_id,))
    student = cur.fetchone()
    conn.close()

    if not student:
        flash("Student not found", "danger")
        return redirect("/dashboard")

    flash(
        "Extra mask registration is no longer needed. Register one normal face only; attendance now also checks masked and glasses-style variants automatically.",
        "info"
    )
    return redirect(f"/class/{student['class_id']}")

# ================= EXCEL SHEET =================
@app.route("/export_excel/<int:class_id>")
def export_excel(class_id):

    conn = connect_db()
    cur = conn.cursor()

    # ================= CHECK TOTAL LECTURES =================
    cur.execute("""
        SELECT MAX(lecture_no)
        FROM attendance
        WHERE class_id=?
    """, (class_id,))

    result = cur.fetchone()[0]

    if result is None:
        conn.close()
        flash("No attendance taken yet ❌")
        return redirect(f"/class/{class_id}")

    total_lectures = result

    # ================= GET STUDENTS =================
    cur.execute("""
        SELECT id, name
        FROM students
        WHERE class_id=?
        ORDER BY name
    """, (class_id,))

    students = cur.fetchall()

    report = []

    # ================= BUILD REPORT =================
    for student in students:

        sid = student["id"]
        name = f'{student["name"]} (ID:{sid})'

        for lecture in range(1, total_lectures + 1):

            # lecture date
            cur.execute("""
                SELECT date
                FROM attendance
                WHERE class_id=? AND lecture_no=?
                LIMIT 1
            """, (class_id, lecture))

            lecture_row = cur.fetchone()
            lecture_date = lecture_row["date"] if lecture_row else "-"

            # check attendance
            cur.execute("""
                SELECT status
                FROM attendance
                WHERE class_id=? AND student_id=? AND lecture_no=?
            """, (class_id, sid, lecture))

            row = cur.fetchone()

            if row:
                status = row["status"]
            else:
                status = "Absent"

            report.append([name, lecture, lecture_date, status])

    conn.close()

    # ================= CREATE EXCEL =================
    df = pd.DataFrame(
        report,
        columns=["Name", "Lecture No", "Date", "Status"]
    )

    file_path = f"class_{class_id}_attendance.xlsx"

    df.to_excel(file_path, index=False)

    return send_file(file_path, as_attachment=True)
    # ================= CREATE DATAFRAME =================
    df = pd.DataFrame(
        report,
        columns=["Name", "Lecture No", "Date", "Status"]
    )

    file_path = f"class_{class_id}_attendance.xlsx"

    df.to_excel(file_path, index=False)

    return send_file(file_path, as_attachment=True)

# ================= TEACHER SIGNUP =================

@app.route("/signup", methods=["GET", "POST"])
def signup():

    if request.method == "POST":

        name = request.form["name"]
        email = request.form["email"]
        password = request.form["password"]

        conn = connect_db()
        cur = conn.cursor()

        # Check if email already exists
        cur.execute(
            "SELECT * FROM teachers WHERE email=?",
            (email,)
        )

        existing_user = cur.fetchone()

        if existing_user:
            flash("Email already registered ❌", "danger")
            conn.close()
            return redirect("/signup")

        # Insert new teacher
        cur.execute(
            """
            INSERT INTO teachers (name, email, password)
            VALUES (?, ?, ?)
            """,
            (name, email, password)
        )

        conn.commit()
        conn.close()

        flash("Signup successful ✅ Please login", "success")
        return redirect("/")

    return render_template("signup.html")
# ================= DELETE STUDENT =================
@app.route("/delete_student/<int:student_id>", methods=["POST"])
def delete_student(student_id):

    # 🔒 Login protection
    if "teacher_id" not in session:
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()

    # 1️⃣ Get class_id BEFORE delete
    cur.execute("""
        SELECT class_id
        FROM students
        WHERE id = ?
    """, (student_id,))

    row = cur.fetchone()

    if not row:
        conn.close()
        flash("Student not found ❌")
        return redirect("/dashboard")

    class_id = row["class_id"]

    # 2️⃣ Delete encodings
    cur.execute("""
        DELETE FROM encodings
        WHERE student_id = ?
    """, (student_id,))

    # 3️⃣ Delete attendance
    cur.execute("""
        DELETE FROM attendance
        WHERE student_id = ?
    """, (student_id,))

    # 4️⃣ Delete student
    cur.execute("""
        DELETE FROM students
        WHERE id = ?
    """, (student_id,))

    conn.commit()
    conn.close()

    flash("Student Deleted Successfully 🗑️")

    return redirect(f"/class/{class_id}")

# ================= DELETE CLASS =================
@app.route("/delete_class/<int:class_id>", methods=["POST"])
def delete_class(class_id):

    if "teacher_id" not in session:
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()

    # Get all students of class
    cur.execute("""
        SELECT id FROM students
        WHERE class_id=?
    """, (class_id,))

    students = cur.fetchall()

    # Delete each student data
    for s in students:
        sid = s["id"]

        cur.execute("DELETE FROM encodings WHERE student_id=?", (sid,))
        cur.execute("DELETE FROM attendance WHERE student_id=?", (sid,))

    # Delete students
    cur.execute("DELETE FROM students WHERE class_id=?", (class_id,))

    # Delete attendance by class
    cur.execute("DELETE FROM attendance WHERE class_id=?", (class_id,))

    # Delete class
    cur.execute("DELETE FROM classes WHERE id=?", (class_id,))

    conn.commit()
    conn.close()

    flash("Class Deleted Successfully 🗂️")

    return redirect("/dashboard")

# ================= INDIVIDUAL REPORT =================
@app.route("/student_report/<int:student_id>", methods=["GET", "POST"])
def student_report(student_id):

    if "teacher_id" not in session:
        return redirect("/")

    conn = connect_db()
    cur = conn.cursor()

    # Student info
    cur.execute("SELECT name, class_id FROM students WHERE id=?", (student_id,))
    student = cur.fetchone()

    if not student:
        conn.close()
        flash("Student not found ❌")
        return redirect("/dashboard")

    name = student["name"]
    class_id = student["class_id"]

    if request.method == "POST":
        lecture_no = int(request.form["lecture_no"])
        new_status = request.form["status"]

        if new_status not in ("Present", "Absent"):
            conn.close()
            flash("Invalid attendance status", "danger")
            return redirect(f"/student_report/{student_id}")

        cur.execute("""
            SELECT date
            FROM attendance
            WHERE class_id=? AND lecture_no=?
            ORDER BY id ASC
            LIMIT 1
        """, (class_id, lecture_no))
        lecture_row = cur.fetchone()

        if not lecture_row:
            conn.close()
            flash("Lecture not found", "danger")
            return redirect(f"/student_report/{student_id}")

        lecture_date = lecture_row["date"]

        cur.execute("""
            SELECT id
            FROM attendance
            WHERE class_id=? AND student_id=? AND lecture_no=?
        """, (class_id, student_id, lecture_no))
        attendance_row = cur.fetchone()

        if attendance_row:
            cur.execute("""
                UPDATE attendance
                SET status=?, date=?
                WHERE id=?
            """, (new_status, lecture_date, attendance_row["id"]))
        else:
            cur.execute("""
                INSERT INTO attendance
                (student_id, class_id, lecture_no, date, status)
                VALUES (?, ?, ?, ?, ?)
            """, (student_id, class_id, lecture_no, lecture_date, new_status))

        conn.commit()
        flash(f"Lecture {lecture_no} updated to {new_status}", "success")

    cur.execute("""
        SELECT lecture_no, MIN(date) AS date
        FROM attendance
        WHERE class_id=?
        GROUP BY lecture_no
        ORDER BY lecture_no
    """, (class_id,))
    lectures = cur.fetchall()
    total_lectures = len(lectures)

    # Per-lecture attendance for this student
    records = []
    present_count = 0

    for lecture in lectures:
        lecture_no = lecture["lecture_no"]
        lecture_date = lecture["date"] or "-"
        # Get this student's status for this lecture
        cur.execute("""
            SELECT status FROM attendance
            WHERE class_id=? AND student_id=? AND lecture_no=?
        """, (class_id, student_id, lecture_no))
        status_row = cur.fetchone()
        status = status_row["status"] if status_row else "Absent"

        if status == "Present":
            present_count += 1

        records.append({
            "lecture_no": lecture_no,
            "date": lecture_date,
            "status": status
        })

    conn.close()

    percentage = round((present_count / total_lectures) * 100, 2) if total_lectures > 0 else 0

    return render_template(
        "student_report.html",
        student_id=student_id,
        name=name,
        present=present_count,
        total=total_lectures,
        percentage=percentage,
        records=records,
        class_id=class_id
    )

# ================= RUN =================

if __name__=="__main__":
    app.run(debug=True, use_reloader=False)

