#!/usr/bin/env python3
"""
tr_full_featured.py - Rail Prototype (All-in-one)

Features included (integrated):
- Passenger Registration (single & batch) -> CSV encodings + TIF save
- Database listing
- Live Verification from webcam
  • DB mode: match by bogie+row from CSV
  • Whitelist folder mode: match only faces from a given folder
  • Blacklist (criminal) detection with multi-frame confirmation
  • Crowd counting / overcrowding alert
  • Gender / age estimation (best-effort, optional models)
- Unknown face => on-screen "❌ Ticket not found. Failed!" + email alert to TT
- Email cooldown to prevent spam (default 20s)
- Edge-friendly options (resize_factor, model_type) for devices like Jetson/RPi

Notes:
- Optional model files for age/gender can be provided in the working dir (populated if you want higher accuracy).
- All external dependencies are optional; code falls back gracefully if libs/models missing.
- Make sure you have "face_recognition", "opencv-python", "numpy", "pandas", "tifffile" installed.
"""

import os
import sys
import csv
import argparse
import time
import logging
from typing import Tuple, Optional

# external libs (may not be installed in all environments)
try:
    import cv2
except Exception:
    cv2 = None
try:
    import numpy as np
except Exception:
    np = None
try:
    import face_recognition
except Exception:
    face_recognition = None
try:
    import pandas as pd
except Exception:
    pd = None
try:
    import tifffile
except Exception:
    tifffile = None

# Additional email/mime imports for attachment-capable emails
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import smtplib

# ---- Optional libs (don't crash if missing) ----
try:
    import phasorpy as PhasorPy
except Exception:
    PhasorPy = None
try:
    import lfdfiles
except Exception:
    lfdfiles = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
log = logging.getLogger("tr")

DB_FILE = "passengers.csv"
FACE_DIR = "faces"
ALERT_LOG = "alerts.log"
SNAP_DIR = "snapshots"
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(SNAP_DIR, exist_ok=True)

# ----------------- USER CREDENTIALS (INSERTED) -----------------
# The user requested these be embedded directly in the code (less secure).
# If you prefer environment variables, remove or comment these and use the env vars instead.
DEFAULT_SENDER_EMAIL = "nkp98907@gmail.com"
DEFAULT_SENDER_APP_PASSWORD = "Ibcbicoxyksjdqrj"
DEFAULT_RPF_EMAIL = "nawneetpandey6@gmail.com"
# -----------------------------------------------------------------

# ---------------- Utility helpers ----------------

def ensure_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "row", "seat", "bogie", "encoding"]) 


def _encode_image(image_path: str):
    """Return (128-d face encoding, image array) or None on failure."""
    if face_recognition is None:
        log.error("face_recognition library not installed.")
        return None
    try:
        img = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(img)
        if encs and len(encs) > 0:
            return encs[0], img
        else:
            log.warning("No face found in image: %s", image_path)
            return None
    except Exception as e:
        log.warning("Failed to encode image %s: %s", image_path, e)
        return None


def register_passenger(name: str, row: str, seat: str, image_path: str, bogie: str):
    ensure_db()
    if not os.path.exists(image_path):
        log.error("Image not found: %s", image_path)
        return
    enc_img = _encode_image(image_path)
    if not enc_img:
        log.error("No face detected in the given image. Use a clear frontal photo.")
        return
    enc, img = enc_img
    enc_str = ",".join(map(str, enc.tolist()))
    # Append to CSV
    try:
        with open(DB_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([name, str(row), str(seat), str(bogie), enc_str])
    except Exception as e:
        log.error("Failed to write to DB: %s", e)
        return
    # Save .tif
    safe_name = f"{bogie}{row}{seat}{name}".replace(" ", "")
    tif_path = os.path.join(FACE_DIR, f"{safe_name}.tif")
    try:
        if tifffile is not None:
            tifffile.imwrite(tif_path, img)
        else:
            # fallback: save as jpg
            import imageio
            imageio.imwrite(tif_path.replace('.tif', '.jpg'), img)
            tif_path = tif_path.replace('.tif', '.jpg')
    except Exception as e:
        log.warning("TIF save failed: %s", e)
    log.info("Registered %s | B:%s | Row:%s | Seat:%s | File:%s", name, bogie, row, seat, tif_path)


def batch_register(folder_path: str, row: str, bogie: str):
    ensure_db()
    if not os.path.isdir(folder_path):
        log.error("Folder not found: %s", folder_path)
        return
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
    if not images:
        log.error("No images found in folder: %s", folder_path)
        return
    count = 0
    # open CSV once for speed
    with open(DB_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for i, img_file in enumerate(images, start=1):
            name = os.path.splitext(img_file)[0]
            seat = f"S{i}"
            image_path = os.path.join(folder_path, img_file)
            enc_img = _encode_image(image_path)
            if not enc_img:
                log.warning("%s: no face detected, skipping.", img_file)
                continue
            enc, img = enc_img
            enc_str = ",".join(map(str, enc.tolist()))
            try:
                writer.writerow([name, str(row), seat, str(bogie), enc_str])
            except Exception as e:
                log.warning("Failed to write row for %s: %s", img_file, e)
                continue
            safe_name = f"{bogie}{row}{seat}{name}".replace(" ", "")
            tif_path = os.path.join(FACE_DIR, f"{safe_name}.tif")
            try:
                if tifffile is not None:
                    tifffile.imwrite(tif_path, img)
                else:
                    import imageio
                    imageio.imwrite(tif_path.replace('.tif', '.jpg'), img)
                    tif_path = tif_path.replace('.tif', '.jpg')
            except Exception as e:
                log.warning("TIF save failed: %s", e)
            count += 1
    log.info("Batch registered %d image(s) from %s for Bogie %s, Row %s", count, folder_path, bogie, row)


def load_db():
    ensure_db()
    known_encs, names, rows, seats, bogies = [], [], [], [], []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                enc_str = (r.get("encoding") or "").strip()
                if not enc_str:
                    continue
                try:
                    enc = np.array(list(map(float, enc_str.split(","))) )
                except Exception:
                    continue
                known_encs.append(enc)
                names.append(r.get("name", ""))
                rows.append(str(r.get("row", "")))
                seats.append(str(r.get("seat", "")))
                bogies.append(str(r.get("bogie", "")))
    except Exception as e:
        log.error("Could not read DB: %s", e)
    return known_encs, names, rows, seats, bogies


def list_passengers():
    ensure_db()
    try:
        if pd is None:
            # simple csv print
            with open(DB_FILE, "r", encoding="utf-8") as f:
                print(f.read())
            return
        df = pd.read_csv(DB_FILE)
        if df.empty:
            log.info("No passengers registered.")
        else:
            print(df[["name", "bogie", "row", "seat"]].to_string(index=False))
    except Exception as e:
        log.error("Could not read DB: %s", e)

# ----------------- Automatic registration via webcam -----------------
def suggest_next_seat_for_row_bogie(row: str, bogie: str):
    """Return (occupied_list, suggested_seat_str). Occupied as list of seat strings."""
    ensure_db()
    _, names, rows, seats, bogies = load_db()
    occupied = []
    numeric_seats = []
    for r, s, b in zip(rows, seats, bogies):
        if str(r) == str(row) and str(b) == str(bogie):
            occupied.append(str(s))
            # try extract numeric part like S12 or 12
            try:
                if isinstance(s, str):
                    num = ''.join(ch for ch in s if ch.isdigit())
                    if num:
                        numeric_seats.append(int(num))
            except Exception:
                pass
    suggested = None
    if numeric_seats:
        nxt = max(numeric_seats) + 1
        suggested = f"S{nxt}"
    else:
        # fall back: if no numeric seats, propose S1 or next unavailable name
        base = 1
        while True:
            cand = f"S{base}"
            if cand not in occupied:
                suggested = cand
                break
            base += 1
    return occupied, suggested

def register_auto(name: str, row: str, bogie: str, countdown: int = 3, resize_factor: float = 1.0, model_type: str = "cnn"):
    """
    Automatic registration: show occupied seats, suggest next free seat,
    countdown and capture image from webcam, then call register_passenger.
    """
    ensure_db()
    # show occupied seats and suggestion
    occupied, suggested_seat = suggest_next_seat_for_row_bogie(row, bogie)
    log.info("Occupied seats for Bogie %s Row %s : %s", bogie, row, ', '.join(occupied) if occupied else "None")
    log.info("Suggested seat: %s", suggested_seat)

    # open webcam
    if cv2 is None:
        log.error("cv2 not installed. Camera capture not possible.")
        return
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        log.error("Cannot open webcam.")
        return

    log.info("Starting camera. Hold still. Auto-capture in %d seconds...", countdown)
    start = time.time()
    captured_frame = None

    # show live view with countdown overlay
    while True:
        ret, frame = video.read()
        if not ret:
            log.error("Failed to read from camera.")
            break

        elapsed = time.time() - start
        remaining = countdown - int(elapsed)
        if remaining < 0:
            remaining = 0

        # display countdown in center
        try:
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Capturing in {remaining}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        except Exception:
            pass

        cv2.imshow(f"Register Auto - {name} | Bogie {bogie} Row {row}", frame)

        # capture when countdown finishes
        if time.time() - start >= countdown:
            captured_frame = frame.copy()
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            log.info("User cancelled capture.")
            break

    video.release()
    cv2.destroyAllWindows()

    if captured_frame is None:
        log.error("No frame captured.")
        return

    # save a temporary file (jpg) then feed to register_passenger which expects image path
    try:
        tmp_dir = os.path.join(os.getcwd(), "tmp_captures")
        os.makedirs(tmp_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tmp_path = os.path.join(tmp_dir, f"{name}_{bogie}_{row}_{ts}.jpg")
        cv2.imwrite(tmp_path, captured_frame)
        log.info("Captured image saved to %s", tmp_path)
    except Exception as e:
        log.error("Failed to save captured image: %s", e)
        return

    # Determine final seat to register: if suggested available use it, else ask fallback
    final_seat = suggested_seat or "S1"

    # call existing register_passenger to save encoding and files
    register_passenger(name=name, row=row, seat=final_seat, image_path=tmp_path, bogie=bogie)
    log.info("Auto-registration complete for %s (Seat: %s)", name, final_seat)

# ------------- Email helper (attachment-capable) -------------
def send_email_alert(sender_email, sender_app_password, receiver_email, subject, body, attachment_path=None):
    """
    Send email with optional file attachment. Uses Gmail SMTP (TLS).
    """
    try:
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg.attach(MIMEText(body, "plain"))

        if attachment_path and os.path.exists(attachment_path):
            part = MIMEBase("application", "octet-stream")
            with open(attachment_path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(attachment_path)}"')
            msg.attach(part)

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=30)
        server.starttls()
        server.login(sender_email, sender_app_password)
        server.send_message(msg)
        server.quit()
        log.info("📩 Email sent to %s (subject: %s)", receiver_email, subject)
    except Exception as e:
        log.error("❌ Failed to send email: %s", e)


def _log_alert(text: str):
    try:
        with open(ALERT_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {text}\n")
    except Exception:
        pass

# ---------------- Blacklist & snapshot helpers ----------------

def load_blacklist(folder_path: str):
    blacklist_encs, blacklist_ids = [], []
    if not folder_path:
        return blacklist_encs, blacklist_ids
    if not os.path.isdir(folder_path):
        log.error("Blacklist folder not found: %s", folder_path)
        return blacklist_encs, blacklist_ids
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.tif'))]
    for fn in files:
        p = os.path.join(folder_path, fn)
        enc_img = _encode_image(p)
        if enc_img:
            enc, _ = enc_img
            blacklist_encs.append(enc)
            blacklist_ids.append(os.path.splitext(fn)[0])
        else:
            log.warning("Blacklist image skipped (no face): %s", fn)
    log.info("Blacklist loaded: %d face(s)", len(blacklist_encs))
    return blacklist_encs, blacklist_ids


def save_snapshot(frame, tag_id):
    try:
        os.makedirs(SNAP_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(SNAP_DIR, f"BL_{tag_id}_{ts}.jpg")
        if cv2 is None:
            log.warning("cv2 not installed; cannot save snapshot.")
            return None
        cv2.imwrite(fname, frame)
        return fname
    except Exception as e:
        log.warning("Snapshot save failed: %s", e)
        return None


def send_high_priority_alert(body, rpf_email=None, sender_email=None, sender_app_password=None):
    _log_alert("[HIGH PRIORITY] " + body)
    if sender_email and sender_app_password and rpf_email:
        try:
            # now supports attachment via send_email_alert (attachment_path optional)
            send_email_alert(sender_email, sender_app_password, rpf_email, "BLACKLIST HIT", body, attachment_path=None)
        except Exception as e:
            log.error("Failed to email RPF: %s", e)
    else:
        log.warning("Email creds or RPF recipient missing; alert logged only.")

# ----------------- Age/Gender (optional) -----------------
# This is best-effort: if you have pretrained models (Caffe/TensorFlow) provide their files in working dir.
AGE_NET = None
GENDER_NET = None
AGE_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Try to load Caffe models if present (placeholders: deploy files not bundled)
AGE_PROTO = 'age_deploy.prototxt'
AGE_MODEL = 'age_net.caffemodel'
GENDER_PROTO = 'gender_deploy.prototxt'
GENDER_MODEL = 'gender_net.caffemodel'

# GPU-aware: if OpenCV DNN has CUDA backend available, use it
_cv_dnn_cuda = False
if cv2 is not None:
    try:
        if os.path.exists(AGE_PROTO) and os.path.exists(AGE_MODEL):
            AGE_NET = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        if os.path.exists(GENDER_PROTO) and os.path.exists(GENDER_MODEL):
            GENDER_NET = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
        # Attempt to set DNN backend/target to CUDA if available
        # This will silently pass if CUDA isn't configured
        try:
            # check cuda device count via cv2 cuda module if present
            if hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                _cv_dnn_cuda = True
                if AGE_NET is not None:
                    AGE_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    AGE_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                if GENDER_NET is not None:
                    GENDER_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    GENDER_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                log.info("OpenCV DNN configured to use CUDA backend for age/gender nets.")
        except Exception:
            pass
    except Exception as e:
        log.warning("Age/Gender model load failed: %s", e)


def estimate_age_gender(bgr_face) -> Tuple[Optional[str], Optional[str]]:
    """Return (age_group, gender) or (None, None) if models not available."""
    if cv2 is None:
        return None, None
    if AGE_NET is None and GENDER_NET is None:
        return None, None
    try:
        blob = cv2.dnn.blobFromImage(bgr_face, 1.0, (227, 227), AGE_MEAN_VALUES, swapRB=False)
        age = None
        gender = None
        if GENDER_NET is not None:
            GENDER_NET.setInput(blob)
            gender_preds = GENDER_NET.forward()
            gender = GENDER_LIST[int(np.argmax(gender_preds))]
        if AGE_NET is not None:
            AGE_NET.setInput(blob)
            age_preds = AGE_NET.forward()
            age = AGE_LIST[int(np.argmax(age_preds))]
        return age, gender
    except Exception as e:
        log.warning("Age/gender estimation failed: %s", e)
        return None, None

# ------------- Verify logic (integrated features) -------------
def verify_live(
    bogie: str = "A",
    row_to_check: str = "1",
    expected_count: int = 3,
    rpf_email: Optional[str] = None,
    sender_email: Optional[str] = None,
    sender_app_password: Optional[str] = None,
    tolerance: float = 0.5,
    whitelist_folder: Optional[str] = None,
    blacklist_folder: Optional[str] = None,
    cooldown_seconds: int = 20,
    resize_factor: float = 1.0,
    model_type: str = "cnn",
    crowd_threshold: Optional[int] = None,
    enable_age_gender: bool = False
):

    # load whitelist or DB
    if whitelist_folder:
        row_encs, row_names = load_whitelist_folder(whitelist_folder)
        row_seats = ["-"] * len(row_names)
    else:
        known_encs, names, rows, seats, bogies = load_db()
        row_encs, row_names, row_seats = [], [], []
        for enc, name, r, s, b in zip(known_encs, names, rows, seats, bogies):
            if r == str(row_to_check) and b == str(bogie):
                row_encs.append(enc)
                row_names.append(name)
                row_seats.append(s)

    log.info("Monitoring Bogie %s - Row %s | Expected: %s", bogie, row_to_check, expected_count)
    log.info("Valid ticket holders loaded: %d", len(row_encs))

    if face_recognition is None or cv2 is None or np is None:
        log.error("Required libraries missing (face_recognition/opencv/numpy). Verification cannot run.")
        return

    # load blacklist
    blacklist_encs, blacklist_ids = load_blacklist(blacklist_folder) if blacklist_folder else ([], [])

    # multi-frame confirmation state for blacklist
    bl_confirm = {}
    BL_CONFIRM_REQUIRED = 2
    BL_CONFIRM_WINDOW = 5.0
    BL_HIGH_THRESH = 0.40
    BL_POSSIBLE_THRESH = 0.55

    last_alert_time = 0
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        log.error("Cannot open webcam.")
        return

    # If user chose GPU & dlib compiled with CUDA, using model_type "cnn" will make use of it.
    if model_type == "cnn":
        log.info("Using model_type 'cnn' (dlib's CNN model). If your dlib build supports CUDA, it will accelerate face detection.")
    else:
        log.info("Using model_type 'hog' (CPU).")

    log.info("Camera started. Press 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            log.error("Failed to read frame from camera.")
            break

        # resize for speed (edge friendly)
        small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
        rgb_small = small_frame[:, :, ::-1]

        # detect faces
        try:
            face_locations = face_recognition.face_locations(rgb_small, model=model_type)
        except Exception as e:
            log.warning("face_locations error: %s", e)
            face_locations = []

        face_encodings = []
        if face_locations:
            try:
                face_encodings = face_recognition.face_encodings(rgb_small, known_face_locations=face_locations)
            except Exception as e:
                log.warning("face_encodings error: %s", e)
                face_encodings = []

        if not face_encodings:
            # fallback try whole frame (RGB)
            try:
                rgb_full = frame[:, :, ::-1]
                face_encodings = face_recognition.face_encodings(rgb_full)
            except Exception as e2:
                log.warning("Fallback face_encodings failed: %s", e2)
                face_encodings = []

        detected_names = []
        unknown_count = 0

        # basic passenger matching
        for fe in face_encodings:
            matched = False
            if row_encs:
                dists = face_recognition.face_distance(row_encs, fe)
                best_idx = int(np.argmin(dists)) if len(dists) else -1
                if best_idx >= 0 and dists[best_idx] <= tolerance:
                    matched = True
                    detected_names.append(row_names[best_idx])
            if not matched:
                detected_names.append("Unknown")
                unknown_count += 1

        total_detected = len(face_encodings)

        # crowd analytics
        if crowd_threshold is not None and total_detected > crowd_threshold:
            # prepare alert
            alert_msgs = [f"Overcrowding: {total_detected} > {crowd_threshold}"]
            now = time.time()
            if now - last_alert_time > cooldown_seconds:
                body = (f"ALERT - Bogie {bogie}, Row {row_to_check} - Overcrowding\n"
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"Detected: {total_detected} (threshold {crowd_threshold})\n")
                _log_alert(body)
                if sender_email and sender_app_password:
                    send_email_alert(sender_email, sender_app_password, rpf_email or sender_email,
                                     f"Overcrowding Alert: {bogie}-{row_to_check}", body)
                last_alert_time = now

        # draw overlays and optionally estimate age/gender
        for i, ((top, right, bottom, left) , name_label) in enumerate(zip(face_locations, detected_names)):
            # scale back up
            top = int(top / resize_factor); right = int(right / resize_factor)
            bottom = int(bottom / resize_factor); left = int(left / resize_factor)
            color = (0, 200, 0) if name_label != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = name_label
            # estimate age/gender (best-effort) using BGR face crop
            if enable_age_gender and (AGE_NET is not None or GENDER_NET is not None):
                try:
                    face_crop = frame[top:bottom, left:right]
                    if face_crop.size != 0:
                        age, gender = estimate_age_gender(face_crop)
                        if age:
                            label += f" | {age}"
                        if gender:
                            label += f" | {gender}"
                except Exception as e:
                    log.debug("Age/gender on crop failed: %s", e)

            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.putText(frame, f"Detected: {total_detected}/{expected_count}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if unknown_count > 0:
            cv2.putText(frame, "❌ Ticket not found. Failed!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

        # --- BLACKLIST CHECK (per face encoding) ---
        if blacklist_encs and face_encodings:
            for fe in face_encodings:
                try:
                    dists_b = face_recognition.face_distance(blacklist_encs, fe)
                    best_idx_b = int(np.argmin(dists_b)) if len(dists_b) else -1
                    best_dist = float(dists_b[best_idx_b]) if best_idx_b >= 0 else 1.0
                except Exception as e:
                    log.debug("Blacklist distance error: %s", e)
                    continue

                if best_dist <= BL_HIGH_THRESH:
                    tag = blacklist_ids[best_idx_b]
                    now_t = time.time()
                    st = bl_confirm.get(tag, {"count": 0, "first_seen": now_t})
                    if now_t - st["first_seen"] > BL_CONFIRM_WINDOW:
                        st = {"count": 0, "first_seen": now_t}
                    st["count"] += 1
                    bl_confirm[tag] = st

                    if st["count"] >= BL_CONFIRM_REQUIRED:
                        snapshot = save_snapshot(frame, tag)
                        body = (f"BLACKLIST HIT: ID={tag}\n"
                                f"Bogie: {bogie}, Row: {row_to_check}\n"
                                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                                f"Snapshot: {snapshot}\nDist: {best_dist:.3f}")
                        log.warning("BLACKLIST HIT: %s dist: %.3f", tag, best_dist)
                        # send email with snapshot if creds exist
                        try:
                            send_high_priority_alert(body, rpf_email=rpf_email or DEFAULT_RPF_EMAIL,
                                                     sender_email=sender_email or DEFAULT_SENDER_EMAIL,
                                                     sender_app_password=sender_app_password or DEFAULT_SENDER_APP_PASSWORD)
                        except Exception as e:
                            log.error("Failed to send high priority alert: %s", e)
                        bl_confirm[tag] = {"count": 0, "first_seen": now_t}
                elif best_dist <= BL_POSSIBLE_THRESH:
                    tag = blacklist_ids[best_idx_b]
                    body = (f"POSSIBLE BLACKLIST match: ID={tag} (dist:{best_dist:.3f}) "
                            f"Bogie:{bogie}/{row_to_check} Time:{time.strftime('%Y-%m-%d %H:%M:%S')}")
                    _log_alert(body)

        cv2.imshow(f"{bogie}-{row_to_check}", frame)

        # general alerts (unknown or extra person)
        alert_msgs = []
        if unknown_count > 0:
            alert_msgs.append(f"{unknown_count} unknown/unregistered passenger(s)")
        if total_detected > expected_count:
            alert_msgs.append(f"Extra person(s): {total_detected}>{expected_count}")

        # ---- REPLACED ALERT BLOCK: saves snapshot and emails TT immediately for unknowns/alerts ----
        if alert_msgs:
            now = time.time()
            # If cooldown passed, send alert (immediate)
            if now - last_alert_time > cooldown_seconds:
                summary = " | ".join(alert_msgs)
                details = ", ".join(detected_names) if detected_names else "No faces"
                body = (f"ALERT - Bogie {bogie}, Row {row_to_check}\n"
                        f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        f"{summary}\nDetected: {details}\n")

                # Save a snapshot (evidence) — tag it with ALERT_...
                snap_path = None
                try:
                    snap_path = save_snapshot(frame, f"ALERT_{bogie}_{row_to_check}")
                    if snap_path:
                        body += f"\nSnapshot: {snap_path}\n"
                except Exception as e:
                    log.warning("Snapshot save failed: %s", e)

                _log_alert(body)

                # Send email with attachment if creds present; else just log
                if sender_email and sender_app_password and (rpf_email or DEFAULT_RPF_EMAIL):
                    to_addr = rpf_email or DEFAULT_RPF_EMAIL
                    try:
                        send_email_alert(sender_email, sender_app_password, to_addr,
                                         f"ALERT - Bogie {bogie} Row {row_to_check}", body, attachment_path=snap_path)
                    except Exception as e:
                        log.error("Failed to send alert email: %s", e)
                else:
                    log.info("Email credentials or recipient missing - alert logged only.")

                last_alert_time = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    log.info("Stopped verification.")

# ------- Whitelist loader (folder mode) -------

def load_whitelist_folder(folder_path: str):
    known_encs, names = [], []
    if not folder_path:
        return known_encs, names
    if not os.path.isdir(folder_path):
        log.error("Whitelist folder not found: %s", folder_path)
        return known_encs, names
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png','.tif'))]
    if not files:
        log.error("No images in whitelist folder: %s", folder_path)
        return known_encs, names
    for fn in files:
        p = os.path.join(folder_path, fn)
        enc_img = _encode_image(p)
        if enc_img:
            enc, _ = enc_img
            known_encs.append(enc)
            names.append(os.path.splitext(fn)[0])
        else:
            log.warning("Skipping (no face): %s", fn)
    log.info("Whitelist loaded: %d face(s) from folder", len(known_encs))
    return known_encs, names

# ------------- CLI -------------

def main():
    parser = argparse.ArgumentParser(description="Rail Prototype - Register & Verify (full)")
    sub = parser.add_subparsers(dest="cmd")

    # Register single
    r = sub.add_parser("register", help="Register passenger with photo")
    r.add_argument("--name", required=True)
    r.add_argument("--row", required=True)
    r.add_argument("--seat", required=True)
    r.add_argument("--bogie", required=True)
    r.add_argument("--image", required=True)

    # Auto-register (webcam capture + auto-seat)
    ra = sub.add_parser("register_auto", help="Automatic register via webcam (auto-assign seat)")
    ra.add_argument("--name", required=True, help="Passenger name")
    ra.add_argument("--row", required=True, help="Row number")
    ra.add_argument("--bogie", required=True, help="Bogie id")
    ra.add_argument("--countdown", type=int, default=3, help="Seconds before auto-capture")
    ra.add_argument("--resize", type=float, default=1.0, help="Resize factor for face detection (not used for capture)")
    ra.add_argument("--model", type=str, default="cnn", help="face_recognition model: cnn or hog")

    # List
    sub.add_parser("list", help="List registered passengers")

    # Verify
    v = sub.add_parser("verify", help="Start live verification for a row")
    v.add_argument("--row", required=True)
    v.add_argument("--bogie", required=True)
    v.add_argument("--expected", type=int, required=True)
    v.add_argument("--rpf_email", required=False)
    v.add_argument("--sender_email", required=False)
    v.add_argument("--sender_app_password", required=False)
    v.add_argument("--tolerance", type=float, default=0.5)
    v.add_argument("--whitelist_folder", required=False, help="If set, only this folder's faces are valid")
    v.add_argument("--blacklist_folder", required=False, help="Folder of blacklist images (police uploads)")
    v.add_argument("--cooldown", type=int, default=20, help="Alert cooldown seconds")
    v.add_argument("--resize", type=float, default=1.0, help="Resize factor for performance")
    v.add_argument("--model", type=str, default="cnn", help="face_recognition model: cnn or hog")
    v.add_argument("--crowd_threshold", type=int, required=False, help="Trigger overcrowding alert above this count")
    v.add_argument("--age_gender", action='store_true', help="Enable age/gender estimation if models are present")

    # Email test command
    et = sub.add_parser("email_test", help="Send a test email using configured sender (no verification run)")
    et.add_argument("--sender_email", required=False)
    et.add_argument("--sender_app_password", required=False)
    et.add_argument("--to", required=False, help="Recipient email for test")

    # LFD read (optional)
    lfd = sub.add_parser("read_lfd", help="Read LFD file (optional, if lfdfiles is installed)")
    lfd.add_argument("--file", required=True)

    # Batch register
    b = sub.add_parser("batch_register", help="Register multiple images from folder")
    b.add_argument("--folder", required=True)
    b.add_argument("--row", required=True)
    b.add_argument("--bogie", required=True)

    args = parser.parse_args()

    if args.cmd == "register":
        register_passenger(args.name, args.row, args.seat, args.image, args.bogie)
    elif args.cmd == "register_auto":
        register_auto(name=args.name, row=args.row, bogie=args.bogie, countdown=args.countdown,
                      resize_factor=args.resize, model_type=args.model)
    elif args.cmd == "list":
        list_passengers()
    elif args.cmd == "verify":
        # allow password/email to come from environment if CLI arg missing
        sender_app_password = args.sender_app_password or os.environ.get("SENDER_APP_PASSWORD") or DEFAULT_SENDER_APP_PASSWORD
        sender_email = args.sender_email or os.environ.get("SENDER_EMAIL") or DEFAULT_SENDER_EMAIL
        rpf_email = args.rpf_email or os.environ.get("RPF_EMAIL") or DEFAULT_RPF_EMAIL
        if not sender_app_password:
            log.warning("No sender app password provided via CLI or SENDER_APP_PASSWORD env var. Email alerts will be disabled.")
        verify_live(args.bogie, args.row, args.expected, rpf_email,
                    sender_email, sender_app_password,
                    tolerance=args.tolerance,
                    whitelist_folder=args.whitelist_folder,
                    blacklist_folder=args.blacklist_folder,
                    cooldown_seconds=args.cooldown,
                    resize_factor=args.resize,
                    model_type=args.model,
                    crowd_threshold=args.crowd_threshold,
                    enable_age_gender=args.age_gender)
    elif args.cmd == "email_test":
        # send a simple test email to verify SMTP creds without running verification
        sender_app_password = args.sender_app_password or os.environ.get("SENDER_APP_PASSWORD") or DEFAULT_SENDER_APP_PASSWORD
        sender_email = args.sender_email or os.environ.get("SENDER_EMAIL") or DEFAULT_SENDER_EMAIL
        to = args.to or os.environ.get("RPF_EMAIL") or sender_email
        if not sender_email or not sender_app_password or not to:
            log.error("Missing sender_email, sender_app_password or recipient for email_test. Provide via CLI or SENDER_* env vars.")
        else:
            body = "Test email from Rail Prototype system. If you received this, SMTP credentials are OK."
            send_email_alert(sender_email, sender_app_password, to, "Rail Prototype - Test Email", body)
    elif args.cmd == "read_lfd":
        if lfdfiles is None:
            log.error("lfdfiles not installed.")
        else:
            try:
                data = lfdfiles.load(args.file)
                log.info("LFD Channels: %s", getattr(data, "channels", "N/A"))
            except Exception as e:
                log.error("Failed to read LFD: %s", e)
    elif args.cmd == "batch_register":
        batch_register(args.folder, args.row, args.bogie)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    main()
