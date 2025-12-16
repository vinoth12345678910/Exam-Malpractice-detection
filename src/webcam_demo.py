from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque

# =========================
# LOAD MODEL
# =========================
model = YOLO("best.pt")

# =========================
# TEMPORAL BUFFERS
# =========================
HEAD_HISTORY = deque(maxlen=15)
MOTION_HISTORY = deque(maxlen=15)
STATUS_HISTORY = deque(maxlen=10)

# =========================
# TUNED PARAMETERS
# =========================
YOLO_CONF = 0.15
HEAD_CONF = 0.3

HEAD_SUSPICIOUS_FRAMES = 6
HEAD_MALPRACTICE_FRAMES = 10

MOTION_THRESH = 20        # avg pixel diff
MOTION_FRAMES = 8

ESCALATION_FRAMES = 5       # confirmation window

# =========================
# DECISION LOGIC
# =========================
def decide_status(detections, motion_score):
    """
    detections: [(cls, conf, (x1,y1,x2,y2))]
    motion_score: float
    """

    # ---- RULE 1: PHONE (HARD RULE)
    for cls, conf, _ in detections:
        if cls == 1 and conf >= 0.15:
            return "MALPRACTICE"

    # ---- RULE 2: HEAD POSE (TEMPORAL)
    head_turn = any(
        cls in [2, 3] and conf >= HEAD_CONF
        for cls, conf, _ in detections
    )
    HEAD_HISTORY.append(1 if head_turn else 0)

    head_sum = sum(HEAD_HISTORY)

    if head_sum >= HEAD_MALPRACTICE_FRAMES:
        return "MALPRACTICE"

    if head_sum >= HEAD_SUSPICIOUS_FRAMES:
        return "SUSPICIOUS"

    # ---- RULE 3: MOTION (TEMPORAL)
    MOTION_HISTORY.append(motion_score)

    high_motion_frames = sum(1 for m in MOTION_HISTORY if m > MOTION_THRESH)

    if high_motion_frames >= MOTION_FRAMES:
        return "SUSPICIOUS"

    return "SAFE"

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(0)
prev_gray = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # -------------------------
    # MOTION SCORE (GLOBAL)
    # -------------------------
    motion_score = 0
    if prev_gray is not None:
        diff = cv2.absdiff(prev_gray, gray)
        motion_score = np.mean(diff)
    prev_gray = gray.copy()

    # -------------------------
    # YOLO INFERENCE
    # -------------------------
    results = model(frame, conf=YOLO_CONF, verbose=False)

    detections = []
    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            detections.append((cls, conf, (x1, y1, x2, y2)))

            # draw boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            cv2.putText(frame, f"{cls}:{conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # -------------------------
    # DECISION + STABILIZATION
    # -------------------------
    status = decide_status(detections, motion_score)
    STATUS_HISTORY.append(status)

    # majority vote to stabilize
    final_status = max(set(STATUS_HISTORY), key=STATUS_HISTORY.count)

    # -------------------------
    # DISPLAY
    # -------------------------
    color = (0, 255, 0)
    if final_status == "SUSPICIOUS":
        color = (0, 255, 255)
    elif final_status == "MALPRACTICE":
        color = (0, 0, 255)

    cv2.putText(frame, f"STATUS: {final_status}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    cv2.imshow("Exam Malpractice Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()