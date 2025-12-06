import asyncio
import json
import base64
import cv2
import numpy as np
import time
from collections import deque
import websockets
import mediapipe as mp
from deepface import DeepFace


# ================= GLOBAL MODELS =================
mp_face_mesh = mp.solutions.face_mesh

NEGATIVE_EMOTIONS = {"fear", "sad", "angry", "disgust"}
NEG_THRESHOLD = 40.0
NEG_REQUIRED = 1

# Iris landmarks
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_OUTER = 33
LEFT_INNER = 133
RIGHT_OUTER = 362
RIGHT_INNER = 263


# ===========================================================
# Decode Furhat base64 frame
# ===========================================================
def decode_frame_base64(image_b64: str):
    try:
        arr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return None


# ===========================================================
# DeepFace emotion analyzer
# ===========================================================
def analyze_emotion(rgb_frame):
    try:
        result = DeepFace.analyze(
            img_path=rgb_frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
        )
        if isinstance(result, list):
            result = result[0]
        emo = result["dominant_emotion"]
        score = float(result["emotion"].get(emo, 0.0))
        return emo, score
    except:
        return None, 0.0


# ===========================================================
# Iris gaze detection
# ===========================================================
def iris_ratio(lm, which="left"):
    if which == "left":
        outer, inner, iris = LEFT_OUTER, LEFT_INNER, LEFT_IRIS
    else:
        outer, inner, iris = RIGHT_OUTER, RIGHT_INNER, RIGHT_IRIS

    ox = lm[outer].x
    ix = lm[inner].x
    iris_x = sum(lm[i].x for i in iris) / len(iris)

    width = ix - ox
    if abs(width) < 1e-6:
        return 0.5

    return max(0, min(1, (iris_x - ox) / width))


def analyze_gaze(frame_bgr, face_mesh):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return 0.0, False

    lm = results.multi_face_landmarks[0].landmark
    left = iris_ratio(lm, "left")
    right = iris_ratio(lm, "right")

    center_l = 1 - abs(left - 0.5) * 2
    center_r = 1 - abs(right - 0.5) * 2
    gaze_center = max(0, min(1, (center_l + center_r) / 2))
    return gaze_center, True


# ===========================================================
# 1️⃣ Baseline collection
# ===========================================================
async def _baseline_async(host, duration_s=10):
    WS_URL = f"ws://{host}:9000/v1/events"
    gaze_vals = []

    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "request.camera.start"}))
        print("📷 Baseline started...")

        start = time.time()
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:
            while time.time() - start < duration_s:
                msg = await ws.recv()
                data = json.loads(msg)

                if data.get("type") != "response.camera.data":
                    continue

                frame = decode_frame_base64(data.get("image", ""))
                if frame is None:
                    continue

                gaze_center, ok = analyze_gaze(frame, fm)
                if ok:
                    gaze_vals.append(gaze_center)

        await ws.send(json.dumps({"type": "request.camera.stop"}))

    baseline = sum(gaze_vals) / len(gaze_vals) if gaze_vals else 0.7
    print(f"✔ Video baseline gaze={baseline:.3f}")
    return baseline


def run_video_baseline(host, duration_s=10):
    return asyncio.run(_baseline_async(host, duration_s))


# ===========================================================
# 2️⃣ Segment-level detection (video)
# ===========================================================
async def _segment_async(host, stop_event, baseline_gaze, max_duration_s=15):

    WS_URL = f"ws://{host}:9000/v1/events"

    emo_hist = deque(maxlen=5)
    gaze_hist = deque(maxlen=5)

    frame_scores = []
    frame_count = 0
    face_count = 0

    # adaptive gaze threshold
    gaze_threshold = 0.7 - 0.15*(baseline_gaze - 0.8)
    gaze_threshold = max(0.3, min(0.9, gaze_threshold))

    # DeepFace  Every N frames
    N = 5
    frame_id = 0
    last_emo = (None, 0.0)

    async with websockets.connect(WS_URL) as ws:

        await ws.send(json.dumps({"type": "request.camera.start"}))
        print("🎥 Video segment started...")

        start = time.time()
        neg_frames = 0

        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as fm:

            while True:
                now = time.time()

                # stopping from audio
                if stop_event.is_set():
                    print("🔚 Audio signaled stop → ending video.")
                    break

                if now - start > max_duration_s:
                    print("⏱ Timeout → ending video.")
                    break

                msg = await ws.recv()
                data = json.loads(msg)
                if data.get("type") != "response.camera.data":
                    continue

                frame = decode_frame_base64(data.get("image", ""))
                if frame is None:
                    continue

                frame_id += 1
                frame_count += 1

                # ========== EMOTION (DeepFace 每 N 帧运行一次) ==========
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if frame_id % N == 0:
                    emo, emo_score = analyze_emotion(rgb)
                    last_emo = (emo, emo_score)
                else:
                    emo, emo_score = last_emo

                if emo in NEGATIVE_EMOTIONS and emo_score > NEG_THRESHOLD:
                    neg_frames += 1
                else:
                    neg_frames = 0

                emo_anx = int(neg_frames >= NEG_REQUIRED)
                emo_hist.append(emo_anx)
                emo_level = sum(emo_hist) / len(emo_hist)

                # ========== GAZE ==========
                gaze_center, ok = analyze_gaze(frame, fm)
                if ok:
                    face_count += 1

                gaze_level = 1 if (not ok or gaze_center < gaze_threshold) else 0
                gaze_hist.append(gaze_level)
                gaze_level = sum(gaze_hist) / len(gaze_hist)

                # ========== TOTAL ANXIETY ==========
                anxiety = 0.3 * emo_level + 0.7 * gaze_level
                anxiety = max(0, min(1, anxiety))
                frame_scores.append(anxiety)

        await ws.send(json.dumps({"type": "request.camera.stop"}))

    # =================================================
    # FINAL SEGMENT SCORE
    # =================================================
    if not frame_scores:
        return 0.0, 0.0

    mean_s = float(np.mean(frame_scores))
    peak_s = float(np.max(frame_scores))
    var_s = float(np.var(frame_scores))

    segment_score = 0.3 * mean_s + 0.6 * peak_s + 0.1 * var_s
    segment_score = max(0, min(1, segment_score))

    reliability = face_count / frame_count if frame_count else 0

    print("\n📊 Video Segment Summary")
    print(f"  mean={mean_s:.3f}, peak={peak_s:.3f}, var={var_s:.3f}")
    print(f"  segment_score={segment_score:.3f}")
    print(f"  reliability={reliability:.3f}")

    return segment_score, reliability


def run_video_segment(host, stop_event, baseline_gaze, max_duration_s=15):
    return asyncio.run(
        _segment_async(host, stop_event, baseline_gaze, max_duration_s)
    )
