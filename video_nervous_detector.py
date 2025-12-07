# video_nervous_detector.py
# ============================================================
# Pure algorithmic video-based nervousness detection module.
#
# This module DOES NOT directly call the Furhat API. It only
# processes frames already received from the Furhat Realtime API.
#
# ======================== HOW TO USE =========================
#
# 1) BASELINE STAGE (e.g., first 8–10 seconds of the interview)
#
#       baseline_est = VideoBaselineEstimator()
#
#       # In your websocket loop:
#       #   send:  {"type": "request.camera.start"}
#       #   receive multiple: {"type": "response.camera.data", "image": ...}
#
#       frame = decode_frame_base64(event["image"])
#       baseline_est.update(frame)
#
#       # After baseline window:
#       #   send: {"type": "request.camera.stop"}
#
#       baseline_gaze = baseline_est.finalize()
#
# 2) ANSWER SEGMENT STAGE (video during one interview answer)
#
#       seg_est = VideoNervousSegment(baseline_gaze)
#
#       # For every frame during the user answer:
#       frame = decode_frame_base64(event["image"])
#       seg_est.update(frame)
#
#       segment_score, reliability = seg_est.finalize()
#
# 3) Use segment_score in your fusion:
#
#       fused = audio_weight * audio_score + video_weight * segment_score
#
# ============================================================

from __future__ import annotations

import base64
from collections import deque
from typing import List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace

# ================= GLOBAL MODELS & CONSTANTS =================
mp_face_mesh = mp.solutions.face_mesh

# Negative emotions considered as “nervous indicators”
NEGATIVE_EMOTIONS = {"fear", "sad", "angry", "disgust"}
NEG_THRESHOLD = 40.0      # minimum confidence for emotion to count
NEG_REQUIRED = 1          # number of consecutive frames required

# Iris landmark indices used for gaze estimation
LEFT_IRIS = [469, 470, 471, 472]
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_OUTER = 33
LEFT_INNER = 133
RIGHT_OUTER = 362
RIGHT_INNER = 263


# ============================================================
# Base64 → OpenCV BGR frame
# ============================================================
def decode_frame_base64(image_b64: str) -> Optional[np.ndarray]:
    """
    Decode the "image" field (base64 JPEG) from response.camera.data
    into an OpenCV BGR frame.

    Parameters:
        image_b64 (str): Base64-encoded JPEG string.

    Returns:
        np.ndarray | None:
            Decoded BGR image, or None if decoding fails.
    """
    try:
        arr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


# ============================================================
# DeepFace emotion analysis
# ============================================================
def analyze_emotion(rgb_frame: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Run DeepFace emotion detection on an RGB frame.

    Returns:
        (emotion_name, confidence)
        emotion_name: str or None
        confidence: float in [0,100]
    """
    try:
        result = DeepFace.analyze(
            img_path=rgb_frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
        )
        if isinstance(result, list):
            result = result[0]

        emo = result.get("dominant_emotion")
        score = float(result.get("emotion", {}).get(emo, 0.0))
        return emo, score

    except Exception:
        return None, 0.0


# ============================================================
# Iris-based gaze estimation
# ============================================================
def iris_ratio(lm, which: str = "left") -> float:
    """
    Compute the relative iris horizontal position inside the eye.

    Returns:
        ratio ∈ [0,1], where ~0.5 means centered.
    """
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

    return max(0.0, min(1.0, (iris_x - ox) / width))


def analyze_gaze(frame_bgr: np.ndarray, face_mesh) -> Tuple[float, bool]:
    """
    Estimate gaze direction / centrality using iris and eye landmarks.

    Returns:
        gaze_center (float ∈ [0,1]):
            Higher → more centered gaze toward the robot.
        ok (bool):
            Whether face landmarks were detected.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return 0.0, False

    lm = results.multi_face_landmarks[0].landmark
    left = iris_ratio(lm, "left")
    right = iris_ratio(lm, "right")

    # Convert from [0,1] location to "how centered?" measure
    center_l = 1 - abs(left - 0.5) * 2
    center_r = 1 - abs(right - 0.5) * 2
    gaze_center = max(0.0, min(1.0, (center_l + center_r) / 2))
    return gaze_center, True


# ============================================================
# Baseline gaze estimator
# ============================================================
class VideoBaselineEstimator:
    """
    Accumulates multiple frames during a baseline window and computes
    the average gaze centrality (baseline_gaze ∈ [0,1]).

    Usage:
        est = VideoBaselineEstimator()
        est.update(frame_bgr)  # repeatedly
        baseline_gaze = est.finalize()
    """

    def __init__(self):
        self.gaze_vals: List[float] = []
        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
        )
        self._closed = False

    def update(self, frame_bgr: Optional[np.ndarray]):
        """Feed one baseline frame into the estimator."""
        if frame_bgr is None or self._closed:
            return

        gaze_center, ok = analyze_gaze(frame_bgr, self._face_mesh)
        if ok:
            self.gaze_vals.append(gaze_center)

    def finalize(self) -> float:
        """
        Compute baseline gaze.
        If no valid frames were detected, return a fallback (0.7).
        """
        if not self._closed:
            self._face_mesh.close()
            self._closed = True

        if not self.gaze_vals:
            return 0.7

        return float(sum(self.gaze_vals) / len(self.gaze_vals))


# ============================================================
# Nervousness detection for a single answer segment
# ============================================================
class VideoNervousSegment:
    """
    Processes video frames captured during a single interview answer
    and estimates nervousness based on:
        - Negative facial emotions (DeepFace)
        - Gaze aversion relative to baseline

    Usage:
        seg = VideoNervousSegment(baseline_gaze)
        seg.update(frame_bgr)    # for each frame
        score, reliability = seg.finalize()
    """

    def __init__(
        self,
        baseline_gaze: float,
        deepface_interval: int = 5,
    ):
        # Baseline gaze used to compute adaptive gaze threshold
        self.baseline_gaze = baseline_gaze
        self.deepface_interval = max(1, deepface_interval)

        # Histories (smoothing)
        self.emo_hist = deque(maxlen=5)
        self.gaze_hist = deque(maxlen=5)

        # Raw anxiety scores across frames
        self.frame_scores: List[float] = []
        self.frame_count: int = 0
        self.face_count: int = 0

        # Adaptive gaze threshold based on baseline
        gaze_threshold = 0.7 - 0.15 * (baseline_gaze - 0.8)
        self.gaze_threshold = max(0.3, min(0.9, gaze_threshold))

        # Internal tracking
        self.frame_id: int = 0
        self.last_emo: Tuple[Optional[str], float] = (None, 0.0)
        self.neg_frames: int = 0

        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
        )
        self._closed = False

    def update(self, frame_bgr: Optional[np.ndarray]):
        """
        Process one frame during the user's answer.
        Updates emotion & gaze indicators and builds a temporal
        nervousness measure for this segment.
        """
        if frame_bgr is None or self._closed:
            return

        self.frame_id += 1
        self.frame_count += 1

        # -------------------------------------------------------
        # EMOTION (DeepFace)
        # -------------------------------------------------------
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Only analyze every N frames for efficiency
        if self.frame_id % self.deepface_interval == 0:
            emo, emo_score = analyze_emotion(rgb)
            self.last_emo = (emo, emo_score)
        else:
            emo, emo_score = self.last_emo

        # Check negative emotion
        if emo in NEGATIVE_EMOTIONS and emo_score > NEG_THRESHOLD:
            self.neg_frames += 1
        else:
            self.neg_frames = 0

        emo_anx = int(self.neg_frames >= NEG_REQUIRED)
        self.emo_hist.append(emo_anx)
        emo_level = sum(self.emo_hist) / len(self.emo_hist) if self.emo_hist else 0.0

        # -------------------------------------------------------
        # GAZE
        # -------------------------------------------------------
        gaze_center, ok = analyze_gaze(frame_bgr, self._face_mesh)

        if ok:
            self.face_count += 1

        # Gaze_aversion = frame is considered anxious if gaze < threshold
        gaze_level = 1 if (not ok or gaze_center < self.gaze_threshold) else 0

        self.gaze_hist.append(gaze_level)
        gaze_level = sum(self.gaze_hist) / len(self.gaze_hist) if self.gaze_hist else 0.0

        # -------------------------------------------------------
        # FINAL ANXIETY SCORE FOR THE FRAME
        # Weighted combination:
        #   30% negative emotion
        #   70% gaze aversion
        # -------------------------------------------------------
        anxiety = 0.3 * emo_level + 0.7 * gaze_level
        anxiety = max(0.0, min(1.0, anxiety))

        self.frame_scores.append(anxiety)

    def finalize(self) -> Tuple[float, float]:
        """
        Produce the final nervousness score for the entire answer segment.

        Returns:
            segment_score (float ∈ [0,1]):
                Combined nervousness score over all frames.

            reliability (float ∈ [0,1]):
                Ratio of frames where a face was detected.
        """
        if not self._closed:
            self._face_mesh.close()
            self._closed = True

        if not self.frame_scores or self.frame_count == 0:
            return 0.0, 0.0

        mean_s = float(np.mean(self.frame_scores))
        peak_s = float(np.max(self.frame_scores))
        var_s = float(np.var(self.frame_scores))

        # Aggregate segment score
        segment_score = 0.3 * mean_s + 0.6 * peak_s + 0.1 * var_s
        segment_score = max(0.0, min(1.0, segment_score))

        reliability = self.face_count / self.frame_count if self.frame_count else 0.0
        return segment_score, reliability
