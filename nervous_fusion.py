# ============================================================
#  Multi-Modal Nervousness Fusion Module
#
#  This module combines:
#      (1) Audio nervousness detection
#      (2) Video nervousness detection
#
#  The main program should call only two APIs:
#
#      1) build_baseline(...)
#         → Run once after the first “relaxed” answer
#           (build both audio baseline + video gaze baseline)
#
#      2) evaluate_answer(...)
#         → Run after each interview answer
#           Returns: (is_nervous: bool, scores: FusionScores)
#
#  No Furhat API is called here. This module is purely algorithmic.
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

# ==== Audio nervousness detection ====
# We use:
#   - build_baseline_on_first_answer(...)
#   - is_user_nervous_on_answer(...)  → returns float nervous_score in [0,1]
from nervous_detector_voice import (
    build_baseline_on_first_answer,
    is_user_nervous_on_answer,
    NervousBaseline,
)

# ==== Video nervousness detection ====
from video_nervous_detector import (
    VideoBaselineEstimator,
    VideoNervousSegment,
)


# ===================================================================
# Configuration for the fusion algorithm
# ===================================================================
@dataclass
class FusionConfig:
    """
    Configuration for the nervousness fusion.

    Parameters:
        nervous_threshold:
            Final fused score above this threshold is considered nervous.

        audio_weight / video_weight:
            Relative weights for audio and video scores.
            (Automatically normalized so audio+video=1)
    """

    nervous_threshold: float = 0.60     # Final decision threshold
    audio_weight: float = 0.7          # Audio is dominant
    video_weight: float = 0.3          # Video contributes less

    def normalize(self):
        """Ensure weights sum to 1."""
        total = self.audio_weight + self.video_weight
        if total <= 0:
            self.audio_weight = 0.5
            self.video_weight = 0.5
        else:
            self.audio_weight /= total
            self.video_weight /= total


# ===================================================================
# Structure to store detailed scores (useful for logs and debugging)
# ===================================================================
@dataclass
class FusionScores:
    # Audio nervousness score in [0,1]
    audio_score: float

    # Video nervousness score in [0,1]
    video_score: float

    # Weighted fusion score in [0,1]
    fused_score: float

    # Reliability of video (ratio of frames where a face is detected)
    video_reliability: float


# ===================================================================
#                  Main Fusion Class (User API)
# ===================================================================
class NervousFusion:

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()
        self.config.normalize()

        # Baselines computed after first answer
        self.audio_baseline: Optional[NervousBaseline] = None
        self.video_baseline_gaze: Optional[float] = None

        # Last evaluation detail
        self.last_scores: Optional[FusionScores] = None

    # ---------------------------------------------------------------
    # Build baseline from the first relaxed answer
    # ---------------------------------------------------------------
    def build_baseline(
        self,
        first_audio_bytes: bytes,
        sample_rate: int,
        first_text: str,
        baseline_frames: List[np.ndarray],
    ):
        """
        Build both audio and video baselines.
        Should be called only once.

        Parameters:
            first_audio_bytes:
                PCM16LE audio of the first answer.

            sample_rate:
                Audio sampling rate (e.g., 16000).

            first_text:
                ASR final text of the first answer.

            baseline_frames:
                List of video frames collected during the baseline window,
                e.g., 8–10 seconds of relaxed talking.
        """

        # ---- Audio baseline ----
        self.audio_baseline = build_baseline_on_first_answer(
            audio_pcm16le=first_audio_bytes,
            sample_rate=sample_rate,
            text=first_text,
        )

        # ---- Video baseline ----
        vb = VideoBaselineEstimator()
        for frame in baseline_frames:
            vb.update(frame)
        self.video_baseline_gaze = vb.finalize()

        print(f"[Fusion] Baseline built. video_gaze={self.video_baseline_gaze:.3f}")

    def _check_ready(self):
        """Ensure baseline is built before evaluation."""
        if self.audio_baseline is None or self.video_baseline_gaze is None:
            raise RuntimeError("Fusion baseline not built. Call build_baseline() first.")

    # ---------------------------------------------------------------
    # Evaluate one interview answer → return nervous/not nervous
    # ---------------------------------------------------------------
    def evaluate_answer(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        text: str,
        reaction_time_s: float,
        expected_answer_time_s: float,
        extra_filler_count: int,
        segment_frames: List[np.ndarray],
    ) -> Tuple[bool, FusionScores]:
        """
        Evaluate nervousness for a single interview answer.

        Parameters:
            audio_bytes:
                PCM16LE audio for this answer.

            sample_rate:
                Audio sampling rate (e.g., 16000).

            text:
                ASR final text for this answer.

            reaction_time_s:
                Time from Furhat finishing the question → user starts speaking.
                (Currently not used directly in the fusion logic, but kept
                 in the function signature for future extensions.)

            expected_answer_time_s:
                Expected speaking duration for this question.
                Passed to the audio module.

            extra_filler_count:
                Number of filler words ("um", "uh", etc.) detected during partial ASR.
                Currently unused here but kept for extension.

            segment_frames:
                Video frames collected during the user's answer.

        Returns:
            is_nervous: bool
                True if the fused nervousness score >= threshold

            scores: FusionScores
                Contains audio_score, video_score, fused_score, reliability
        """

        self._check_ready()

        # ===========================================================
        #   1) Audio nervousness score in [0,1]
        # ===========================================================
        # is_user_nervous_on_answer already returns a float nervous_score ∈ [0,1]
        audio_score = is_user_nervous_on_answer(
            baseline=self.audio_baseline,
            audio_pcm16le=audio_bytes,
            sample_rate=sample_rate,
            text=text,
            expected_answer_time_s=expected_answer_time_s,
            reaction_time_s=reaction_time_s,
            extra_filler_count=extra_filler_count,
        )
        # Safety clip
        audio_score = float(np.clip(audio_score, 0.0, 1.0))

        # ===========================================================
        #   2) Video nervousness score
        # ===========================================================
        seg = VideoNervousSegment(
            baseline_gaze=self.video_baseline_gaze
        )
        for frame in segment_frames:
            seg.update(frame)

        video_score, reliability = seg.finalize()

        # ===========================================================
        #   3) Weighted fusion (audio-dominant)
        # ===========================================================
        A = self.config.audio_weight   # normalized
        V = self.config.video_weight   # normalized

        fused_score = A * audio_score + V * video_score

        print(
            "[Fusion] audio_raw={:.3f}, video={:.3f}, fused={:.3f}, rel={:.3f}".format(
                audio_score, video_score, fused_score, reliability
            )
        )

        scores = FusionScores(
            audio_score=audio_score,
            video_score=video_score,
            fused_score=fused_score,
            video_reliability=reliability,
        )
        self.last_scores = scores

        # ===========================================================
        #   4) Final binary decision
        # ===========================================================
        is_nervous = fused_score >= self.config.nervous_threshold

        return is_nervous, scores
