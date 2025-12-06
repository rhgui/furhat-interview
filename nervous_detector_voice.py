# nervous_detector_voice.py
# ------------------------------------------------------------
# Pure audio/text nervousness detector
#
# ============================================================
# NOTE ON CORRECT AUDIO SEGMENT (Furhat Realtime API integration)
# ============================================================
# For this module to interpret speaking duration correctly, the audio you pass
# in MUST cover only the user’s actual speech, NOT the silent reaction time
# after the robot’s question.
#
# Recommended pattern on the Furhat side:
#
#   1) In on_speak_end (when the robot finishes asking the question):
#
#        await furhat.request_audio_start(
#            sample_rate=16000,
#            microphone=True,
#            speaker=False,
#        )
#        self.is_audio_recording = False   # start streaming, but DO NOT write to buffer yet
#
#   2) In on_hear_start (when the user starts speaking for the first time):
#
#        self.t_speech_start = time.time()  # for reaction_time_s
#        self.is_audio_recording = True     # from this moment, append audio frames to buffer
#
#   3) In on_audio_data:
#
#        if self.is_audio_recording:
#            # append microphone PCM16LE bytes to current_answer_audio
#
# With this pattern:
#   - The PCM segment passed into nervous_detector_voice only contains
#     “user speaking from first word to end of answer”.
#   - af["duration_s"] then matches your intuitive “speaking time”, without
#     reaction-time silence at the beginning.
#   - The expected_answer_time_s rule truly compares “spoken length vs. expected
#     answer length” instead of “(reaction time + speech) vs. expected”, which
#     makes the nervousness judgement more meaningful.

#
# ============================================================
# HOW TO USE – STEP BY STEP (FOR FURHAT + EXPRESSION FUSION)
# ============================================================
#
# This module only does AUDIO + TEXT nervousness scoring.
# You integrate it with:
#   - Furhat Realtime API (for ASR / audio stream)
#   - Your facial-expression module (for visual nervousness)
#
# ------------------------------------------------------------
# 0) IMPORT & INITIAL SETUP (Python side)
# ------------------------------------------------------------
#
# from nervous_detector_voice import (
#     NervousBaseline,
#     FillerCounter,
#     build_baseline_on_first_answer,
#     finalize_nervous_answer,
# )
#
# # Create ONE filler counter object and reuse:
# filler_counter = FillerCounter()
#
# # Keep the baseline somewhere (global / session state):
# nervous_baseline: NervousBaseline | None = None
#
#
# ------------------------------------------------------------
# 1) WHEN YOU ASK THE VERY FIRST QUESTION (BUILD BASELINE)
# ------------------------------------------------------------
#
# Example: robot asks: "Can you introduce yourself?"
#
# 1.1 Start listening with Furhat Realtime API:
#
#     await furhat.request_listen_config(
#         languages=["en-US"],
#         phrases=["um", "uh", "er", "ah", "eh"],   # help ASR keep fillers
#     )
#
#     filler_counter.reset()
#
#     await furhat.request_listen_start(
#         partial=True,    # VERY IMPORTANT: we want partials
#         concat=True,     # full utterance as one string
#         # ... other options ...
#     )
#
# 1.2 While listening, for each response.hear.partial:
#
#     # partial_text is the current partial hypothesis from Furhat
#     filler_counter.on_partial(partial_text)
#
#     # DO NOT call nervous detector here; just accumulate fillers.
#
# 1.3 When you detect the FIRST answer is finished:
#
#     - You should have:
#         * audio_bytes (PCM16LE mono) for the whole answer
#         * sample_rate (e.g., 16000)
#         * final_text: final ASR result (from hear.end)
#
#     Then build the baseline:
#
#     nervous_baseline = build_baseline_on_first_answer(
#         audio_bytes,
#         sample_rate,
#         final_text,
#     )
#
#     # For the first answer, you usually do NOT use nervous score yet.
#     # This is considered the "neutral reference" for this user.
#
#
# ------------------------------------------------------------
# 2) FOR EACH SUBSEQUENT QUESTION (SECOND, THIRD, ...)
# ------------------------------------------------------------
#
# Example: robot asks: "Why are you interested in this role?"
#
# 2.1 Ask the question normally. When the robot finishes speaking,
#     record the timestamp, e.g.:
#
#     t_question_end = now()
#
# 2.2 Start listening again:
#
#     await furhat.request_listen_config(
#         languages=["en-US"],
#         phrases=["um", "uh", "er", "ah", "eh"],
#     )
#
#     filler_counter.reset()
#
#     await furhat.request_listen_start(
#         partial=True,
#         concat=True,
#         # ... other options ...
#     )
#
# 2.3 While listening, for each response.hear.partial:
#
#     filler_counter.on_partial(partial_text)
#
#     # (No scoring yet; we just track added fillers in streaming fashion.)
#
# 2.4 Detect when the user STARTS SPEAKING:
#
#     - When you receive the FIRST non-empty partial or first audio frame
#       that passes a simple energy threshold, treat that as "speech start":
#
#       t_speech_start = now()
#       reaction_time_s = t_speech_start - t_question_end
#
#     - Save reaction_time_s; you'll pass it into finalize_nervous_answer().
#
# 2.5 Detect when the user FINISHES the answer (end of current turn):
#
#     - At that moment you should have:
#         * audio_bytes: PCM16LE mono for this answer
#         * sample_rate: e.g. 16000
#         * final_text: final ASR transcript (from hear.end)
#         * reaction_time_s: computed above
#         * extra_filler_count: filler_counter.count
#
#     - Also choose an expected answer duration for this question, e.g.:
#
#         expected_answer_time_s = 8.0   # "typical" length in seconds
#
#     - Then call:
#
#         result = finalize_nervous_answer(
#             baseline=nervous_baseline,
#             audio_pcm16le=audio_bytes,
#             sample_rate=sample_rate,
#             text=final_text,
#             expected_answer_time_s=expected_answer_time_s,
#             reaction_time_s=reaction_time_s,
#             extra_filler_count=filler_counter.count,
#         )
#
#         nervous_audio   = result["nervous_score"]   # float in [0.0, 1.0]
#         stop_expression = result["stop_expression"] # always True here
#
# 2.6 Tell your facial-expression module to STOP:
#
#     - When you get stop_expression == True, send a message / flag to your
#       face-expression pipeline:
#
#         if stop_expression:
#             expression_module.stop_current_estimation()
#
#     - Up to this moment, the face-expression module has been running
#       continuously on the camera feed during the answer.
#
#
# ------------------------------------------------------------
# 3) FUSING AUDIO NERVOUSNESS + FACIAL NERVOUSNESS
# ------------------------------------------------------------
#
# Assume the face module also gives a nervousness score in [0, 1], e.g.:
#
#     nervous_face = expression_module.current_nervous_score  # 0..1
#
# You can combine them with a simple weighted average:
#
#     fused_nervous = 0.6 * nervous_audio + 0.4 * nervous_face
#
# Then you can threshold / discretize if needed, e.g.:
#
#     if fused_nervous >= 0.70:
#         label = "NERVOUS"
#     elif fused_nervous <= 0.30:
#         label = "RELAXED"
#     else:
#         label = "NEUTRAL"
#
#
# ------------------------------------------------------------
# 4) SUMMARY OF PUBLIC FUNCTIONS / CLASSES
# ------------------------------------------------------------
#
#   - class FillerCounter:
#       * on_partial(text: str) -> None
#       * reset() -> None
#       * .count: int  (total fillers seen in streaming partials)
#
#   - build_baseline_on_first_answer(audio_pcm16le, sample_rate, text)
#       -> NervousBaseline
#
#   - is_user_nervous_on_answer(
#         baseline,
#         audio_pcm16le,
#         sample_rate,
#         text,
#         expected_answer_time_s=None,
#         reaction_time_s=None,
#         extra_filler_count=0,
#     ) -> float nervous_score in [0, 1]
#
#   - finalize_nervous_answer(
#         baseline,
#         audio_pcm16le,
#         sample_rate,
#         text,
#         expected_answer_time_s=None,
#         reaction_time_s=None,
#         extra_filler_count=0,
#     ) -> dict:
#         {
#            "nervous_score": float in [0, 1],
#            "stop_expression": True
#         }
#
# ============================================================
# END OF USAGE GUIDE – IMPLEMENTATION BELOW
# ============================================================

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


# ============================================================
# Data structures
# ============================================================

@dataclass
class NervousBaseline:
    sr: int

    # audio baseline
    silence_ratio: float
    pause_rate: float           # pauses per second (pause >= min_pause_s)
    mean_pause_s: float
    max_pause_s: float
    energy_cv: float            # std(rms) / mean(rms), scale-invariant
    loge_std: float             # std(20*log10(rms)) in dB
    rms_db: float
    duration_s: float

    # text baseline
    wps: float
    filler_rate: float
    repeat_rate: float
    hedge_rate: float
    repair_rate: float


# ============================================================
# Text features (English) + streaming filler counter
# ============================================================

_word_re = re.compile(r"[a-z']+")

_FILLERS = [
    "um", "uh", "er", "ah", "eh", "em",
    "like", "you know", "i mean", "sort of", "kind of",
]
_HEDGES = [
    "maybe", "probably", "perhaps", "i guess", "i think",
    "not sure", "to be honest", "kind of", "sort of",
]
_REPAIRS = [
    "sorry", "actually", "i mean", "no wait", "wait",
    "let me rephrase", "what i meant", "rather",
]

# For streaming partial ASR: only short pure fillers
FILLER_SET = {"um", "uh", "er", "ah", "eh"}


class FillerCounter:
    """
    Incremental, "safe" filler counter for partial ASR hypotheses.
    """
    def __init__(self):
        self._last_partial = ""
        self.count = 0

    def reset(self):
        self._last_partial = ""
        self.count = 0

    def on_partial(self, text: str):
        t = (text or "").lower()

        # Find longest common prefix between previous and current partial.
        # Only the "new tail" is scanned and counted, so edits in the front
        # do NOT cause double-counting.
        i = 0
        m = min(len(self._last_partial), len(t))
        while i < m and self._last_partial[i] == t[i]:
            i += 1
        added = t[i:]  # only newly added tail

        # Tokenize new tail and count fillers
        toks = _word_re.findall(added)
        self.count += sum(1 for w in toks if w in FILLER_SET)

        self._last_partial = t


def _tokenize(text: str) -> list[str]:
    return _word_re.findall((text or "").lower())

def _count_phrases(text_lower: str, phrases: list[str]) -> int:
    c = 0
    for p in phrases:
        c += len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
    return c

def _text_features(text: str, duration_s: float) -> Dict[str, float]:
    t = (text or "").strip()
    tl = t.lower()
    toks = _tokenize(t)
    n = len(toks)

    fillers = _count_phrases(tl, _FILLERS)
    hedges = _count_phrases(tl, _HEDGES)
    repairs = _count_phrases(tl, _REPAIRS)

    # immediate repeats: "I I", "the the"
    repeats = 0
    for i in range(1, n):
        if toks[i] == toks[i - 1]:
            repeats += 1

    wps = (n / duration_s) if duration_s > 1e-6 else 0.0
    denom = max(n, 1)

    return {
        "words": float(n),
        "wps": float(wps),
        "filler_rate": float(fillers / denom),
        "repeat_rate": float(repeats / denom),
        "hedge_rate": float(hedges / denom),
        "repair_rate": float(repairs / denom),
    }


# ============================================================
# Audio features (PCM16LE mono)
# ============================================================

def _pcm16le_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.zeros(0, dtype=np.float32)
    x = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return np.clip(x, -1.0, 1.0)

def _rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    return 20.0 * math.log10(rms)

def _rms_frames(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if x.size < frame:
        return np.array([float(np.sqrt(np.mean(x * x) + 1e-12))], dtype=np.float32)
    n = 1 + (x.size - frame) // hop
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = x[i * hop: i * hop + frame]
        out[i] = float(np.sqrt(np.mean(seg * seg) + 1e-12))
    return out

def _pause_stats(silent: np.ndarray, hop_s: float, min_pause_s: float) -> Tuple[int, float, float]:
    if silent.size == 0:
        return 0, 0.0, 0.0
    min_len = max(1, int(math.ceil(min_pause_s / hop_s)))

    count = 0
    total = 0.0
    maxp = 0.0

    run = 0
    for s in silent:
        if s:
            run += 1
        else:
            if run >= min_len:
                dur = run * hop_s
                count += 1
                total += dur
                maxp = max(maxp, dur)
            run = 0

    if run >= min_len:
        dur = run * hop_s
        count += 1
        total += dur
        maxp = max(maxp, dur)

    meanp = (total / count) if count > 0 else 0.0
    return count, float(meanp), float(maxp)

def _audio_features(
    audio_pcm16le: bytes,
    sr: int,
    silence_mult: float = 2.5,
    min_pause_s: float = 0.20,
) -> Dict[str, float]:
    x = _pcm16le_bytes_to_float32(audio_pcm16le)
    dur = x.size / float(sr) if sr > 0 else 0.0
    if x.size == 0 or dur <= 1e-6:
        return {
            "duration_s": 0.0,
            "silence_ratio": 1.0,
            "pause_rate": 0.0,
            "mean_pause_s": 0.0,
            "max_pause_s": 0.0,
            "energy_cv": 0.0,
            "loge_std": 0.0,
            "rms_db": -120.0,
        }

    frame = max(1, int(0.02 * sr))  # 20ms
    hop   = max(1, int(0.01 * sr))  # 10ms
    hop_s = hop / float(sr)

    rms = _rms_frames(x, frame, hop)
    noise_floor = float(np.percentile(rms, 10))
    thr = max(1e-4, noise_floor * float(silence_mult))
    silent = (rms < thr)

    silence_ratio = float(np.mean(silent))

    pause_count, mean_pause_s, max_pause_s = _pause_stats(silent, hop_s=hop_s, min_pause_s=min_pause_s)
    pause_rate = float(pause_count / max(dur, 1e-6))

    rms_mean = float(np.mean(rms) + 1e-12)
    rms_std = float(np.std(rms))
    energy_cv = float(rms_std / rms_mean)

    loge = 20.0 * np.log10(rms + 1e-12)
    loge_std = float(np.std(loge))

    return {
        "duration_s": float(dur),
        "silence_ratio": silence_ratio,
        "pause_rate": pause_rate,
        "mean_pause_s": mean_pause_s,
        "max_pause_s": max_pause_s,
        "energy_cv": energy_cv,
        "loge_std": loge_std,
        "rms_db": _rms_db(x),
    }


# ============================================================
# Scoring helpers
# ============================================================

def _safe_z(x: float, mu: float, sig: float) -> float:
    sig = max(sig, 1e-4)
    z = (x - mu) / sig
    return float(np.clip(z, -4.0, 4.0))

def _sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))

def _pos_evidence(z: float) -> float:
    """Only treat increases (z>0) as evidence."""
    return _sigmoid(max(0.0, z))

def _abs_change_evidence(z: float) -> float:
    """Treat big changes in either direction as evidence."""
    return _sigmoid(abs(z))


# ============================================================
# ✅ Public API: two core functions + one high-level helper
# ============================================================

def build_baseline_on_first_answer(audio_pcm16le: bytes, sample_rate: int, text: str) -> NervousBaseline:
    """
    Build a per-user neutral baseline from their FIRST answer.
    """
    af = _audio_features(audio_pcm16le, sample_rate)
    tf = _text_features(text, max(af["duration_s"], 1e-6))

    return NervousBaseline(
        sr=sample_rate,

        silence_ratio=af["silence_ratio"],
        pause_rate=af["pause_rate"],
        mean_pause_s=af["mean_pause_s"],
        max_pause_s=af["max_pause_s"],
        energy_cv=af["energy_cv"],
        loge_std=af["loge_std"],
        rms_db=af["rms_db"],
        duration_s=af["duration_s"],

        wps=tf["wps"],
        filler_rate=tf["filler_rate"],
        repeat_rate=tf["repeat_rate"],
        hedge_rate=tf["hedge_rate"],
        repair_rate=tf["repair_rate"],
    )


def is_user_nervous_on_answer(
    baseline: NervousBaseline,
    audio_pcm16le: bytes,
    sample_rate: int,
    text: str,
    expected_answer_time_s: Optional[float] = None,
    reaction_time_s: Optional[float] = None,
    extra_filler_count: int = 0,
) -> float:
    """
    Estimate nervousness for a single answer, given a baseline.

    Returns:
        nervous_score in [0.0, 1.0]
    """
    af = _audio_features(audio_pcm16le, sample_rate)
    tf = _text_features(text, max(af["duration_s"], 1e-6))

    # Integrate extra fillers from partial ASR, if provided
    if extra_filler_count > 0 and tf["words"] > 0:
        base_fillers = tf["filler_rate"] * tf["words"]
        total_fillers = base_fillers + float(extra_filler_count)
        tf["filler_rate"] = float(total_fillers / tf["words"])

    # Conservative sigma floors (baseline from one utterance)
    SIG = {
        # audio
        "silence_ratio": 0.08,
        "pause_rate": 0.10,      # pauses/sec
        "mean_pause_s": 0.18,
        "max_pause_s": 0.35,
        "energy_cv": 0.18,
        "loge_std": 2.5,         # dB std
        "rms_db": 4.0,

        # text
        "wps": 0.9,
        "filler_rate": 0.05,
        "repeat_rate": 0.03,
        "hedge_rate": 0.05,
        "repair_rate": 0.03,
    }

    # z-scores vs baseline
    z_sil   = _safe_z(af["silence_ratio"], baseline.silence_ratio, SIG["silence_ratio"])
    z_pr    = _safe_z(af["pause_rate"],    baseline.pause_rate,    SIG["pause_rate"])
    z_mps   = _safe_z(af["mean_pause_s"],  baseline.mean_pause_s,  SIG["mean_pause_s"])
    z_xps   = _safe_z(af["max_pause_s"],   baseline.max_pause_s,   SIG["max_pause_s"])
    z_ecv   = _safe_z(af["energy_cv"],     baseline.energy_cv,     SIG["energy_cv"])
    z_loge  = _safe_z(af["loge_std"],      baseline.loge_std,      SIG["loge_std"])
    z_loud  = _safe_z(af["rms_db"],        baseline.rms_db,        SIG["rms_db"])

    z_wps   = _safe_z(tf["wps"],           baseline.wps,           SIG["wps"])
    z_fill  = _safe_z(tf["filler_rate"],   baseline.filler_rate,   SIG["filler_rate"])
    z_rep   = _safe_z(tf["repeat_rate"],   baseline.repeat_rate,   SIG["repeat_rate"])
    z_hedge = _safe_z(tf["hedge_rate"],    baseline.hedge_rate,    SIG["hedge_rate"])
    z_repai = _safe_z(tf["repair_rate"],   baseline.repair_rate,   SIG["repair_rate"])

    # evidence in [0, 1]
    e_sil  = _pos_evidence(z_sil)
    e_pr   = _pos_evidence(z_pr)
    e_mps  = _pos_evidence(z_mps)
    e_xps  = _pos_evidence(z_xps)

    e_ecv  = _pos_evidence(z_ecv)
    e_loge = _pos_evidence(z_loge)

    e_loud = _abs_change_evidence(z_loud)  # loudness change either direction
    e_wps  = _abs_change_evidence(z_wps)   # speaking rate change either direction

    e_text = float(np.mean([
        _pos_evidence(z_fill),
        _pos_evidence(z_rep),
        _pos_evidence(z_hedge),
        _pos_evidence(z_repai),
    ]))

    # --- Time-group evidence: answer duration & reaction time ----

    # 1) Answer duration vs expected
    time_short_e = 0.0
    if expected_answer_time_s is not None and expected_answer_time_s > 0.0:
        ratio = af["duration_s"] / float(expected_answer_time_s)  # ~1.0 -> as expected
        if ratio < 0.3:
            time_short_e = 0.9
        elif ratio < 0.5:
            time_short_e = 0.7
        elif ratio < 0.7:
            time_short_e = 0.4
        else:
            time_short_e = 0.0  # long / normal answer -> no extra evidence

    # 2) Reaction time before starting to speak
    time_rt_e = 0.0
    if reaction_time_s is not None and reaction_time_s > 0.0:
        # ~0.5s -> ~0 evidence; ~1.5s -> moderate; >=3s -> strong evidence
        rt_z = max(0.0, (reaction_time_s - 0.8) / 1.2)
        time_rt_e = _sigmoid(rt_z)

    time_components = []
    if time_short_e > 0.0:
        time_components.append(time_short_e)
    if time_rt_e > 0.0:
        time_components.append(time_rt_e)
    time_group = float(np.mean(time_components)) if time_components else 0.0

    # group scores
    pause_group  = float(0.38 * e_sil + 0.32 * e_pr + 0.18 * e_mps + 0.12 * e_xps)
    instab_group = float(0.40 * e_ecv + 0.30 * e_loge + 0.15 * e_loud + 0.15 * e_wps)
    text_group   = e_text

    # fuse all groups, including time_group (weights sum to 1.0)
    score = float(
        0.35 * pause_group +
        0.30 * instab_group +
        0.20 * text_group +
        0.15 * time_group
    )

    return float(np.clip(score, 0.0, 1.0))


def finalize_nervous_answer(
    baseline: NervousBaseline,
    audio_pcm16le: bytes,
    sample_rate: int,
    text: str,
    expected_answer_time_s: Optional[float] = None,
    reaction_time_s: Optional[float] = None,
    extra_filler_count: int = 0,
) -> Dict[str, float]:
    """
    High-level helper to be called ONCE when you decide the user's
    answer is finished.
    """
    score = is_user_nervous_on_answer(
        baseline,
        audio_pcm16le,
        sample_rate,
        text,
        expected_answer_time_s=expected_answer_time_s,
        reaction_time_s=reaction_time_s,
        extra_filler_count=extra_filler_count,
    )
    return {
        "nervous_score": score,
        "stop_expression": True,
    }
