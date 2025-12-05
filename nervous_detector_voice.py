# nervous_detector_voice.py
# ------------------------------------------------------------
#
# - Use audio fluency cues as the main signal:
#     silence_ratio + pause stats + energy instability (scale-invariant) + loudness/speaking-rate change
# - Text cues are secondary support only (hedges/repairs/repeats/etc.).
# - Expected answer time rule:
#     if actual_duration < 50% * expected_answer_time_s -> nervous=True
#     else expected time is neutral.
#
# Public API:
#   - build_baseline_on_first_answer(audio_pcm16le, sample_rate, text) -> NervousBaseline
#   - is_user_nervous_on_answer(baseline, audio_pcm16le, sample_rate, text, expected_answer_time_s=None) -> bool
#
# Audio expected:
#   mono PCM16LE bytes (raw). Recommended 16kHz.

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

    # text baseline (kept, but used lightly)
    wps: float
    filler_rate: float
    repeat_rate: float
    hedge_rate: float
    repair_rate: float


# ============================================================
# Text features (English) - supportive only
# ============================================================

_word_re = re.compile(r"[a-z']+")

# Note: um/uh/er may be dropped by ASR; we keep them but do NOT rely on them heavily.
_FILLERS = [
    "um", "uh", "er", "ah", "eh",
    "like", "you know", "i mean", "sort of", "kind of",
]
_HEDGES = [
    "maybe", "probably", "perhaps", "i guess", "i think",
    "not sure", "to be honest", "kind of", "sort of",
    "i don't know", "im not sure", "i'm not sure",
]
_REPAIRS = [
    "sorry", "actually", "i mean", "no wait", "wait",
    "let me rephrase", "what i meant", "rather",
]

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
# Public API: two functions
# ============================================================

def build_baseline_on_first_answer(audio_pcm16le: bytes, sample_rate: int, text: str) -> NervousBaseline:
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
) -> bool:
    """
    Expected-answer-time rule:
      - if duration < 50% expected -> nervous=True
      - else time treated as neutral
    Then audio-dominant nervousness detection vs baseline.
    Text is only supportive.
    """
    af = _audio_features(audio_pcm16le, sample_rate)
    tf = _text_features(text, max(af["duration_s"], 1e-6))

    # --- strict expected-time rule (your current policy) ---
    if expected_answer_time_s is not None and expected_answer_time_s > 0:
        ratio = af["duration_s"] / float(expected_answer_time_s)
        if ratio < 0.50:
            return True

    # Conservative sigma floors (baseline is one utterance)
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

    # text evidence (supportive only)
    e_text = float(np.mean([
        _pos_evidence(z_fill),
        _pos_evidence(z_rep),
        _pos_evidence(z_hedge),
        _pos_evidence(z_repai),
    ]))

    # group scores
    pause_group  = float(0.38 * e_sil + 0.32 * e_pr + 0.18 * e_mps + 0.12 * e_xps)
    instab_group = float(0.40 * e_ecv + 0.30 * e_loge + 0.15 * e_loud + 0.15 * e_wps)
    text_group   = e_text

    # Audio-dominant fusion (text is minor)
    score = float(
        0.56 * pause_group +
        0.36 * instab_group +
        0.08 * text_group
    )
    threshold = 0.62

    # Must have at least one AUDIO group meaningfully high
    audio_hit = (pause_group >= 0.62) or (instab_group >= 0.62)
    if not audio_hit:
        return False

    # More robust gating:
    both_audio = (pause_group >= 0.62 and instab_group >= 0.62)
    strong_one = (max(pause_group, instab_group) >= 0.80)
    text_support = (text_group >= 0.65)

    return bool((score >= threshold and both_audio) or (score >= threshold and strong_one and text_support))
