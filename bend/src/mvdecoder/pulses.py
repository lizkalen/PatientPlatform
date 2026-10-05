"""Pulse detection and period estimation, offline and streaming.

The trigger channel dips low once per stimulation pulse (~30.4 Hz). `estimate_trig_mid`
sets a robust threshold between the idle level and the pulse floor; `detect_pulses`
returns one pulse per high->low crossing (with a refractory) plus a fractional period
P (~67.4 samples). `stim_on_mask` marks the samples under stimulation.

`StreamPulses` is the streaming counterpart: crossings with the refractory carried
across chunks, P fixed from calibration, and a causal (post-only) stim-on mask, which
turns on about one period later than the offline mask at each bout onset.

`estimate_trig_mid` and `detect_pulses` follow movement_training.estimate_trig_mid and
removers.detect_pulses. numpy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig


def estimate_trig_mid(trig: np.ndarray, k: float = 5.0) -> float:
    """Pulse-detection threshold, anchored to the idle noise scale (MAD) rather than
    the min/max extremes. Placed `k` MAD off the resting median toward the pulses,
    floored at a fraction of the excursion depth and clamped halfway to the extreme.
    Follows movement_training.estimate_trig_mid."""
    trig = np.asarray(trig, float)
    med = float(np.nanmedian(trig))
    lo, hi = float(np.nanpercentile(trig, 1)), float(np.nanpercentile(trig, 99))
    down, up = med - lo, hi - med
    if max(down, up) < 1e-9:
        return med
    mad = float(np.nanmedian(np.abs(trig - med)))
    if down >= up:                          # pulses dip LOW: threshold below rest
        return max(med - max(k * mad, 0.3 * down), 0.5 * (med + lo))
    return min(med + max(k * mad, 0.3 * up), 0.5 * (med + hi))   # pulses go HIGH


def detect_pulses(trig: np.ndarray, refractory: int = 45, fs: float = 2048.0,
                  k: float = 5.0):
    """Offline pulse onsets (high->low crossings) with a refractory, plus the fractional
    period P (mean of the near-median inter-pulse intervals). Returns (pulses, P, thr).
    Follows removers.detect_pulses."""
    trig = np.asarray(trig, float)
    T = estimate_trig_mid(trig, k)
    below = trig < T
    onsets = np.flatnonzero((~below[:-1]) & below[1:]) + 1
    pulses, last = [], -10 ** 9
    for t in onsets:
        if t - last >= refractory:
            pulses.append(int(t)); last = int(t)
    pulses = np.asarray(pulses, int)
    d = np.diff(pulses)
    if d.size:
        med = np.median(d)
        near = d[np.abs(d - med) <= 2]          # keep only regular (non-missed) intervals
        P = float(near.mean()) if near.size else float(med)
    else:
        P = fs / 30.0
    return pulses, P, float(T)


def period_from_pulses(pulses: np.ndarray, fs: float = 2048.0) -> float:
    """Sub-sample period from an existing pulse-index array (near-median mean), for the
    no-trigger PulseTimesStimSource when P is not supplied."""
    pulses = np.asarray(pulses, int)
    d = np.diff(pulses)
    if not d.size:
        return fs / 30.0
    med = np.median(d)
    near = d[np.abs(d - med) <= 2]
    return float(near.mean()) if near.size else float(med)


def stim_on_mask(n: int, pulses: np.ndarray, P: float,
                 pre: float = 1.0, post: float = 2.0) -> np.ndarray:
    """Boolean [n], True where stimulation is running (guarded by `pre`/`post` periods
    around each pulse). Non-causal (marks `pre` periods before each pulse); used for
    offline gating. Follows removers.stim_on_mask."""
    m = np.zeros(n, bool)
    g1, g2 = int(round(pre * P)), int(round(post * P))
    for t in np.asarray(pulses, int):
        m[max(0, t - g1):min(n, t + g2)] = True
    return m


class StreamPulses:
    """Streaming pulse detector with a causal stim-on mask.

    `push(trig_chunk, s0)` detects new pulses in a chunk (absolute base s0), carrying
    the below-threshold state and refractory across chunk boundaries; it returns the
    new pulse indices and appends them to `hist`. P is fixed at construction, not
    re-estimated.

    `on_block(s0, n)` marks samples t where some past pulse p satisfies
    p <= t <= p + round(post*P). Post-only, so it is causal.
    """

    def __init__(self, thr: float, P: float, refractory: int = 45, post: float = 2.0):
        self.thr = float(thr)
        self.P = float(P)
        self.ref = int(refractory)
        self.post = float(post)
        self.prev_below = False
        self.last = -10 ** 9
        self.hist: list[int] = []          # every detected pulse still able to reach a future sample

    def push(self, trig: np.ndarray, s0: int) -> list[int]:
        trig = np.asarray(trig, float)
        below = trig < self.thr
        sh = np.empty(len(trig), bool)
        sh[0] = self.prev_below
        sh[1:] = below[:-1]
        out = []
        for i in np.flatnonzero(below & ~sh):
            t = s0 + int(i)
            if t - self.last >= self.ref:
                self.last = t
                self.hist.append(t)
                out.append(t)
        self.prev_below = bool(below[-1])
        return out

    def on_block(self, s0: int, n: int) -> np.ndarray:
        on = np.zeros(n, bool)
        g2 = int(round(self.post * self.P))
        while self.hist and self.hist[0] + g2 < s0:      # drop pulses too old to reach s0
            self.hist.pop(0)
        for p in self.hist:
            if p > s0 + n - 1:
                break
            a = max(p - s0, 0)
            b = min(p - s0 + g2 + 1, n)
            if a < b:
                on[a:b] = True
        return on
