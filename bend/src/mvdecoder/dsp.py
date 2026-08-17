"""Causal filtering: filter design and a stateful, chunk-agnostic filter.

`design_filters` builds a stacked 50 Hz-harmonic notch and a 4th-order 20-450 Hz
Butterworth bandpass, both as SOS (same design as movement_engine.design_filters).
`CausalFilter` wraps scipy.signal.sosfilt and carries the per-channel filter state
between calls, so a signal fed in one chunk or many gives the same output and no
sample depends on future data. Uses sosfilt, not sosfiltfilt (which is non-causal).

numpy/scipy only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt

from .config import DecoderConfig


def design_filters(cfg: DecoderConfig):
    """Return (bp_sos, notch_sos): a 4th-order Butterworth bandpass and an iirnotch at
    every `notch_base` harmonic below Nyquist, each as SOS and stacked."""
    bp_sos = butter(4, [cfg.bp_lo, cfg.bp_hi], btype="band", fs=cfg.fs, output="sos")
    notch = [tf2sos(*iirnotch(w0=f0, Q=cfg.notch_q, fs=cfg.fs))
             for f0 in np.arange(cfg.notch_base, cfg.fs / 2, cfg.notch_base)]
    return bp_sos, np.vstack(notch)


class CausalFilter:
    """Stateful per-channel causal SOS filter. `apply(x)` filters x[C, n] in place of
    a running stream, carrying the filter memory `zi` between calls. The number of
    channels is fixed at construction; the chunk length may vary."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = np.asarray(sos, float)
        self.nch = int(n_channels)
        self.zi = np.zeros((self.nch, self.sos.shape[0], 2))

    def reset(self):
        self.zi = np.zeros((self.nch, self.sos.shape[0], 2))

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Filter one chunk x[C, n]; returns the filtered chunk, updates state."""
        x = np.asarray(x, float)
        out = np.empty_like(x)
        for c in range(x.shape[0]):
            out[c], self.zi[c] = sosfilt(self.sos, x[c], zi=self.zi[c])
        return out
