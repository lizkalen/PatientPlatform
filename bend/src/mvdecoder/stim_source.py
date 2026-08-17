"""Stim-condition and pulse-time source.

`StimSource` provides two things the decoder needs: the stim-on condition (which
samples are under stimulation, for routing and for gating PARRM) and the PARRM period
P. Three implementations cover the ways this information can be supplied:

  * TriggerStimSource(trig_ch)    - a trigger channel gives pulses, P and the mask.
  * PulseTimesStimSource(times)   - no trigger: an explicit sorted array of pulse-centre
                                    sample indices (the stimpulses_<key>.npy format).
  * MaskStimSource(mask, times|P) - a precomputed boolean stim-on mask gives the
                                    condition; P (from `times` or given) is still needed
                                    because the PARRM comb requires the period.

Each source has an offline face (`calibrate` -> lock thr/P; `offline_mask(n)` -> the
non-causal mask used by `parrm_offline`) and a streaming face (`begin_stream` /
`push` / `on_block` -> the causal mask). Keeping both faces here isolates the small
offline-vs-causal difference at bout edges. numpy only.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import DecoderConfig
from .pulses import (estimate_trig_mid, detect_pulses, stim_on_mask,
                     period_from_pulses, StreamPulses)


def causal_on_from_pulses(pulses: np.ndarray, s0: int, n: int, P: float,
                          post: float = 2.0) -> np.ndarray:
    """Strictly-causal stim-on for [s0, s0+n): sample t is on iff some pulse p with
    p <= t <= p + round(post*P). Post-only (no lookahead) — the streaming semantics."""
    on = np.zeros(n, bool)
    g2 = int(round(post * P))
    pulses = np.asarray(pulses, int)
    lo = np.searchsorted(pulses, s0 - g2)                # first pulse that can still reach s0
    hi = np.searchsorted(pulses, s0 + n - 1, side="right")
    for p in pulses[lo:hi]:
        a = max(p - s0, 0)
        b = min(p - s0 + g2 + 1, n)
        if a < b:
            on[a:b] = True
    return on


class StimSource(ABC):
    """P is the locked PARRM period; set by `calibrate`."""

    def __init__(self, cfg: DecoderConfig):
        self.cfg = cfg
        self.P: float = 0.0
        self.thr: float | None = None
        self.pulses: np.ndarray | None = None

    def trig_channel(self):
        """Row index of the trigger channel in the raw chunk, or None."""
        return None

    @abstractmethod
    def calibrate(self, trig: np.ndarray | None = None):
        """Lock thr/P (and offline pulses). Returns (thr, P)."""

    @abstractmethod
    def offline_mask(self, n: int) -> np.ndarray:
        """Non-causal stim-on mask over [0, n) for whole-array PARRM."""

    # -- streaming face --
    @abstractmethod
    def begin_stream(self) -> None: ...

    @abstractmethod
    def push(self, chunk_full: np.ndarray, s0: int) -> None:
        """Ingest one raw chunk (channels x samples) at absolute base s0."""

    @abstractmethod
    def on_block(self, s0: int, n: int) -> np.ndarray:
        """Strictly-causal stim-on mask for [s0, s0+n)."""


class TriggerStimSource(StimSource):
    """Stim condition + pulses from a trigger channel row of the raw stream."""

    def __init__(self, trig_ch: int, cfg: DecoderConfig):
        super().__init__(cfg)
        self.trig_ch = int(trig_ch)
        self._sp: StreamPulses | None = None

    def trig_channel(self):
        return self.trig_ch

    def calibrate(self, trig: np.ndarray | None = None):
        if trig is None:
            raise ValueError("TriggerStimSource.calibrate needs the trigger row")
        self.pulses, self.P, self.thr = detect_pulses(
            trig, self.cfg.refractory, self.cfg.fs, self.cfg.trig_k)
        return self.thr, self.P

    def offline_mask(self, n: int) -> np.ndarray:
        return stim_on_mask(n, self.pulses, self.P, self.cfg.parrm_pre, self.cfg.parrm_post)

    def begin_stream(self) -> None:
        if self.thr is None or not self.P:
            raise RuntimeError("calibrate() before begin_stream() (thr/P must be locked)")
        self._sp = StreamPulses(self.thr, self.P, self.cfg.refractory, self.cfg.parrm_post)

    def push(self, chunk_full: np.ndarray, s0: int) -> None:
        self._sp.push(np.asarray(chunk_full)[self.trig_ch], s0)

    def on_block(self, s0: int, n: int) -> np.ndarray:
        return self._sp.on_block(s0, n)


class PulseTimesStimSource(StimSource):
    """No-trigger path: an explicit array of pulse-centre sample indices (the
    `stimpulses_<key>.npy` format). Provides both the condition mask and P."""

    def __init__(self, pulse_samples, cfg: DecoderConfig, P: float | None = None):
        super().__init__(cfg)
        self.pulses = np.asarray(sorted(int(p) for p in np.asarray(pulse_samples)), int)
        self._P_given = P

    def calibrate(self, trig: np.ndarray | None = None):
        self.P = float(self._P_given) if self._P_given else period_from_pulses(self.pulses, self.cfg.fs)
        return None, self.P

    def offline_mask(self, n: int) -> np.ndarray:
        return stim_on_mask(n, self.pulses, self.P, self.cfg.parrm_pre, self.cfg.parrm_post)

    def begin_stream(self) -> None:
        if not self.P:
            self.calibrate()

    def push(self, chunk_full: np.ndarray, s0: int) -> None:
        pass                                              # pulses are static

    def on_block(self, s0: int, n: int) -> np.ndarray:
        return causal_on_from_pulses(self.pulses, s0, n, self.P, self.cfg.parrm_post)


class MaskStimSource(StimSource):
    """A precomputed per-sample boolean stim-on mask defines the condition (and the
    PARRM gate). P is still required for the comb: pass it directly, or pass
    `pulse_times` to estimate it from their spacing."""

    def __init__(self, mask, cfg: DecoderConfig, P: float | None = None,
                 pulse_times=None):
        super().__init__(cfg)
        self.mask = np.asarray(mask, bool)
        if P is None and pulse_times is None:
            raise ValueError("MaskStimSource needs P or pulse_times (PARRM's comb needs "
                             "the period; a mask alone cannot supply it)")
        self._P_given = P
        self.pulses = (np.asarray(sorted(int(p) for p in np.asarray(pulse_times)), int)
                       if pulse_times is not None else None)

    def calibrate(self, trig: np.ndarray | None = None):
        self.P = float(self._P_given) if self._P_given else period_from_pulses(self.pulses, self.cfg.fs)
        return None, self.P

    def _slice(self, s0: int, n: int) -> np.ndarray:
        out = np.zeros(n, bool)
        a, b = max(s0, 0), min(s0 + n, len(self.mask))
        if a < b:
            out[a - s0:b - s0] = self.mask[a:b]
        return out

    def offline_mask(self, n: int) -> np.ndarray:
        out = np.zeros(n, bool)
        m = min(n, len(self.mask))
        out[:m] = self.mask[:m]
        return out

    def begin_stream(self) -> None:
        if not self.P:
            self.calibrate()

    def push(self, chunk_full: np.ndarray, s0: int) -> None:
        pass

    def on_block(self, s0: int, n: int) -> np.ndarray:
        return self._slice(s0, n)
