"""Offline (whole-array) feature extraction and decisioning.

`extract_offline` runs notch -> PARRM -> bandpass over a whole recording, then windows
at the given end samples and builds the gate/gesture matrices with the same
`FeatureExtractor` kernel the streaming path uses. `decide_offline` runs the same
`Decider` over those windows in time order. Use this for training-set feature building
and for fast evaluation without a real-time clock.

The stim source must be calibrated (P set) before calling. numpy/scipy only.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt

from .config import DecoderConfig
from .dsp import design_filters
from .parrm import parrm_offline, refine_period
from .features import FeatureExtractor, deltas_offline
from .decoder import TwoStageDecoder
from .decider import Decider
from .stim_source import StimSource


def extract_offline(emg: np.ndarray, source: StimSource, cfg: DecoderConfig,
                    good: np.ndarray, ends=None, refine=False, return_clean=False) -> dict:
    """Whole-array extraction. Returns a dict with E (window ends), Xgate, Xgest,
    S (per-window stim flag) and `on` (the sample-level stim mask). If `ends` is None,
    windows are tiled at the configured step.

    With `refine=True` and a pulse-based source, the PARRM period is refined against the
    residual stim artifact before removal (for the no-trigger path); `source.P` is
    updated in place so a later streaming pass can reuse it.

    With `return_clean=True`, the result also holds `clean` (the cleaned, band-passed
    subset signal [n_subset, N], float32) and `keep_channels` (its absolute channel
    indices) and `notch` (the notch-only signal, for before/after comparison)."""
    feat = FeatureExtractor(cfg, good)
    keep_abs = np.asarray(good, int)[feat.keep_pos]
    x = np.asarray(emg, float)[keep_abs]
    N = x.shape[1]
    bp_sos, notch_sos = design_filters(cfg)
    notchC = sosfilt(notch_sos, x, axis=1)
    if refine and getattr(source, "pulses", None) is not None:
        source.P = refine_period(notchC, source.pulses, source.P, cfg)
    on = source.offline_mask(N)
    y = parrm_offline(notchC, source.P, on, cfg.parrm_m, cfg.parrm_skip)
    bp = sosfilt(bp_sos, y, axis=1)

    if ends is None:
        ends = np.arange(cfg.win - 1, N, cfg.step)
    ends = np.asarray(ends, int)

    A_rows, gest_rows, stim = [], [], []
    for e in ends:
        seg = bp[:, e - cfg.win + 1:e + 1]
        A, spatial, eg, mf, A_sub = feat.window_kernel(seg)
        A_rows.append(A)
        gest_rows.append(feat.gesture_vector(spatial, eg, mf, A_sub))
        stim.append(bool(on[e - cfg.win + 1:e + 1].any()))
    AG = np.array(A_rows)
    if feat.used_lags:
        dA = deltas_offline(AG, ends, feat.used_lags)
        Xgate = np.hstack([AG, dA])
    else:
        Xgate = AG
    res = dict(E=ends, Xgate=Xgate, Xgest=np.array(gest_rows),
               S=np.array(stim, bool), on=on)
    if return_clean:
        res.update(clean=bp.astype(np.float32), notch=notchC.astype(np.float32),
                   keep_channels=keep_abs)
    return res


def decide_offline(decoder: TwoStageDecoder, Xgate, Xgest, E, cfg: DecoderConfig) -> dict:
    """Run the decider over the windows in time order. Returns gate/gesture smoothed
    outputs and the per-window decision dicts (in the input order of E)."""
    E = np.asarray(E, int)
    gate_pb = decoder.gate_proba_batch(Xgate)
    gest_pb = decoder.gesture_proba_batch(Xgest)
    dec = Decider(cfg)
    out = [None] * len(E)
    last_end = None
    for idx in np.argsort(E):
        e = int(E[idx])
        new_run = last_end is None or (e - last_end) > 1.5 * cfg.step
        last_end = e
        out[idx] = dec.update(gate_pb[idx], gest_pb[idx], e / cfg.fs * 1000.0, new_run)
    return dict(gate_proba=gate_pb, gest_proba=gest_pb, decisions=out)
