"""StreamingDecoder: run the full causal pipeline on a chunk stream.

`process(chunk)` takes one raw chunk [channels, samples] of any length and returns the
decision events finalised during it. The pipeline per chunk is: select subset channels
-> causal notch -> PARRM (gated by the causal stim-on mask) -> causal bandpass ->
trailing windows -> gate/gesture features -> decoder -> decider. All state lives on the
instance and is carried across calls, so the output does not depend on how the stream
is chunked.

The stim source must be calibrated (P locked) before use; PARRM's history takes about
25*P samples to fill, and events before that are flagged `warm=False`.

If the decoder carries a clean (no-stim) head, each window is routed by its causal stim
flag: stim-on windows go to the gate and gesture heads, stim-off windows to the clean
head. Only the head that owns the window runs, so a clean window costs one model call,
not two. Each event names its head. Without a clean head every window takes the gate +
gesture path, unchanged.

numpy/scipy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig
from .dsp import design_filters, CausalFilter
from .parrm import StreamPARRM
from .features import FeatureExtractor, AmplitudeHistory
from .decoder import TwoStageDecoder
from .decider import Decider
from .stim_source import StimSource


class StreamingDecoder:
    def __init__(self, decoder: TwoStageDecoder, source: StimSource, keep_clean=False):
        """`keep_clean` keeps every cleaned, band-passed subset sample for later plotting
        (see `clean_signal`). It costs 4 bytes per subset channel per sample, so a few
        hundred MB over a long recording. Leave it off for a plain run."""
        self.cfg: DecoderConfig = decoder.cfg
        self.decoder = decoder
        self.source = source
        self.keep_clean = bool(keep_clean)
        self.feat = FeatureExtractor(self.cfg, decoder.good)
        self.decider = Decider(self.cfg)
        self.keep_abs = decoder.good[self.feat.keep_pos]        # absolute subset channels
        self.nsub = len(self.keep_abs)
        bp_sos, notch_sos = design_filters(self.cfg)
        self.notch = CausalFilter(notch_sos, self.nsub)
        self.bp = CausalFilter(bp_sos, self.nsub)
        self.amp_hist = AmplitudeHistory(self.feat.used_lags)
        self.reset()

    def reset(self):
        """Clear all streaming state. Requires the source's P to be set."""
        if not self.source.P:
            self.source.calibrate()
        self.source.begin_stream()
        self.notch.reset()
        self.bp.reset()
        self.parrm = StreamPARRM(self.nsub, self.source.P, self.cfg.parrm_m, self.cfg.parrm_skip)
        self.amp_hist.reset()
        self.decider.reset()
        self.warmup = self.parrm.need
        self.total_in = 0
        self.bpbuf = np.zeros((self.nsub, 0))
        self.onbuf = np.zeros(0, bool)
        self.buf_start = 0
        self.win_next = 0
        self._first = True
        self._clean_log, self._on_log = [], []

    def clean_signal(self):
        """The cleaned band-passed subset signal the decoder saw, as
        (clean[n_subset, N] float32, on[N] bool, keep_channels). Needs keep_clean=True."""
        if not self.keep_clean:
            raise RuntimeError("construct with keep_clean=True to keep the cleaned signal")
        return (np.hstack(self._clean_log), np.concatenate(self._on_log), self.keep_abs)

    def process(self, chunk: np.ndarray) -> list:
        chunk = np.asarray(chunk)
        n = chunk.shape[1]
        if n == 0:
            return []
        s0 = self.total_in
        x = chunk[self.keep_abs].astype(float)
        self.source.push(chunk, s0)
        y = self.notch.apply(x)
        on = self.source.on_block(s0, n)
        cl = self.parrm.push(y)
        cl[:, ~on] = y[:, ~on]                      # gate: only remove where stim runs
        bp = self.bp.apply(cl)
        if self.keep_clean:
            self._clean_log.append(bp.astype(np.float32))
            self._on_log.append(on.copy())
        self.bpbuf = np.hstack([self.bpbuf, bp])
        self.onbuf = np.concatenate([self.onbuf, on])
        self.total_in += n
        return self._emit()

    def _emit(self) -> list:
        win, step = self.cfg.win, self.cfg.step
        fs = self.cfg.fs
        events = []
        while self.win_next + win <= self.buf_start + self.bpbuf.shape[1]:
            off = self.win_next - self.buf_start
            seg = self.bpbuf[:, off:off + win]
            stim_flag = bool(self.onbuf[off:off + win].any())
            A, spatial, eg, mf, A_sub = self.feat.window_kernel(seg)
            dA = self.amp_hist.push(A)
            xg = self.feat.gate_vector(A, dA)
            xs = self.feat.gesture_vector(spatial, eg, mf, A_sub)
            end = self.win_next + win - 1
            new_run = self._first
            self._first = False
            t_ms = end / fs * 1000.0
            gate_p = gest_p = clean_p = None
            if not self.decoder.has_clean:                  # stim-only decoder
                gate_p = self.decoder.gate_proba(xg)
                gest_p = self.decoder.gesture_proba(xs)
                dec = self.decider.update(gate_p, gest_p, t_ms, new_run)
            elif stim_flag:                                 # routed: stim regime
                gate_p = self.decoder.gate_proba(xg)
                gest_p = self.decoder.gesture_proba(xs)
                dec = self.decider.update_routed(True, t_ms, gate_p, gest_p,
                                                 new_run=new_run)
            else:                                           # routed: clean regime
                clean_p = self.decoder.clean_proba(xs)
                dec = self.decider.update_routed(False, t_ms, clean_proba=clean_p,
                                                 new_run=new_run)
            events.append(dict(end=end, stim=stim_flag, warm=end >= self.warmup,
                               gate_proba=gate_p, gest_proba=gest_p, clean_proba=clean_p,
                               xg=xg, xs=xs, **dec))
            self.win_next += step
        if self.win_next > self.buf_start:              # drop consumed history
            cut = self.win_next - self.buf_start
            self.bpbuf = self.bpbuf[:, cut:]
            self.onbuf = self.onbuf[cut:]
            self.buf_start = self.win_next
        return events
