"""Online movement-discrimination engine — the transplantable runtime.

A stateful, chunk-size-agnostic classifier for stimulated HD-EMG that turns a
live chunk stream into per-window movement decisions (close / trp / ext / rest).
It carries all DSP state internally so it can be fed ONE device chunk at a time
(any length) and never sees a future sample.

This is the single source of truth shared by:
  * processing/scripts/movement_disc/online_sim.py  (trains + validates it, and
    guards it with the chunk-invariance test), and
  * bend MovementClassifierManager  (runs it live off the Quattrocento stream).

Deliberately dependency-light: numpy + scipy only. The classifier and the
optional scaler+PCA pre-transform arrive already-fitted inside the artifact
(pickled sklearn objects); this module never imports sklearn, so it loads on the
acquisition machine without the training stack.

Per released sample the pipeline is: stateful notch -> stim blank (sample-and-
hold over [t-pre, t+post) at trigger pulses, carried across chunk boundaries) ->
stateful bandpass -> trailing-window RMS features -> frozen classifier -> causal
EMA. Decisions fire at step boundaries: 0, 1, or several per chunk.
"""
from __future__ import annotations

import pickle
import numpy as np
from scipy.signal import butter, iirnotch, tf2sos, sosfilt

FS = 2048
N_EMG = 192
STIM_HZ = 30
CLS = ["close", "trp", "ext"]                       # class 3 = rest / no-intent
RLABS = CLS + ["rest"]
DEFAULT_GRIDS = {"Grid1": (0, 64), "Grid2": (64, 128), "Grid3": (128, 192)}


# ── stateful causal filters (as SOS + per-channel zi) ───────────────────────────
def design_filters(fs=FS, bp=(20.0, 450.0), notch_base=50.0, Q=30.0):
    """Bandpass + stacked 50 Hz-harmonic notch, in SOS form for stateful sosfilt."""
    bp_sos = butter(4, list(bp), btype="band", fs=fs, output="sos")
    notch = [tf2sos(*iirnotch(w0=f0, Q=Q, fs=fs))
             for f0 in np.arange(notch_base, fs / 2, notch_base)]
    return bp_sos, np.vstack(notch)


def _zi(sos, nch):
    return np.zeros((nch, sos.shape[0], 2))


def _filt_mc(sos, x, zi):
    """Per-channel stateful sosfilt over (nch, m); updates and returns zi."""
    out = np.empty_like(x)
    for c in range(x.shape[0]):
        out[c], zi[c] = sosfilt(sos, x[c], zi=zi[c])
    return out, zi


# ══════════════════════════════════════════════════════════════════════════════
class OnlineMovementClassifier:
    """Stateful, chunk-size-agnostic movement classifier for stimulated HD-EMG.

    process(chunk) consumes ONE (n_channels, n_samples) chunk (any length; EMG
    rows selected by `gidx`, trigger row = `trig_ch`) and returns the list of
    decision/feature events finalised during this chunk. All DSP state lives in
    the instance and is carried across chunks.
    """

    def __init__(self, gidx, trig_mid, fs=FS, win_ms=256, step_ms=64, smooth=0.3,
                 pre=8, post=23, trig_ch=N_EMG, grids=None,
                 bp=(20.0, 450.0), notch_base=50.0, Q=30.0):
        self.gidx = np.asarray(gidx, int)
        self.ng = len(self.gidx)
        self.trig_ch = trig_ch
        self.trig_mid = float(trig_mid)
        self.fs = fs
        self.win_ms, self.step_ms, self.smooth = win_ms, step_ms, smooth
        self.win = int(win_ms / 1000 * fs)
        self.step = int(step_ms / 1000 * fs)
        self.pre, self.post = pre, post
        self.HOLD = pre + post          # commit latency (samples)
        self.grids = grids or DEFAULT_GRIDS
        self.bp_sos, self.notch_sos = design_filters(fs, bp, notch_base, Q)

        # per-grid row positions (into the good-channel signal) + window freqs,
        # precomputed for the fixed window length (feature math == mvd.window_feats
        # with td=False: mean-centered log-RMS spatial map + per-grid energy + median freq)
        self._grid_pos = {g: np.where((self.gidx >= a) & (self.gidx < b))[0]
                          for g, (a, b) in self.grids.items()}
        self._freqs = np.fft.rfftfreq(self.win, 1 / fs)

        self.model = None
        self.pretf = None               # frozen scaler+PCA pre-transform (NOT blank pre)
        # Optional CLEAN (nostim) specialist: used per-window when the window carries
        # NO blanked samples (pre-stim / clean-onset regime). When set, `model`/`pretf`
        # act as the STIM specialist (windows that DO contain blanked samples). This is
        # the two-stage router — matches how each model is trained (nostim never
        # blanked, stim blanked). If None, every window uses `model` (single-model).
        self.model_clean = None
        self.pretf_clean = None
        # Was the CLEAN specialist trained with the trailing global log-RMS level
        # feature appended? (a magnitude cue for rest-vs-move the map discards).
        self.clean_amp = False
        # When True, the STIM head's live decision is reported as a move/rest gate
        # (1 - P(rest)); the 4-class output still rides along. Runtime-toggleable.
        self.stim_binary = False
        self.class_order = None
        self.reset()

    # -- lifecycle ----------------------------------------------------------------
    def reset(self):
        """Clear all streaming state (filters, buffers, counters). Keeps model."""
        self.notch_zi = _zi(self.notch_sos, self.ng)
        self.bp_zi = _zi(self.bp_sos, self.ng)
        self.nbuf = np.zeros((self.ng, 0))     # notched, not-yet-released
        self.nbuf_start = 0                     # abs index of nbuf column 0
        self.bpbuf = np.zeros((self.ng, 0))    # bandpassed, awaiting windowing
        self.blankbuf = np.zeros(0, bool)       # per-column: was this sample blanked?
        self.bpbuf_start = 0
        self.total_in = 0                       # EMG samples ingested (frontier)
        self.prev_below = False                 # trigger below-threshold carry
        self.pending = []                       # [t_abs, src_val] pulses in flight
        self.win_start = 0                      # abs start of the next window
        self.ema = np.array([0.0, 0.0, 0.0, 1.0])   # start certain of "rest"
        self._events = []

    def set_model(self, model, pre=None):
        """Freeze a fitted classifier (predict_proba over 0=close,1=trp,2=ext,
        3=rest) plus an optional frozen pre-transform `pre` (scaler+PCA) applied to
        each feature row before the classifier. Resets the smoother, not the DSP.
        This is the STIM specialist (used on windows that contain blanked samples)."""
        self.model = model
        self.pretf = pre
        self.ema = np.array([0.0, 0.0, 0.0, 1.0])

    def set_clean_model(self, model, pre=None, amp=False):
        """Freeze the CLEAN (nostim) specialist, used on windows with no blanked
        samples (pre-stim / clean onset). Pass model=None to disable two-stage
        routing and run `model` on every window. `amp=True` means the clean model
        was trained with the global log-RMS level appended as a trailing feature
        (see _features / _proba4), so feed it that extra column live."""
        self.model_clean = model
        self.pretf_clean = pre
        self.clean_amp = bool(amp)

    def set_stim_binary(self, on):
        """Toggle the STIM head between full 4-class output and a move/rest gate
        (1 - P(rest)). Runtime-safe; reporting only — the 4-class pred/probs still
        ride along and actuation keeps using pred. The clean head is never collapsed."""
        self.stim_binary = bool(on)

    @classmethod
    def from_artifact(cls, art, trig_ch=None):
        """Build an engine from a saved artifact dict (see save_artifact)."""
        eng = cls(gidx=art["gidx"], trig_mid=art["trig_mid"], fs=art["fs"],
                  win_ms=art["win_ms"], step_ms=art["step_ms"], smooth=art["smooth"],
                  pre=art["pre"], post=art["post"],
                  trig_ch=art["trig_ch"] if trig_ch is None else trig_ch,
                  grids=art.get("grids"), bp=tuple(art.get("bp", (20.0, 450.0))),
                  notch_base=art.get("notch_base", 50.0), Q=art.get("Q", 30.0))
        eng.set_model(art["model"], art.get("pretf"))
        eng.set_clean_model(art.get("model_clean"), art.get("pretf_clean"), art.get("clean_amp", False))
        eng.class_order = art.get("class_order", RLABS)
        eng.stim_binary = bool(art.get("stim_binary", False))
        return eng

    # -- per-chunk ----------------------------------------------------------------
    def process(self, chunk):
        chunk = np.asarray(chunk)
        m = chunk.shape[1]
        if m == 0:
            return []
        self._events = []
        emg = chunk[self.gidx].astype(np.float64)
        trig = chunk[self.trig_ch].astype(np.float64)

        # 1) stateful notch, appended to the not-yet-released buffer
        y, self.notch_zi = _filt_mc(self.notch_sos, emg, self.notch_zi)
        base = self.total_in
        self.nbuf = np.hstack([self.nbuf, y])
        self.total_in += m

        # 2) trigger pulse detection (high->low crossings) with carry-over
        below = trig < self.trig_mid
        shifted = np.empty(m, bool)
        shifted[0] = self.prev_below
        shifted[1:] = below[:-1]
        for i in np.flatnonzero(below & ~shifted):
            t = base + int(i)
            src_col = (t - self.pre - 1) - self.nbuf_start
            src = self.nbuf[:, src_col].copy() if 0 <= src_col < self.nbuf.shape[1] else None
            self.pending.append([t, src])
        self.prev_below = bool(below[-1])

        # 3) release everything older than HOLD (blank -> bandpass -> window)
        self._release(self.total_in - self.HOLD)
        return self._events

    def _release(self, safe_abs):
        n_rel = safe_abs - self.nbuf_start
        if n_rel <= 0:
            return
        block = self.nbuf[:, :n_rel].copy()             # blanked copy; nbuf stays clean
        blanked = np.zeros(n_rel, bool)                  # which columns get sample-held
        for t, src in self.pending:                     # sample-and-hold over each window
            if src is None:
                continue
            a = max(t - self.pre, self.nbuf_start)
            b = min(t + self.post, safe_abs)
            if a < b:
                block[:, a - self.nbuf_start: b - self.nbuf_start] = src[:, None]
                blanked[a - self.nbuf_start: b - self.nbuf_start] = True
        bp, self.bp_zi = _filt_mc(self.bp_sos, block, self.bp_zi)
        self.bpbuf = np.hstack([self.bpbuf, bp])
        self.blankbuf = np.concatenate([self.blankbuf, blanked])   # stays aligned to bpbuf
        self.nbuf = self.nbuf[:, n_rel:]
        self.nbuf_start = safe_abs
        self.pending = [p for p in self.pending if (p[0] + self.post) > safe_abs]
        self._emit_windows(safe_abs)

    def _emit_windows(self, bp_total):
        while self.win_start + self.win <= bp_total:
            s = self.win_start
            lo, hi = s - self.bpbuf_start, s + self.win - self.bpbuf_start
            seg = self.bpbuf[:, lo:hi]
            stim_on = bool(self.blankbuf[lo:hi].any())   # any blanked sample -> stim regime
            feat, amp = self._features(seg)
            ev = {"end": s + self.win - 1, "feat": feat, "amp": amp, "stim": stim_on}
            if self.model is not None:
                P = self._proba4(feat, stim_on, amp)
                self.ema = self.smooth * P + (1 - self.smooth) * self.ema
                ev["probs"] = P
                ev["sm"] = self.ema.copy()
                ev["pred"] = int(self.ema.argmax())
                # Move/rest readout on the smoothed belief. When stim_binary is on, the
                # STIM head's decision is a move/rest gate (1 - P(rest)); the clean head
                # stays 4-class. 4-class pred/probs ride along for logging + actuation.
                rest = (len(self.class_order) - 1) if self.class_order else 3
                ev["p_move"] = float(1.0 - self.ema[rest])
                ev["binary"] = bool(self.stim_binary and stim_on)
            self._events.append(ev)
            self.win_start += self.step
        if self.win_start > self.bpbuf_start:           # trim consumed history
            cut = self.win_start - self.bpbuf_start
            self.bpbuf = self.bpbuf[:, cut:]
            self.blankbuf = self.blankbuf[cut:]
            self.bpbuf_start = self.win_start

    def _features(self, seg):
        """RMS-only window features (== mvd.window_feats td=False, one window):
        mean-centered log-RMS spatial map (ng) + per-grid energy fraction (3) +
        per-grid median frequency (3). Every block is amplitude-invariant, so the
        vector carries no absolute magnitude. Returns (feat, level) where `level` is
        the global mean log-RMS the map is centered by — the single absolute-amplitude
        scalar the map discards, kept for the CLEAN head (see _proba4)."""
        seg = seg.astype(np.float64)
        rms = np.sqrt(np.mean(seg ** 2, axis=1)) + 1e-9
        logrms = np.log(rms)
        level = float(logrms.mean())
        spatial = logrms - level
        P = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        eg, mf = [], []
        for g in self.grids:
            pos = self._grid_pos[g]
            if len(pos) == 0:
                eg.append(0.0); mf.append(0.0); continue
            eg.append(float(np.sum(rms[pos] ** 2)))
            c = np.cumsum(P[pos].mean(axis=0))
            mf.append(float(self._freqs[np.searchsorted(c, c[-1] / 2)]))
        eg = np.array(eg); eg = eg / (eg.sum() + 1e-12)
        feat = np.concatenate([spatial, eg, np.array(mf)]).astype(np.float32)
        return feat, level

    def _proba4(self, feat, stim_on=True, amp=None):
        # Route to the CLEAN specialist on unblanked windows when one is loaded;
        # otherwise (single-model) always use `model`.
        if not stim_on and self.model_clean is not None:
            model, pretf = self.model_clean, self.pretf_clean
            # The clean head is (optionally) trained with the global log-RMS level
            # appended — an absolute-amplitude cue for rest-vs-move the map lacks.
            if self.clean_amp and amp is not None:
                feat = np.append(feat, np.float32(amp))
        else:
            model, pretf = self.model, self.pretf
        x = feat[None]
        if pretf is not None:
            x = pretf.transform(x)
        pr = model.predict_proba(x)[0]
        P = np.zeros(4)
        for j, c in enumerate(model.classes_):
            P[int(c)] = pr[j]
        return P


# ── artifact persistence (the calibration output bend loads) ────────────────────
def save_artifact(path, *, gidx, trig_mid, model, pretf=None, class_order=None,
                  fs=FS, win_ms=256, step_ms=64, smooth=0.3, pre=8, post=23,
                  trig_ch=N_EMG, grids=None, bp=(20.0, 450.0), notch_base=50.0,
                  Q=30.0, meta=None, model_clean=None, pretf_clean=None, clean_amp=False,
                  stim_binary=False):
    """Freeze everything the live engine needs into one .pkl. `model`/`pretf` are the
    STIM specialist; the optional `model_clean`/`pretf_clean` are the CLEAN (nostim)
    specialist for two-stage routing. All estimators are fitted sklearn objects."""
    art = dict(gidx=np.asarray(gidx, int), trig_mid=float(trig_mid), model=model,
               pretf=pretf, model_clean=model_clean, pretf_clean=pretf_clean,
               clean_amp=bool(clean_amp), stim_binary=bool(stim_binary),
               class_order=class_order or RLABS, fs=fs, win_ms=win_ms,
               step_ms=step_ms, smooth=smooth, pre=pre, post=post, trig_ch=trig_ch,
               grids=grids or DEFAULT_GRIDS, bp=tuple(bp), notch_base=notch_base,
               Q=Q, meta=meta or {})
    with open(path, "wb") as f:
        pickle.dump(art, f)
    return path


def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)
