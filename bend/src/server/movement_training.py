"""Shared movement-model training — the north-star fit logic.

This is the SINGLE source of truth for how a movement-discrimination model is
trained/calibrated, extracted verbatim from online_sim / movement_discrimination
so that BOTH the pseudo-online simulation (online_sim.py, which validates it) and
the live bend train/calibrate commands run the *same* steps:

  * good-channel selection  = per-grid robust RMS-outlier mask (compute_good_mask)
  * feature collection       = the shared OnlineMovementClassifier engine
  * fit                      = optional scaler+PCA (on train) -> CORAL align
                               train->calibration -> classifier

Dependency-light (numpy / scipy / sklearn); imports the shared engine only. Keep
this byte-faithful to online_sim: if you change a step here, re-run online_sim and
confirm the numbers (steady-state acc + chunk-invariance) are unchanged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt

from server.movement_engine import (OnlineMovementClassifier, DEFAULT_GRIDS, N_EMG, FS,
                             save_artifact)


# ── channel selection (== movement_discrimination.compute_good_mask) ────────────
def _notch(x, fs=FS, base=50.0, Q=30.0):
    y = x.astype(np.float64, copy=True)
    for f0 in np.arange(base, fs / 2, base):
        b, a = iirnotch(w0=f0, Q=Q, fs=fs)
        y = filtfilt(b, a, y, axis=1)
    return y


def _bandpass(x, fs=FS, bp=(20.0, 450.0)):
    sos = butter(4, list(bp), btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=1)


def bad_channel_mask(rms, grids=DEFAULT_GRIDS, lo=0.2, hi=5.0):
    """Per-grid robust RMS-outlier mask (True = bad electrode)."""
    bad = np.zeros(len(rms), dtype=bool)
    for a, b in grids.values():
        med = np.median(rms[a:b])
        if med <= 0:
            bad[a:b] = True
            continue
        ratio = rms[a:b] / med
        bad[a:b] = (ratio > hi) | (ratio < lo) | ~np.isfinite(ratio)
    return bad


def good_channel_mask(emg, grids=DEFAULT_GRIDS, fs=FS):
    """Good-channel boolean mask over EMG rows, from per-grid RMS outliers on the
    band-passed signal (the live analogue of compute_good_mask: one multi-movement
    recording instead of the separate single-movement runs). The EMG channel count
    is taken from the grid layout, so this works for any montage."""
    n_emg = max(b for _, b in grids.values())
    xb = _bandpass(_notch(emg[:n_emg].astype(np.float32), fs), fs)
    rms = np.sqrt(np.mean(xb.astype(np.float64) ** 2, axis=1))
    return ~bad_channel_mask(rms, grids)


def good_gidx(emg, grids=DEFAULT_GRIDS, fs=FS):
    return np.where(good_channel_mask(emg, grids, fs))[0]


# ── manual channel-quality review (operator eyeballs each channel's EMG) ────────
def filter_emg(raw, grids=DEFAULT_GRIDS, fs=FS):
    """Notch + band-pass the EMG rows of a raw recording, returning (filt, n_emg).
    Same filtering the good-channel mask uses — so what the operator sees during the
    manual channel review matches what auto-rejection judged and what the model trains
    on. The EMG channel count comes from the grid layout (montage-agnostic)."""
    n_emg = max(b for _, b in grids.values())
    xb = _bandpass(_notch(np.asarray(raw)[:n_emg].astype(np.float32), fs), fs)
    return xb, int(n_emg)


def channel_review(filt, grids=DEFAULT_GRIDS):
    """Per-channel RMS + the AUTO bad-channel mask from already-filtered EMG. The auto
    verdict seeds the manual review; the operator then overrides per channel."""
    rms = np.sqrt(np.mean(filt.astype(np.float64) ** 2, axis=1))
    bad = bad_channel_mask(rms, grids)
    return dict(rms=[round(float(r), 4) for r in rms],
                auto_bad=[int(i) for i in np.where(bad)[0]])


def decimate_minmax(x, n_points=1500):
    """Downsample a 1-D signal to `n_points` (min, max) pairs so a long recording plots
    as a faithful envelope (spikes preserved) in a small payload. Returns (mins, maxs)."""
    x = np.asarray(x, dtype=np.float32)
    n = x.shape[0]
    if n <= n_points:
        return x.tolist(), x.tolist()
    edges = np.linspace(0, n, n_points + 1, dtype=int)
    edges[-1] = n
    mins = np.empty(n_points, np.float32)
    maxs = np.empty(n_points, np.float32)
    for i in range(n_points):
        a, b = int(edges[i]), int(edges[i + 1])
        if b <= a:
            b = a + 1
        seg = x[a:b]
        mins[i] = float(seg.min())
        maxs[i] = float(seg.max())
    return mins.tolist(), maxs.tolist()


# ── feature collection through the shared engine ────────────────────────────────
def bout_labeler(bouts, rest=3):
    """bouts: list of (start, end, class_idx). Returns fn(end_abs) -> class idx if
    the window end falls in a bout, else `rest`. (== online_sim.label_by_bouts.)"""
    lut = [(int(s), int(e), int(c)) for s, e, c in bouts]

    def fn(end):
        for s, e, c in lut:
            if s <= end < e:
                return c
        return rest
    return fn


def collect_features(engine: OnlineMovementClassifier, raw, start, end, label_fn,
                     with_amp=False, blanked_only=False):
    """Feed raw[:, start:end] (channels incl. trigger) through a FRESH engine and
    return (X, y) from the emitted windows, labelled by end-sample. Chunk-invariant,
    so feeding the whole region in one call matches online_sim's fixed-chunk feed.

    With `with_amp`, each window's global log-RMS level (event 'amp') is appended as a
    trailing feature. Used for the CLEAN specialist so it keeps an absolute-amplitude
    cue (rest-vs-move) that the amplitude-invariant spatial map throws away.

    With `blanked_only`, only windows that contain blanked (stim-artifact) samples are
    kept — the regime the STIM head actually sees live. Used so the stim head's rest
    class comes from stimulated-rest bouts, not the unblanked between-bout gaps it
    never encounters at inference."""
    engine.reset()
    evs = engine.process(np.asarray(raw)[:, start:end])
    if blanked_only:
        evs = [e for e in evs if e.get("stim")]
    width = engine.ng + 6 + (1 if with_amp else 0)
    if not evs:
        return np.zeros((0, width), np.float32), np.zeros(0, int)
    if with_amp:
        X = np.array([np.append(e["feat"], np.float32(e["amp"])) for e in evs], np.float32)
    else:
        X = np.array([e["feat"] for e in evs], np.float32)
    y = np.array([label_fn(e["end"]) for e in evs], int)
    return X, y


# ── CORAL (== movement_discrimination.coral_transform) ──────────────────────────
def _sym_pow(C, p):
    w, V = np.linalg.eigh(C)
    return (V * np.clip(w, 1e-12, None) ** p) @ V.T


def coral_transform(Xs, Xt, shrink=1.0):
    """Recolor source features Xs to match target Xt's second-order stats + mean.
    Standardized by the source scale, heavy shrinkage because n_features can exceed
    n_windows. Train on the returned aligned Xs, predict on Xt-domain data."""
    Xs = Xs.astype(np.float64); Xt = Xt.astype(np.float64)
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    Zs, Zt = (Xs - mu) / sd, (Xt - mu) / sd
    d = Zs.shape[1]
    Cs = np.cov(Zs, rowvar=False) + shrink * np.eye(d)
    Ct = np.cov(Zt, rowvar=False) + shrink * np.eye(d)
    A = _sym_pow(Cs, -0.5) @ _sym_pow(Ct, 0.5)
    Zal = (Zs - Zs.mean(0)) @ A + Zt.mean(0)
    return (Zal * sd + mu).astype(np.float32)


# ── classifier factory (== movement_discrimination._clf_factory) ────────────────
def clf_factory(name):
    """Scaler+classifier pipeline (all expose predict_proba) for the RMS features."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    if name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    if name == "qda":
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        return make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.2))
    if name == "svm":
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(C=5, gamma="scale", probability=True, random_state=0))
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(StandardScaler(), RandomForestClassifier(300, n_jobs=-1, random_state=0))
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return make_pipeline(StandardScaler(), KNeighborsClassifier(15))
    raise ValueError(f"unknown clf {name!r}")


# ── supervised, GATED mean recalibration (CORAL alternative for a per-class shift) ─
class RecalQDA:
    """Per-class-covariance discriminant with SUPERVISED, GATED mean recalibration.

    Drop-in alternative to the CORAL->QDA path for the one failure CORAL is
    structurally blind to: a per-CLASS TRANSLATION of the feature means (the
    median->ulnar nerve shift — gestures move 1.6-2.3 SD, rest barely moves 0.31 SD).
    CORAL shifts the source to the GLOBAL target mean, so a rest-dominated target gets
    a near-zero, class-INDEPENDENT correction; QDA then measures an ABSOLUTE
    Mahalanobis distance to each mis-centred class and collapses to constant `rest`.
    This re-estimates the class MEANS from the labelled target reps instead, while
    keeping class SHAPES from the abundant source (the shift is a translation, so shape
    is intact, and a per-class TARGET covariance from ~3 reps would be singular).

    Each class mean is a GATED interpolation between its abundant-source mean and the
    mean of its labelled target reps, so a class only moves when its shift is real:

        mu_c = mu_src[c] + w_c * (mu_tgt[c] - mu_src[c])

      * w_c -> 1  full recentre onto the target reps (the fix for a real shift)
      * w_c -> 0  stay at the source mean (no recentre) — a class that didn't move, or
                  whose apparent move is within few-rep noise, is left alone
      * w_c is chosen so a class recentres only once its shift beats the noise of a
        few-rep mean estimate:
            mean_shrink="js"   James-Stein positive-part
                               w = max(0, 1 - (d / n_c) / ||delta_c||^2_Sw)
                               ||delta_c||^2 in pooled-source-cov units; d/n_c is the
                               squared shift a mean of n_c windows shows under NO real
                               shift. Big real shift -> w~1; shift within noise -> w~0.
                               No threshold to tune.
            mean_shrink="hard" w = 1 iff sqrt(||delta_c||^2 / d) > mean_tau (per-axis SD)
            mean_shrink="none" w = 1 always (full target mean, ungated)
            mean_shrink="off"  w = 0 (no recentring at all)

    Gating toward the SOURCE mean (rather than a shared shift) matters because `rest` is
    a genuine labelled class here: the guided flow cues a STIMULATED-rest bout, so `rest`
    carries its own target reps and is gated like any gesture. Its real shift is small
    (~0.3 SD), so the gate keeps it near its source mean instead of dragging it along a
    common shift it never experienced. A common-mode gain drift is still handled — a
    class fails to recentre only when its TOTAL shift is noise-level, and then any drift
    on it is negligible anyway. A class entirely ABSENT from the calib reps (a
    misconfiguration — the guided sequence covers every class, rest included) falls back
    to its source mean plus the mean observed shift.

    Shapes: per-class source covariance blended toward the pooled source covariance by
    `alpha` (alpha=1 -> shared cov == recalibrated LDA, bracketing the ceiling), shrunk
    toward its diagonal by `shrink`, then ridged by `reg` — this same regularisation is
    also what makes the discriminant survive collinear HD-EMG features. Uniform priors
    (the source's rest count dwarfs each gesture; source priors would re-bury the
    movements under rest).

    `report_` records the per-class decision (residual SD, n, weight, recentred flag)
    for logging; `summary()` renders it. `classes_` is arange(n_classes) so
    predict_proba columns line up with the 0..rest label convention with no adapter.
    """

    def __init__(self, alpha=0.7, shrink=0.2, reg=1e-3, mean_shrink="js", mean_tau=1.0):
        self.alpha, self.shrink, self.reg = alpha, shrink, reg
        self.mean_shrink, self.mean_tau = mean_shrink, float(mean_tau)
        self.report_ = {}

    def _weight(self, m2, n_c, d):
        """Gate in [0,1] for a class residual of squared Mahalanobis size m2, estimated
        from n_c target windows in d dims. `mean_tau` (hard mode) is a PER-AXIS SD
        threshold: the effect size is sqrt(m2/d), so tau=1 means "1 pooled-SD per axis
        on average", comparable to the report's 1.6-2.3 SD gesture shifts."""
        if self.mean_shrink == "none":
            return 1.0
        if self.mean_shrink == "off":
            return 0.0
        if m2 <= 0.0 or n_c <= 0:
            return 0.0
        if self.mean_shrink == "hard":
            return 1.0 if (m2 / d) > self.mean_tau ** 2 else 0.0
        # "js": positive-part James-Stein; d/n_c == E[||noise||^2] under no real shift
        return float(max(0.0, 1.0 - (d / n_c) / m2))

    def fit(self, Xs, ys, Xc, yc, n_classes):
        Xs = np.asarray(Xs, float); ys = np.asarray(ys, int); d = Xs.shape[1]
        mus_src, Sc, Sw, dof = {}, {}, np.zeros((d, d)), 0
        for c in np.unique(ys):
            Xk = Xs[ys == c]; mu = Xk.mean(0); D = Xk - mu
            mus_src[int(c)] = mu
            Sc[int(c)] = (D.T @ D) / max(len(Xk) - 1, 1)     # per-class shape (source)
            Sw += D.T @ D; dof += len(Xk) - 1
        Sw /= max(dof, 1)                                     # pooled shape (source)
        # conditioned pooled cov: the metric for the mean gate AND the alpha=1 floor
        Sw_reg = ((1 - self.shrink) * Sw + self.shrink * (np.trace(Sw) / d) * np.eye(d)
                  + self.reg * np.eye(d))
        Sw_inv = np.linalg.inv(Sw_reg)

        # target class means from the labelled calib reps -> per-class shift delta_c
        Xc = np.asarray(Xc, float); yc = np.asarray(yc, int)
        mu_tgt, n_tgt = {}, {}
        for c in np.unique(yc):
            m = yc == c
            mu_tgt[int(c)] = Xc[m].mean(0); n_tgt[int(c)] = int(m.sum())
        common = [c for c in mu_tgt if c in mus_src]
        delta = {c: mu_tgt[c] - mus_src[c] for c in common}
        # fallback shift for a class ABSENT from the calib reps (degenerate: the guided
        # sequence covers every class, so this is defensive only)
        gshift = np.mean([delta[c] for c in common], 0) if common else np.zeros(d)

        self.mu = np.zeros((n_classes, d))
        self.Pi, self.logdet = [], []
        self.report_ = {}
        for c in range(n_classes):
            if c in delta:
                m2 = float(delta[c] @ Sw_inv @ delta[c])      # shift size, pooled-SD^2
                w = self._weight(m2, n_tgt[c], d)
                self.mu[c] = mus_src[c] + w * delta[c]        # gate full shift -> source
                # report the per-axis effect size sqrt(m2/d) as "SD" (Mahalanobis grows
                # like sqrt(d); this keeps the log on the report's 1.6-2.3 SD scale)
                self.report_[int(c)] = dict(shift_sd=round(float(np.sqrt(max(m2, 0.0) / d)), 3),
                                            n=n_tgt[c], w=round(float(w), 3),
                                            recentered=bool(w > 1e-3))
            else:
                self.mu[c] = mus_src.get(c, np.zeros(d)) + gshift
                self.report_[int(c)] = dict(shift_sd=None, n=0, w=0.0, recentered=False)
            # shape: per-class source cov blended to pooled, shrunk to diagonal, ridged
            S = self.alpha * Sw + (1 - self.alpha) * Sc.get(c, Sw)
            S = ((1 - self.shrink) * S + self.shrink * (np.trace(S) / d) * np.eye(d)
                 + self.reg * np.eye(d))
            _, ld = np.linalg.slogdet(S)
            self.Pi.append(np.linalg.inv(S)); self.logdet.append(float(ld))
        self.logdet = np.asarray(self.logdet)
        self.classes_ = np.arange(n_classes)
        return self

    def summary(self):
        """One-line human-readable gate decision for the trainer log."""
        rec, keep = [], []
        for c, r in sorted(self.report_.items()):
            if r["shift_sd"] is None:
                continue
            tag = f"{c}({r['shift_sd']}SD,w={r['w']})"
            (rec if r["recentered"] else keep).append(tag)
        return (f"recentered: {' '.join(rec) or '-'} | "
                f"left@source: {' '.join(keep) or '-'}")

    def _disc(self, X):
        X = np.asarray(X, float)
        Z = np.empty((len(X), len(self.mu)))
        for c in range(len(self.mu)):
            D = X - self.mu[c]
            Z[:, c] = -0.5 * np.einsum("ij,jk,ik->i", D, self.Pi[c], D) \
                      - 0.5 * self.logdet[c]                # uniform priors
        return Z

    def predict(self, X):
        return self._disc(X).argmax(1)

    def predict_proba(self, X):
        Z = self._disc(X); Z -= Z.max(1, keepdims=True)
        e = np.exp(Z); return e / e.sum(1, keepdims=True)


# ── the fit (== online_sim's PCA -> CORAL -> classifier block) ──────────────────
def fit_movement_model(Xtr, ytr, Xcal, clf="lda", norm="coral", pca=0, n_amp=0,
                       ycal=None, n_classes=None, mean_shrink="js"):
    """Optional scaler+PCA fit on train features (Xtr), applied BEFORE CORAL so the
    alignment (and QDA) run in a well-conditioned reduced space; then CORAL-align
    train->calibration and fit the classifier. Returns (model, pretf, info).

    The returned `model` is trained in the calibration (target) domain, so live
    stream features are fed straight to it (no per-chunk CORAL).

    `clf="recalqda"` takes a DIFFERENT route: it ignores `norm` (does its own,
    supervised alignment) and needs the LABELLED target reps — pass `ycal` and
    `n_classes`. It re-centres each class mean on the calib reps, GATED per class by
    `mean_shrink` so a class only moves off the abundant-source mean when its shift
    beats few-rep noise (see RecalQDA). Runs in the same PCA-reduced space as the
    other clfs. Use it when the source->target change is a per-class TRANSLATION (the
    median->ulnar nerve shift) that CORAL, being class-independent, cannot correct.

    `n_amp`: number of TRAILING columns of Xtr to pass through UNREDUCED — they skip
    PCA and are concatenated after the PCs, so a small hand-picked cue (the global
    log-RMS level for the clean head) reaches the classifier undiluted instead of
    competing with ~n_features axes for a PCA slot. n_amp=0 is the original
    scaler+PCA-on-everything path (byte-faithful to online_sim's stim specialist)."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.compose import ColumnTransformer

    classes = np.unique(ytr)
    if len(classes) < 2:
        raise ValueError(
            f"Only {len(classes)} class present in the training windows "
            f"(labels={classes.tolist()}, {len(ytr)} windows). Movement bouts were NOT "
            f"detected from the trigger, so every window fell into 'rest'. This is almost "
            f"always the TRIGGER THRESHOLD: check the logged trigger stats / bout count — "
            f"when the trigger fires alone (no-stim block) its swing is smaller, so the "
            f"stim-derived threshold can miss the pulses. Lower the threshold or pass an "
            f"explicit trig_mid.")

    pre = None
    Ztr, Zcal = Xtr, Xcal
    if pca > 0:
        base_w = Xtr.shape[1] - n_amp                 # feature columns fed to PCA
        scaler_pca = make_pipeline(StandardScaler(), PCA(n_components=min(pca, base_w)))
        if n_amp > 0:
            # PCA the base map; pass the trailing amp column(s) straight through so
            # they land next to the PCs (post-PCA), not inside the reduction.
            pre = ColumnTransformer([
                ("pca", scaler_pca, list(range(base_w))),
                ("amp", "passthrough", list(range(base_w, Xtr.shape[1]))),
            ]).fit(Xtr)
        else:
            pre = scaler_pca.fit(Xtr)
        Ztr, Zcal = pre.transform(Xtr), pre.transform(Xcal)

    # the PCA object, whether standalone or wrapped in a ColumnTransformer (n_amp>0)
    if pre is None:
        pc = None
    elif hasattr(pre, "named_transformers_"):
        pc = pre.named_transformers_["pca"].steps[-1][1]
    else:
        pc = pre.steps[-1][1]
    pca_comps = int(pc.n_components_) if pc is not None else 0
    explained_var = float(pc.explained_variance_ratio_.sum()) if pc is not None else None

    if clf == "recalqda":
        if ycal is None or n_classes is None:
            raise ValueError("clf='recalqda' needs the labelled target reps: pass ycal "
                             "and n_classes")
        ycal = np.asarray(ycal, int)
        model = RecalQDA(mean_shrink=mean_shrink).fit(Ztr, ytr, Zcal, ycal, n_classes)
        # the number that matters here is accuracy on the TARGET (calib) domain, not a
        # source resub; keep the resub_acc key populated with it for the caller's log
        cal_acc = float((model.predict(Zcal) == ycal).mean())
        info = dict(
            n=int(Ztr.shape[0]), d=int(Ztr.shape[1]), n_amp=int(n_amp),
            pca_comps=pca_comps, explained_var=explained_var,
            resub_acc=cal_acc, cal_acc=cal_acc, norm="recal", clf=clf,
            mean_shrink=mean_shrink,
            recal={int(k): v for k, v in model.report_.items()},
        )
        return model, pre, info

    Ztr_a = coral_transform(Ztr, Zcal) if norm == "coral" else Ztr
    model = clf_factory(clf).fit(Ztr_a, ytr)
    info = dict(
        n=int(Ztr_a.shape[0]), d=int(Ztr_a.shape[1]), n_amp=int(n_amp),
        pca_comps=pca_comps, explained_var=explained_var,
        resub_acc=float((model.predict(Ztr_a) == ytr).mean()),
        norm=norm, clf=clf,
    )
    return model, pre, info


# ── trigger-based bout detection (== movement_discrimination.trigger_bouts) ─────
def estimate_trig_mid(trig, k=5.0):
    """Threshold that detects the STIM PULSES, anchored to the idle NOISE SCALE (MAD) rather
    than to the raw min/max extremes.

    The trigger rests near its MEDIAN (idle level) and a stim pulse is a brief excursion to
    one extreme (on this hardware it dips LOW, ~5873, vs an idle ~31090). Because pulses are
    rare (<~4% of samples), a percentile/min-max midpoint lands ON the idle level, so idle
    noise spuriously crosses it -> every window looks "stimulated" and misroutes to the stim
    model.

    Anchoring the threshold to the min/max EXCURSION is spike-fragile: a single big spike
    polluting a swing sets the extreme, dragging the threshold past the real pulse floor so
    genuine pulses stop crossing (missed bouts). Instead we place the threshold `k` robust
    noise units (MAD) off the resting median, on whichever side the pulses go. MAD is
    computed from the idle-dominated signal, so it reflects rest noise and is immune to a
    few outlier spikes. We clamp to the halfway-to-extreme point so we never overshoot the
    pulse floor — this also preserves detection in the reduced-swing no-stim block."""
    trig = np.asarray(trig, float)
    med = float(np.nanmedian(trig))
    # robust extremes (percentiles, not min/max) only to decide the pulse DIRECTION
    lo, hi = float(np.nanpercentile(trig, 1)), float(np.nanpercentile(trig, 99))
    down, up = med - lo, hi - med               # excursion depth on each side of rest
    if max(down, up) < 1e-9:                     # flat trigger -> nothing to detect
        return med
    mad = float(np.nanmedian(np.abs(trig - med)))   # idle noise scale (spike-robust)
    # Guard against an ultra-flat idle (tiny MAD): when the stim COUPLES INTO the trigger
    # during bouts, the operational noise there is >> the idle MAD, so a pure k*mad margin
    # lands the threshold inside the in-bout wobble and every window looks stimulated. Floor
    # the margin at a fraction of the excursion depth so it clears the coupling band, then keep
    # the halfway clamp so we never overshoot the pulse floor.
    if down >= up:                              # pulses dip LOW: threshold below rest
        return max(med - max(k * mad, 0.3 * down), 0.5 * (med + lo))
    return min(med + max(k * mad, 0.3 * up), 0.5 * (med + hi))   # pulses go HIGH: above rest


def trigger_bouts(trig, fs=FS, trig_mid=None, dilate_ms=60, merge_gap_s=1.0,
                  min_bout_s=1.5):
    """Movement/stim bouts = contiguous trigger-active blocks, as (start, end).
    The ~30 Hz pulse transitions dilate into one block per bout, with flat rest in
    between. Sample-accurate boundaries — this is how online_sim labels its bouts."""
    trig = np.asarray(trig, float)
    if trig_mid is None:
        trig_mid = estimate_trig_mid(trig)
    below = (trig < trig_mid).astype(np.int8)
    edges = np.flatnonzero(np.abs(np.diff(below)) == 1) + 1
    if len(edges) == 0:
        return []
    active = np.zeros(len(trig), bool)
    d = int(dilate_ms / 1000 * fs)
    for e in edges:
        active[max(0, e - d):min(len(trig), e + d)] = True
    runs, i, n = [], 0, len(active)
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            runs.append([i, j]); i = j
        else:
            i += 1
    gap = int(merge_gap_s * fs); merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    mn = int(min_bout_s * fs)
    return [(s, e) for s, e in merged if e - s >= mn]


# ── high-level trainer (orchestrates the primitives; train then calibrate) ──────
class MovementTrainer:
    """Server-side train/calibrate, faithful to online_sim's phases:

        train_base(train_raw, class_sequence, class_order)
            -> good-channel mask + per-bout features from the multi-rep recording
        finalize(calib_raw, class_sequence_calib, save_path)
            -> CORAL-align train->calibration, fit classifier, freeze the artifact

    Bouts come from the TRIGGER channel of each RAW recording (trigger_bouts);
    `class_sequence` is the ordered class index per bout (from the session config).
    """

    def __init__(self, fs=FS, win_ms=256, step_ms=64, smooth=0.3, pre=8, post=23,
                 trig_ch=N_EMG, grids=None, clf="qda", norm="coral", pca=30,
                 mean_shrink="js"):
        self.p = dict(fs=fs, win_ms=win_ms, step_ms=step_ms, smooth=smooth, pre=pre,
                      post=post, trig_ch=trig_ch, grids=grids or DEFAULT_GRIDS,
                      clf=clf, norm=norm, pca=pca, mean_shrink=mean_shrink)
        self.base = None            # STIM specialist train features (trigger-segmented)
        self.base_clean = None      # CLEAN specialist train features (cue-segmented)

    def _engine(self, gidx, trig_mid):
        p = self.p
        return OnlineMovementClassifier(gidx, trig_mid, fs=p["fs"], win_ms=p["win_ms"],
                                        step_ms=p["step_ms"], smooth=p["smooth"],
                                        pre=p["pre"], post=p["post"], trig_ch=p["trig_ch"],
                                        grids=p["grids"])

    def _trig(self, raw):
        """The trigger row, with a clear error if the stream doesn't carry it."""
        tc = self.p["trig_ch"]
        if tc >= raw.shape[0]:
            raise ValueError(
                f"trigger channel index {tc} is out of range for the captured stream "
                f"({raw.shape[0]} channels). The guided flow labels movement bouts and "
                f"blanks stim from the trigger channel, so the streamed data MUST include "
                f"it. Replay/stream data that contains the trigger (dai5 npz has it at "
                f"192), or set trig_ch to its actual channel index.")
        return raw[tc]

    def _bouts(self, raw, class_sequence, trig_mid):
        spans = trigger_bouts(self._trig(raw), self.p["fs"], trig_mid)
        n = min(len(spans), len(class_sequence))
        # (start, end, class) to match bout_labeler's unpacking order.
        return [(int(s), int(e), int(class_sequence[i]))
                for i, (s, e) in enumerate(spans[:n])], len(spans)

    def _trig_stats(self, trig, trig_mid):
        """Numeric summary of the trigger channel + the threshold, for logging/UI."""
        trig = np.asarray(trig, float)
        below = trig < trig_mid
        return dict(min=round(float(np.nanmin(trig)), 4), max=round(float(np.nanmax(trig)), 4),
                    p2=round(float(np.nanpercentile(trig, 2)), 4),
                    p98=round(float(np.nanpercentile(trig, 98)), 4),
                    mid=round(float(trig_mid), 4), frac_below=round(float(below.mean()), 4))

    def _segment_and_log(self, raw, class_sequence, trig_mid, tag):
        """Detect trigger bouts and LOG the trigger stats, bout count vs expected cue
        count, and per-bout spans. Returns (bouts, n_spans, trig_stats, spans_s)."""
        st = self._trig_stats(self._trig(raw), trig_mid)
        bouts, n_spans = self._bouts(raw, class_sequence, trig_mid)
        fs = self.p["fs"]
        spans_s = [[round(s / fs, 2), round(e / fs, 2)] for s, e, _ in bouts]
        print(f"[MovementTrainer] {tag}: trigger min={st['min']:.4g} max={st['max']:.4g} "
              f"p2={st['p2']:.4g} p98={st['p98']:.4g} mid={st['mid']:.4g} "
              f"below={st['frac_below']*100:.1f}%")
        print(f"[MovementTrainer] {tag}: {n_spans} trigger bouts detected "
              f"(expected {len(class_sequence)} from the cue sequence); spans[s]={spans_s}")
        if n_spans == 0:
            print(f"[MovementTrainer] {tag}: WARNING — 0 bouts. The trigger never crossed "
                  f"the threshold; its swing may be too small (no-stim block). Check stats above.")
        elif n_spans != len(class_sequence):
            print(f"[MovementTrainer] {tag}: WARNING — bout count {n_spans} != cue count "
                  f"{len(class_sequence)}; labels may be misaligned.")
        return bouts, n_spans, st, spans_s

    def train_base(self, train_raw, class_sequence, class_order):
        raw = np.asarray(train_raw)
        gidx = good_gidx(raw, self.p["grids"], self.p["fs"])
        trig_mid = estimate_trig_mid(self._trig(raw))
        bouts, n_spans, st, spans_s = self._segment_and_log(raw, class_sequence, trig_mid, "STIM train")
        rest = len(class_order) - 1
        # If the stim block includes stimulated-REST bouts (a rest label appears among the
        # trigger bouts, not just as the between-bout fallback), the stim head's rest class
        # comes from those BLANKED windows — so train it on blanked windows only, matching
        # inference (unblanked between-bout rest never reaches the stim head live). Without
        # stim-rest bouts, keep every window (legacy: rest from the unblanked gaps).
        has_stim_rest = rest in list(class_sequence)
        eng = self._engine(gidx, trig_mid)
        Xtr, ytr = collect_features(eng, raw, 0, raw.shape[1], bout_labeler(bouts, rest),
                                    blanked_only=has_stim_rest)
        cc = np.bincount(ytr, minlength=len(class_order)).tolist()
        print(f"[MovementTrainer] STIM train: {len(gidx)} good ch, {Xtr.shape[0]} windows, "
              f"{'blanked-only ' if has_stim_rest else ''}"
              f"class counts {dict(zip(list(class_order), cc))}")
        self.base = dict(Xtr=Xtr, ytr=ytr, gidx=gidx, class_order=list(class_order))
        return dict(phase="stim", n_good=int(len(gidx)), n_bouts=len(bouts),
                    n_spans_detected=n_spans, n_windows=int(Xtr.shape[0]), class_counts=cc,
                    class_order=list(class_order), trigger=st, spans_s=spans_s)

    def train_clean(self, train_raw, class_sequence, class_order=None):
        """Train the CLEAN (nostim) specialist from a NO-stim recording where the
        patient receives no stimulation but the stimulator still drives the trigger
        (its ch3 is wired to the trigger input, not the patient). So bouts are
        trigger-segmented exactly like the stim block — BUT blanking is disabled (the
        EMG carries no stim artifact), via the feature engine's trig_mid=-inf, so the
        features are clean. Segmentation uses the real trigger; only blanking is off.
        Shares the stim specialist's good-channel mask when one exists (same montage)."""
        raw = np.asarray(train_raw)
        gidx = self.base["gidx"] if self.base else good_gidx(raw, self.p["grids"], self.p["fs"])
        class_order = list(class_order or (self.base["class_order"] if self.base else []))
        if not class_order:
            raise ValueError("train_clean needs class_order (or a prior train_base)")
        trig_mid = estimate_trig_mid(self._trig(raw))   # REAL trigger -> segmentation only
        bouts, n_spans, st, spans_s = self._segment_and_log(raw, class_sequence, trig_mid, "NOSTIM train")
        rest = len(class_order) - 1
        eng = self._engine(gidx, float("-inf"))         # NO blanking -> clean features
        # Keep the global log-RMS level as a trailing feature for the clean head only.
        Xtr, ytr = collect_features(eng, raw, 0, raw.shape[1], bout_labeler(bouts, rest),
                                    with_amp=True)
        cc = np.bincount(ytr, minlength=len(class_order)).tolist()
        print(f"[MovementTrainer] NOSTIM train: {Xtr.shape[0]} windows, "
              f"class counts {dict(zip(class_order, cc))}")
        self.base_clean = dict(Xtr=Xtr, ytr=ytr, gidx=gidx, class_order=class_order)
        return dict(phase="nostim", n_bouts=len(bouts), n_spans_detected=n_spans,
                    n_windows=int(Xtr.shape[0]), class_counts=cc, class_order=class_order,
                    trigger=st, spans_s=spans_s)

    def finalize(self, calib_raw, class_sequence_calib, save_path):
        if self.base is None:
            raise RuntimeError("call train_base before finalize")
        raw = np.asarray(calib_raw)
        gidx = self.base["gidx"]
        class_order = self.base["class_order"]
        trig_mid = estimate_trig_mid(self._trig(raw))
        bouts, n_spans, st, spans_s = self._segment_and_log(raw, class_sequence_calib, trig_mid, "CALIB")
        rest = len(class_order) - 1
        # If calibration includes a stimulated-REST bout, align the stim head to its own
        # regime: blanked windows only (as in train_base), so its rest class is CORAL-
        # aligned to STIMULATED rest, not unstimulated between-bout gaps it never sees live.
        has_stim_rest = rest in list(class_sequence_calib)
        eng = self._engine(gidx, trig_mid)
        Xcal, ycal = collect_features(eng, raw, 0, raw.shape[1], bout_labeler(bouts, rest),
                                      blanked_only=has_stim_rest)
        recal = self.p["clf"] == "recalqda"
        norm_lbl = f"gate={self.p['mean_shrink']}" if recal else f"norm={self.p['norm']}"
        print(f"[MovementTrainer] CALIB: {Xcal.shape[0]} windows{' (blanked-only)' if has_stim_rest else ''}; "
              f"fitting {self.p['clf']} ({norm_lbl}, pca={self.p['pca']}) stim specialist ...")
        # recalqda re-centres the class means on the LABELLED calib reps (ycal), gated per
        # class; the other clfs align via CORAL and never use ycal.
        model, pre, info = fit_movement_model(self.base["Xtr"], self.base["ytr"], Xcal,
                                              clf=self.p["clf"], norm=self.p["norm"],
                                              pca=self.p["pca"], ycal=ycal,
                                              n_classes=len(class_order),
                                              mean_shrink=self.p["mean_shrink"])
        acc_lbl = "cal acc" if recal else "resub acc"
        print(f"[MovementTrainer] stim specialist fit: {acc_lbl}={info['resub_acc']:.3f}")
        if recal:
            print(f"[MovementTrainer] recal gate -> {model.summary()}")

        # CLEAN specialist (two-stage): if a no-stim block was captured, fit it too and
        # freeze it into the SAME artifact as model_clean. It self-calibrates on its own
        # clean features (the guided calib recording is under stim, so it can't serve the
        # clean model) -> CORAL is identity there, hence norm="none".
        model_clean = pretf_clean = None
        info_c = {}
        if self.base_clean is not None:
            # the clean head self-calibrates on its own clean features (Xc == Xtr): there
            # is no source->target shift to recentre, so recalqda would be a no-op with no
            # labelled target set — fall back to plain qda in the same PCA space.
            clean_clf = "qda" if self.p["clf"] == "recalqda" else self.p["clf"]
            print(f"[MovementTrainer] fitting CLEAN (nostim) specialist "
                  f"({self.base_clean['Xtr'].shape[0]} windows) ...")
            model_clean, pretf_clean, info_c = fit_movement_model(
                self.base_clean["Xtr"], self.base_clean["ytr"], self.base_clean["Xtr"],
                clf=clean_clf, norm="none", pca=self.p["pca"], n_amp=1)
            print(f"[MovementTrainer] clean specialist fit: resub acc={info_c['resub_acc']:.3f}")
        else:
            print("[MovementTrainer] no no-stim block captured -> single-stage model "
                  "(stim only, no clean specialist)")

        p = self.p
        save_artifact(save_path, gidx=gidx, trig_mid=trig_mid, model=model, pretf=pre,
                      model_clean=model_clean, pretf_clean=pretf_clean,
                      clean_amp=model_clean is not None,
                      class_order=class_order, fs=p["fs"], win_ms=p["win_ms"],
                      step_ms=p["step_ms"], smooth=p["smooth"], pre=p["pre"], post=p["post"],
                      trig_ch=p["trig_ch"], grids=p["grids"],
                      meta=dict(pca=p["pca"], two_stage=model_clean is not None,
                                clean=info_c, **info))   # info already has clf/norm
        print(f"[MovementTrainer] artifact saved -> {save_path} "
              f"(two_stage={model_clean is not None})")
        return dict(phase="calib", save_path=save_path, n_calib_bouts=len(bouts),
                    n_calib_windows=int(Xcal.shape[0]), two_stage=model_clean is not None,
                    trigger=st, spans_s=spans_s, clean_resub=info_c.get("resub_acc"), **info)
