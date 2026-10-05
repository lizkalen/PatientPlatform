"""Corrected convolutive-kernel-compensation MU decomposition + stim-condition recovery.

Productionized from the diagnosis in
``processing/scripts/movement_disc_1307/mu_xclean/diagnosis`` (DIAGNOSIS.md / METHODS.md /
TRACKING.md / CHANGES.md). The shipped MUAP/CBSS path returns ~0 physiologically-real motor
units on the exp13072026 HD-sEMG because of four defects: an inherited good_mask keeps bad
electrodes; no common-mode removal (one global component holds ~60 % of variance); the extended
covariance is near-singular and the default whitening ridge (~1e-8) amplifies the noise subspace;
and (gСКС) the random-init deflation is dead code. This module fixes all four and adds a stim
comb-notch path so median/both recordings (which PARRM/ERAASR ruin) decompose from the raw signal.

Validated: 0 -> 9-16 real MUs per movement (nostim), median 0 -> 3-7 and both 0 -> 1-4 (stim, from
raw + comb-notch), all passing the reproducibility yardstick + a stim-phase-lock surrogate, with
phase-randomized surrogates giving 0. Motor units track across stim conditions by spatial footprint
(close<->trp cross-transfer RoA 0.75-0.80).

Public API:
    bad_channels(sig)                          -> indices of outlier electrodes
    mu_validity(sig, spikes, fs)               -> dict of reproducibility/physiology scores
    hardened_validity(v)                       -> bool acceptance gate
    decompose(sig, fs, ...)                    -> (units, keep, dropped, n_kept_dims, X)
    comb_notch(sig, fs, f0, half_bw)           -> FFT comb-notch of a stim fundamental
    recover_stim_units(sig, pulses, fs, ...)   -> real MUs from a stim recording (bypasses PARRM/ERAASR)
"""
from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis
from scipy.signal import welch, find_peaks

from .core import extension, est_spike_times


# --------------------------------------------------------------------------- #
# Channel / preprocessing                                                     #
# --------------------------------------------------------------------------- #
def muap_band_bad_channels(sig_bp, fs, mu_lo=60.0, mu_hi=250.0, band_lo=20.0, band_hi=500.0,
                           thr=0.55):
    """Automatic, dataset-agnostic channel-quality rule: flag channels whose fraction of EMG-band
    (band_lo..band_hi Hz) power in the MUAP band (mu_lo..mu_hi Hz) is below ``thr``. Bad electrode
    contacts spread power out of the MU band (excess low-frequency / broadband), so this drops them
    while keeping true EMG channels — reproducibly, with the SAME threshold across datasets.
    Validated: on the experlangen montage it auto-drops the two bad-contact grids (77 % of them,
    6 % of the good grids); on exp13072026 it drops ~2 % of the reviewed-good channels. ``sig_bp``
    must be band-passed. Returns bad-channel indices. Combine with bad_channels() for saturation."""
    from scipy.signal import welch
    f, P = welch(sig_bp, fs=fs, nperseg=min(2048, sig_bp.shape[1]), axis=1)
    tot = P[:, (f >= band_lo) & (f < band_hi)].sum(1) + 1e-12
    mu = P[:, (f >= mu_lo) & (f < mu_hi)].sum(1) / tot
    return np.where(mu < thr)[0]


def auto_muap_band_bad_channels(sig_bp, fs, mu_lo=60.0, mu_hi=250.0, band_lo=20.0, band_hi=500.0,
                                floor=0.35, min_sep=0.15, min_low_frac=0.05, max_low_frac=0.7):
    """Automatic, self-tuning version of muap_band_bad_channels: instead of a fixed threshold, pick
    the cut from THIS recording's own MUAP-band distribution by Otsu (maximum between-class
    variance), and apply it ONLY if the distribution is genuinely bimodal (a real low-quality
    cluster — separation between the two cluster means >= min_sep and the low cluster is
    min_low_frac..max_low_frac of channels). Otherwise fall back to ``floor`` (drop ~nothing).
    This is the SAME method for every experiment; the threshold VALUE is data-driven, so it
    auto-removes bad-contact grids where they exist and leaves clean montages untouched.
    Validated: experlangen -> bimodal, auto-threshold ~0.62 drops the two bad grids; exp13072026 ->
    unimodal, falls back to floor and drops ~0. Returns (bad_indices, threshold_used, was_bimodal)."""
    from scipy.signal import welch
    f, P = welch(sig_bp, fs=fs, nperseg=min(2048, sig_bp.shape[1]), axis=1)
    tot = P[:, (f >= band_lo) & (f < band_hi)].sum(1) + 1e-12
    mu = P[:, (f >= mu_lo) & (f < mu_hi)].sum(1) / tot
    n = mu.size
    best_t, best_var = floor, -1.0
    for t in np.linspace(0.3, 0.75, 46):
        lo, hi = mu[mu < t], mu[mu >= t]
        if lo.size < 3 or hi.size < 3:
            continue
        w0, w1 = lo.size / n, hi.size / n
        var = w0 * w1 * (lo.mean() - hi.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, t
    lo, hi = mu[mu < best_t], mu[mu >= best_t]
    sep = (hi.mean() - lo.mean()) if lo.size >= 3 and hi.size >= 3 else 0.0
    frac_low = float(np.mean(mu < best_t))
    bimodal = sep >= min_sep and min_low_frac < frac_low < max_low_frac
    thr = best_t if bimodal else floor
    return np.where(mu < thr)[0], float(thr), bool(bimodal)


def bad_channels(sig, hi=4.0, lo=0.05, kurt_th=20.0):
    """Outlier electrodes: RMS too hot / ~flat, or high-kurtosis (electrode pops / residual
    stim spikes). Pass kurt_th=None to gate on RMS only (recommended for stim recordings after
    comb-notch, where residual sharp transients would over-trip the kurtosis test)."""
    rms = np.sqrt((sig ** 2).mean(1)); med = np.median(rms)
    bad = np.where((rms > hi * med) | (rms < lo * med))[0]
    if kurt_th is not None:
        bad = np.union1d(bad, np.where(kurtosis(sig, axis=1) > kurt_th)[0])
    return bad


def preprocess(sig, k_common_mode=2, drop=None, kurt_gate=True):
    """Drop bad channels, remove top-K common-mode PCs (SVD over channels), per-channel z-score."""
    dropped = bad_channels(sig, kurt_th=20.0 if kurt_gate else None) if drop is None else np.asarray(drop, int)
    keep = np.setdiff1d(np.arange(sig.shape[0]), dropped)
    X = sig[keep].astype(float)
    X -= X.mean(1, keepdims=True)
    if k_common_mode > 0:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        X = X - U[:, :k_common_mode] @ (U[:, :k_common_mode].T @ X)
    X /= (X.std(1, keepdims=True) + 1e-9)
    return X, keep, dropped


def _extend_edge(X, R):
    e = extension(X, R); e -= e.mean(1, keepdims=True)
    e[:, :R * 2] = 0; e[:, -R * 2:] = 0
    return e


def whiten_truncated(ext, tol_frac=1e-3):
    """Rank-truncated ZCA/PCA whitening: keep only eigen-components with eigenvalue
    > tol_frac * max (drop the near-null noise subspace). Returns (white, Wm, n_kept)."""
    C = (ext @ ext.T) / ext.shape[1]
    w, V = np.linalg.eigh(C); w = w[::-1]; V = V[:, ::-1]
    Kk = int((w > tol_frac * w[0]).sum())
    Wm = (V[:, :Kk] / np.sqrt(w[:Kk])).T
    return Wm @ ext, Wm, Kk


def fastica_deflation(white, n_src=60, n_iter=80, seed=0):
    """FastICA fixed-point with Gram-Schmidt deflation, skewness contrast g=x^2 (rewards the
    sparse one-sided pulse trains of motor units)."""
    rng = np.random.default_rng(seed)
    d = white.shape[0]; filt = []
    for _ in range(n_src):
        w = rng.standard_normal(d); w /= np.linalg.norm(w) + 1e-12
        for _ in range(n_iter):
            wl = w.copy(); s = w @ white
            w = (white * (s ** 2)).mean(1) - 2 * s.mean() * w
            for wp in filt:
                w = w - (w @ wp) * wp
            w /= np.linalg.norm(w) + 1e-12
            if abs(abs(w @ wl) - 1) < 1e-4:
                break
        filt.append(w)
    return np.array(filt)


# --------------------------------------------------------------------------- #
# Validity yardstick (reproducibility + physiology, not a tunable SIL cutoff)  #
# --------------------------------------------------------------------------- #
def _sta(sig, spikes, half):
    n_ch, T = sig.shape
    sp = np.asarray(spikes, int); sp = sp[(sp >= half) & (sp + half + 1 <= T)]
    if sp.size == 0:
        return np.zeros((n_ch, 2 * half + 1)), 0
    acc = np.zeros((n_ch, 2 * half + 1))
    for s in sp:
        acc += sig[:, s - half:s + half + 1]
    return acc / sp.size, sp.size


def mu_validity(sig, spikes, fs, half_ms=10.0, top_m=16):
    """Split-half STA reproducibility + spatial focality + MUAP shape + refractoriness."""
    half = int(half_ms / 1000 * fs); T = sig.shape[1]
    sp = np.sort(np.asarray(spikes, int)); sp = sp[(sp >= half) & (sp + half + 1 <= T)]
    out = dict(n=int(sp.size), split_half_r=np.nan, focality_frac=np.nan, n_focal_ch=0,
               peak_floor=np.nan, wf_snr=np.nan, biphasic=np.nan, defl_ms=np.nan,
               refrac_viol=np.nan, dom=-1, real=False)
    if sp.size < 20:
        return out
    full, _ = _sta(sig, sp, half)
    p2p = full.max(1) - full.min(1); dom = int(np.argmax(p2p))
    topch = np.argsort(p2p)[::-1][:top_m]
    se, _ = _sta(sig, sp[0::2], half); so, _ = _sta(sig, sp[1::2], half)
    ve, vo = se[topch].ravel(), so[topch].ravel()
    out["split_half_r"] = float(np.corrcoef(ve, vo)[0, 1]) if ve.std() > 0 and vo.std() > 0 else 0.0
    out["focality_frac"] = float(np.mean(p2p > 0.5 * p2p.max()))
    out["n_focal_ch"] = int(np.sum(p2p > 0.5 * p2p.max()))
    out["peak_floor"] = float(p2p.max() / (np.median(p2p) + 1e-12))
    edge = np.r_[full[:, :half // 2], full[:, -half // 2:]]
    out["wf_snr"] = float((full[dom].max() - full[dom].min()) / (np.median(edge.std(1)) + 1e-12))
    wf = full[dom]; lo_i, hi_i = wf.argmin(), wf.argmax()
    out["biphasic"] = float(min(abs(wf[lo_i]), abs(wf[hi_i])) / (max(abs(wf[lo_i]), abs(wf[hi_i])) + 1e-12))
    out["defl_ms"] = float(abs(hi_i - lo_i) / fs * 1000)
    isi = np.diff(sp) / fs
    out["refrac_viol"] = float(np.mean(isi < 0.02))
    out["dom"] = dom
    out["real"] = bool(out["split_half_r"] > 0.6 and out["wf_snr"] > 5.0
                       and out["focality_frac"] < 0.12 and out["refrac_viol"] < 0.10)
    return out


def hardened_validity(v):
    """Two-sided spatial + MUAP-shape gate (rejects single-channel artifacts and common-mode crests)."""
    return bool(v["split_half_r"] > 0.6 and v["wf_snr"] > 6.0 and 3 <= v["n_focal_ch"]
                and v["focality_frac"] < 0.12 and v["peak_floor"] > 6.0
                and v["biphasic"] > 0.25 and v["defl_ms"] < 6.0 and v["refrac_viol"] < 0.10)


# --------------------------------------------------------------------------- #
# Decomposition                                                               #
# --------------------------------------------------------------------------- #
def _dedup(units, fs, coincidence_th=0.3):
    order = sorted(range(len(units)), key=lambda i: -units[i]["wf_snr"])
    kept = []; tol = int(0.005 * fs)
    for i in order:
        a = np.sort(units[i]["spikes"]); dup = False
        for k in kept:
            b = np.sort(units[k]["spikes"]); j = m = 0
            for t in a:
                while j < len(b) and b[j] < t - tol:
                    j += 1
                if j < len(b) and abs(b[j] - t) <= tol:
                    m += 1
            if m / max(len(a), len(b)) > coincidence_th:
                dup = True; break
        if not dup:
            kept.append(i)
    return [units[i] for i in kept]


def decompose(sig, fs, R=8, k_common_mode=2, whiten_tol=1e-3, n_src=60, seed=0,
              drop=None, kurt_gate=True):
    """Corrected CKC decomposition of one preprocessed window. Returns
    (units, keep, dropped, n_kept_dims, X). Each unit passes hardened_validity and carries its
    spikes, dominant channel, and all validity scores."""
    X, keep, dropped = preprocess(sig, k_common_mode, drop=drop, kurt_gate=kurt_gate)
    ext = _extend_edge(X, R)
    white, Wm, Kk = whiten_truncated(ext, whiten_tol)
    filters = fastica_deflation(white, n_src=n_src, seed=seed)
    units = []
    for w in filters:
        s = w @ white
        s = s * np.sign(np.mean(s ** 3) + 1e-12)
        spk = np.asarray(est_spike_times(s, fs, cluster="kmeans")[0], int)
        if spk.size < 20:
            continue
        v = mu_validity(X, spk, fs)
        if hardened_validity(v):
            v["spikes"] = spk; v["keep"] = keep
            units.append(v)
    return _dedup(units, fs), keep, dropped, Kk, X


# --------------------------------------------------------------------------- #
# Stimulation-condition recovery (bypass PARRM/ERAASR)                        #
# --------------------------------------------------------------------------- #
def estimate_f0(sig, fs, lo=29.0, hi=32.0):
    f, P = welch(sig, fs=fs, nperseg=int(min(8 * fs, sig.shape[1])), axis=1)
    Pm = P.mean(0); b = (f >= lo) & (f <= hi)
    return float(f[b][np.argmax(Pm[b])])


def comb_notch(sig, fs, f0, half_bw=0.5):
    """Zero-phase FFT notch of f0 and all harmonics to Nyquist (+/- half_bw Hz)."""
    T = sig.shape[1]; F = np.fft.rfft(sig, axis=1); fr = np.fft.rfftfreq(T, 1 / fs)
    kill = np.zeros(fr.shape, bool)
    for h in range(1, int((fs / 2) / f0) + 1):
        kill |= np.abs(fr - h * f0) <= half_bw
    F[:, kill] = 0
    return np.fft.irfft(F, n=T, axis=1)


def comb_power_fraction(sig, fs, f0, half_bw=0.6, lo=20.0, hi=500.0):
    """Fraction of lo..hi Hz power sitting on the f0 comb. Compare against comb_chance_fraction:
    an unrelated signal scores the width fraction the comb bands occupy, not zero."""
    x = np.atleast_2d(sig)
    n = x.shape[-1]
    fr = np.fft.rfftfreq(n, 1 / fs)
    P = (np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) ** 2).mean(0)
    band = (fr >= lo) & (fr <= hi)
    kill = np.zeros(fr.shape, bool)
    for h in range(1, int(hi / f0) + 1):
        kill |= np.abs(fr - h * f0) <= half_bw
    return float(P[kill & band].sum() / (P[band].sum() + 1e-30))


def comb_chance_fraction(f0, half_bw=0.6, lo=20.0, hi=500.0):
    """Comb power fraction expected from an unrelated signal (pure bandwidth bookkeeping)."""
    return min(1.0, int(hi / f0) * 2 * half_bw / (hi - lo))


def strongest_comb(sig, fs, lo=24.0, hi=36.0, step=0.05, hi_harm=500.0, half_bw=0.6):
    """Fundamental in lo..hi whose harmonic series carries the most power. Unlike estimate_f0
    (which takes the single largest spectral peak in a narrow 29-32 Hz window) this scores the whole
    comb, so it is not fooled when the fundamental is weaker than a harmonic or sits outside 29-32."""
    x = np.atleast_2d(sig)
    n = x.shape[-1]
    fr = np.fft.rfftfreq(n, 1 / fs)
    P = (np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) ** 2).mean(0)
    cand = np.arange(lo, hi, step)
    sc = np.empty(cand.size)
    for i, f0 in enumerate(cand):
        s = 0.0
        for h in range(1, int(hi_harm / f0) + 1):
            s += P[np.abs(fr - h * f0) <= half_bw].sum()
        sc[i] = s
    return float(cand[int(np.argmax(sc))])


def multi_comb_notch(sig, fs, max_combs=4, margin=1.5, lo=24.0, hi=36.0, half_bw=0.5):
    """Iteratively notch the strongest remaining comb until its power reaches the chance level.

    A single-fundamental notch is not enough on these recordings: after removing the nominal
    fundamental the dominant remaining comb can sit ~1 Hz away and still carry several times chance
    power (measured on experlangen: 0.229 of 20-500 Hz power at 31.40 Hz after notching 30.375 Hz,
    against a 0.0375 chance baseline). Causes include a non-integer stim period, a drifting rate,
    and genuinely multiple fundamentals. Returns (cleaned_signal, [fundamentals_removed])."""
    out = np.asarray(sig, float)
    used = []
    for _ in range(max_combs):
        f0 = strongest_comb(out, fs, lo=lo, hi=hi)
        if comb_power_fraction(out, fs, f0) < margin * comb_chance_fraction(f0):
            break
        out = comb_notch(out, fs, f0, half_bw)
        used.append(f0)
    return out, used


def stim_phase_lock(spikes, fs, f0, n_null=2000, seed=0, null="isi"):
    """Rayleigh resultant of discharge phases wrt the stim fundamental (continuous f0 handles a
    non-integer period), tested against a null. p > 0.05 => NOT stim-locked (a real MU).

    ``null='isi'`` (default) draws surrogates by CIRCULARLY SHIFTING the spike train, which
    preserves its inter-spike-interval structure exactly and destroys only the phase alignment.
    ``null='uniform'`` is the original: surrogate phases drawn uniformly, i.e. a null of a POISSON
    train. That is wrong for motor units. A regular discharge (CoV of ISI ~0.1-0.2) concentrates
    phase whenever its rate is commensurate with f0 for reasons that have nothing to do with the
    stimulus, so the uniform null calls regular real units "stim-locked". Measured on synthetic
    trains that are provably not stim-driven: the uniform null rejected 4/14, including 2 of 4
    otherwise-valid planted units. Keep 'uniform' only to reproduce older numbers."""
    rng = np.random.default_rng(seed)
    sp = np.asarray(spikes)
    R = float(np.abs(np.exp(2j * np.pi * ((sp / fs * f0) % 1.0)).mean()))
    if null == "uniform":
        nul = np.array([np.abs(np.exp(2j * np.pi * rng.uniform(0, 1, sp.size)).mean())
                        for _ in range(n_null)])
    else:
        span = int(sp.max() - sp.min()) + 1 if sp.size else 1
        lo = int(0.05 * fs)
        nul = np.empty(n_null)
        for i in range(n_null):
            shifted = sp + rng.integers(lo, max(lo + 1, span))
            nul[i] = np.abs(np.exp(2j * np.pi * ((shifted / fs * f0) % 1.0)).mean())
    return R, float(np.mean(nul >= R))


def recover_stim_units(sig, pulses, fs, half_bw=0.5, seeds=(0, 1), require_unlocked=True, R=8):
    """Recover real MUs from a stim recording. ``sig`` = notch+band-pass good-channel signal with
    the stim artifact INTACT (do NOT feed PARRM/ERAASR output -- it destroys MUs on median/both).
    Comb-notches the stim fundamental, decomposes with an RMS-only channel gate, pools seeds, and
    keeps units that pass hardened_validity AND are not stim-phase-locked. Returns (units, f0)."""
    pulses = np.asarray(pulses)
    f0 = estimate_f0(sig, fs) if pulses.size else 0.0
    x = comb_notch(sig, fs, f0, half_bw) if f0 else sig
    drop = bad_channels(x, kurt_th=None) if pulses.size else None
    pool = []
    for sd in seeds:
        units, keep, dropped, Kk, X = decompose(x, fs, R=R, seed=sd, drop=drop, kurt_gate=False)
        for u in units:
            if require_unlocked and pulses.size:
                Rr, p = stim_phase_lock(u["spikes"], fs, f0)
                u["phase_R"] = Rr; u["phase_p"] = p
                if p <= 0.05:
                    continue
            pool.append(u)
    return _dedup(pool, fs), f0
