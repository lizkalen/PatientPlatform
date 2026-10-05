"""Channel selection and common-mode rejection for multi-channel EMG.

Independent of any decomposition algorithm. The functions take a
``(n_channels, n_samples)`` array and return either channel indices or a
cleaned array, so they can be placed in front of CBSS, CKC, or any other
decomposer.

Two families of channel criteria are provided:

- amplitude: flags channels whose RMS is far from the array median, or whose
  kurtosis is high (electrode pops, residual stimulation transients).
- MUAP band: flags channels that carry a low fraction of their EMG-band power
  inside the MUAP band. Requires a band-passed signal; ``select_channels``
  band-passes a copy for this purpose when ``criterion_bandpass`` is set.

The MUAP-band rule comes in a fixed-threshold form and a self-tuning form. The
self-tuning form picks the cut from the recording's own distribution by Otsu
and applies it only when that distribution is bimodal, otherwise it falls back
to a permissive floor.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis


def bandpass(sig, fs, lo=20.0, hi=500.0, order=2):
    """Zero-phase Butterworth band-pass.

    Args:
        sig (ndarray): Signal, (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        lo (float): High-pass corner in Hz.
        hi (float): Low-pass corner in Hz. Clipped to just below Nyquist.
        order (int): Filter order.

    Returns:
        ndarray: Filtered signal, same shape as ``sig``.
    """
    hi = min(hi, 0.99 * fs / 2.0)
    b, a = butter(order, [lo, hi], fs=fs, btype="band")
    return filtfilt(b, a, np.asarray(sig, float), axis=1)


def amplitude_bad_channels(sig, hi=4.0, lo=0.05, kurt_th=20.0):
    """Flag channels by RMS relative to the array median, and by kurtosis.

    A channel is flagged when its RMS exceeds ``hi`` times the median RMS, falls
    below ``lo`` times the median RMS, or its kurtosis exceeds ``kurt_th``.

    Args:
        sig (ndarray): Signal, (n_channels, n_samples).
        hi (float): Upper RMS multiplier.
        lo (float): Lower RMS multiplier.
        kurt_th (float or None): Kurtosis cut. Pass None to skip the kurtosis
            test. Set it to None on stimulation recordings after comb-notching,
            where residual sharp transients raise kurtosis on every channel.

    Returns:
        ndarray: Sorted indices of flagged channels.
    """
    sig = np.asarray(sig, float)
    rms = np.sqrt((sig ** 2).mean(1))
    med = np.median(rms)
    bad = np.where((rms > hi * med) | (rms < lo * med))[0]
    if kurt_th is not None:
        bad = np.union1d(bad, np.where(kurtosis(sig, axis=1) > kurt_th)[0])
    return np.asarray(bad, int)


def amplitude_breakdown(sig, hi=4.0, lo=0.05, kurt_th=20.0):
    """Same tests as :func:`amplitude_bad_channels`, reported one by one.

    Use this to see which sub-test is responsible when a lot of channels are
    dropped. On a stimulation recording the kurtosis test is the usual answer:
    residual artifact transients raise kurtosis across the whole array, so the
    test stops measuring contact quality.

    Args:
        sig (ndarray): Signal, (n_channels, n_samples).
        hi (float): Upper RMS multiplier.
        lo (float): Lower RMS multiplier.
        kurt_th (float or None): Kurtosis cut. None reports an empty list for it.

    Returns:
        dict: ``rms_high``, ``rms_low``, ``kurtosis`` index arrays, plus the
        ``rms_ratio`` (rms over the array median) and ``kurt`` values per channel.
    """
    sig = np.asarray(sig, float)
    rms = np.sqrt((sig ** 2).mean(1))
    med = np.median(rms)
    ratio = rms / (med + 1e-12)
    kurt = kurtosis(sig, axis=1)
    return {
        "rms_high": np.where(ratio > hi)[0],
        "rms_low": np.where(ratio < lo)[0],
        "kurtosis": np.where(kurt > kurt_th)[0] if kurt_th is not None else np.array([], int),
        "rms_ratio": ratio,
        "kurt": kurt,
    }


def muap_band_ratio(sig, fs, mu_band=(60.0, 250.0), emg_band=(20.0, 500.0), nperseg=2048):
    """Per-channel fraction of EMG-band power that sits in the MUAP band.

    Args:
        sig (ndarray): Band-passed signal, (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        mu_band (tuple): (low, high) of the MUAP band in Hz.
        emg_band (tuple): (low, high) of the reference EMG band in Hz.
        nperseg (int): Welch segment length, capped at the signal length.

    Returns:
        ndarray: Ratio per channel, shape (n_channels,), in [0, 1].
    """
    sig = np.asarray(sig, float)
    f, P = welch(sig, fs=fs, nperseg=min(nperseg, sig.shape[1]), axis=1)
    tot = P[:, (f >= emg_band[0]) & (f < emg_band[1])].sum(1) + 1e-12
    return P[:, (f >= mu_band[0]) & (f < mu_band[1])].sum(1) / tot


def muap_band_bad_channels(sig, fs, thr=0.55, **kwargs):
    """Flag channels whose MUAP-band ratio is below a fixed threshold.

    Args:
        sig (ndarray): Band-passed signal, (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        thr (float): Ratio below which a channel is flagged.
        **kwargs: Passed to :func:`muap_band_ratio` (``mu_band``, ``emg_band``,
            ``nperseg``).

    Returns:
        ndarray: Sorted indices of flagged channels.
    """
    ratio = muap_band_ratio(sig, fs, **kwargs)
    return np.where(ratio < thr)[0]


def auto_muap_band_bad_channels(sig, fs, floor=0.35, min_sep=0.15, min_low_frac=0.05,
                                max_low_frac=0.7, search=(0.30, 0.75, 46), **kwargs):
    """Flag channels by a MUAP-band threshold derived from this recording.

    Scans candidate thresholds and keeps the one with maximum between-class
    variance (Otsu). That threshold is used only when the ratio distribution is
    bimodal, defined as: the gap between the two class means is at least
    ``min_sep``, and the low class holds between ``min_low_frac`` and
    ``max_low_frac`` of channels. Otherwise ``floor`` is used, which flags
    close to nothing on a clean montage.

    Args:
        sig (ndarray): Band-passed signal, (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        floor (float): Fallback threshold used when the distribution is not
            bimodal.
        min_sep (float): Minimum separation between the two class means.
        min_low_frac (float): Minimum fraction of channels in the low class.
        max_low_frac (float): Maximum fraction of channels in the low class.
        search (tuple): (start, stop, num) of the candidate threshold grid.
        **kwargs: Passed to :func:`muap_band_ratio`.

    Returns:
        tuple:
            - ndarray: Sorted indices of flagged channels.
            - float: Threshold applied.
            - bool: Whether the distribution was judged bimodal.
    """
    ratio = muap_band_ratio(sig, fs, **kwargs)
    n = ratio.size
    best_t, best_var = float(floor), -1.0
    for t in np.linspace(*search):
        low, high = ratio[ratio < t], ratio[ratio >= t]
        if low.size < 3 or high.size < 3:
            continue
        w0, w1 = low.size / n, high.size / n
        var = w0 * w1 * (low.mean() - high.mean()) ** 2
        if var > best_var:
            best_var, best_t = var, float(t)

    low, high = ratio[ratio < best_t], ratio[ratio >= best_t]
    sep = (high.mean() - low.mean()) if low.size >= 3 and high.size >= 3 else 0.0
    frac_low = float(np.mean(ratio < best_t))
    bimodal = bool(sep >= min_sep and min_low_frac < frac_low < max_low_frac)
    thr = best_t if bimodal else float(floor)
    return np.where(ratio < thr)[0], thr, bimodal


def select_channels(sig, fs, criteria=("amplitude", "auto_muap_band"),
                    criterion_bandpass=(20.0, 500.0), drop=None,
                    amplitude_kwargs=None, muap_band_kwargs=None):
    """Apply channel criteria and return the channels to keep.

    Criteria are combined by union: a channel is dropped if any active rule
    flags it. MUAP-band criteria are computed on a band-passed copy of ``sig``
    when ``criterion_bandpass`` is set; the returned indices refer to ``sig``
    and no filtering is applied to the data itself.

    Args:
        sig (ndarray): Signal, (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        criteria (tuple): Any of "amplitude", "muap_band", "auto_muap_band".
            An empty tuple keeps every channel.
        criterion_bandpass (tuple or None): (low, high) in Hz used to filter the
            copy that MUAP-band criteria are computed on. Pass None when ``sig``
            is already band-passed.
        drop (sequence or None): Explicit channel indices to drop. When given,
            ``criteria`` is ignored and these indices are used directly.
        amplitude_kwargs (dict or None): Passed to
            :func:`amplitude_bad_channels`.
        muap_band_kwargs (dict or None): Passed to
            :func:`muap_band_bad_channels` or
            :func:`auto_muap_band_bad_channels`.

    Returns:
        tuple:
            - ndarray: Sorted indices of kept channels.
            - ndarray: Sorted indices of dropped channels.
            - dict: Per-rule detail. Keys: ``amplitude_dropped``,
              ``muap_band_dropped``, ``muap_band_threshold``,
              ``muap_band_bimodal``, ``muap_band_ratio``, ``criteria``.
    """
    sig = np.asarray(sig, float)
    n_ch = sig.shape[0]
    info = {"amplitude_dropped": np.array([], int), "muap_band_dropped": np.array([], int),
            "rms_high_dropped": np.array([], int), "rms_low_dropped": np.array([], int),
            "kurtosis_dropped": np.array([], int), "rms_ratio": None, "kurt": None,
            "muap_band_threshold": None, "muap_band_bimodal": None,
            "muap_band_ratio": None, "criteria": tuple(criteria),
            "n_channels": int(n_ch)}

    if drop is not None:
        dropped = np.unique(np.asarray(drop, int))
        info["criteria"] = ("explicit",)
        return np.setdiff1d(np.arange(n_ch), dropped), dropped, info

    dropped = np.array([], int)
    needs_bp = any(c in ("muap_band", "auto_muap_band") for c in criteria)
    sig_bp = bandpass(sig, fs, *criterion_bandpass) if (needs_bp and criterion_bandpass) else sig

    if "amplitude" in criteria:
        detail = amplitude_breakdown(sig, **(amplitude_kwargs or {}))
        bad = np.union1d(np.union1d(detail["rms_high"], detail["rms_low"]),
                         detail["kurtosis"]).astype(int)
        info["amplitude_dropped"] = bad
        info["rms_high_dropped"] = detail["rms_high"]
        info["rms_low_dropped"] = detail["rms_low"]
        info["kurtosis_dropped"] = detail["kurtosis"]
        info["rms_ratio"] = detail["rms_ratio"]
        info["kurt"] = detail["kurt"]
        dropped = np.union1d(dropped, bad)

    if needs_bp:
        mb_kwargs = dict(muap_band_kwargs or {})
        ratio_kwargs = {k: mb_kwargs[k] for k in ("mu_band", "emg_band", "nperseg") if k in mb_kwargs}
        info["muap_band_ratio"] = muap_band_ratio(sig_bp, fs, **ratio_kwargs)
        if "auto_muap_band" in criteria:
            bad, thr, bimodal = auto_muap_band_bad_channels(sig_bp, fs, **mb_kwargs)
            info["muap_band_threshold"], info["muap_band_bimodal"] = thr, bimodal
        else:
            bad = muap_band_bad_channels(sig_bp, fs, **mb_kwargs)
            info["muap_band_threshold"] = mb_kwargs.get("thr", 0.55)
            info["muap_band_bimodal"] = False
        info["muap_band_dropped"] = bad
        dropped = np.union1d(dropped, bad)

    dropped = np.asarray(dropped, int)
    return np.setdiff1d(np.arange(n_ch), dropped), dropped, info


def remove_common_mode(X, k=2):
    """Project out the leading spatial components shared across channels.

    Takes the SVD over channels and subtracts the projection onto the top ``k``
    left singular vectors. ``X`` is expected to be mean-free per channel.

    Args:
        X (ndarray): Signal, (n_channels, n_samples).
        k (int): Number of components to remove. 0 returns ``X`` unchanged.

    Returns:
        tuple:
            - ndarray: Signal with the components removed, same shape as ``X``.
            - ndarray: Removed components, (k, n_samples). Empty when k is 0.
    """
    X = np.asarray(X, float)
    if k <= 0:
        return X, np.zeros((0, X.shape[1]))
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    k = min(k, U.shape[1])
    proj = U[:, :k].T @ X
    return X - U[:, :k] @ proj, proj


def preprocess(sig, fs, criteria=("amplitude", "auto_muap_band"), common_mode_k=2,
               zscore=True, criterion_bandpass=(20.0, 500.0), drop=None,
               amplitude_kwargs=None, muap_band_kwargs=None):
    """Select channels, remove common mode, and normalise channel amplitude.

    Order of operations: channel selection, per-channel mean removal, common-mode
    removal, per-channel division by standard deviation.

    Args:
        sig (ndarray): Signal, (n_channels, n_samples).
        fs (float): Sampling frequency in Hz.
        criteria (tuple): See :func:`select_channels`.
        common_mode_k (int): Number of common-mode components to remove.
        zscore (bool): Divide each channel by its standard deviation.
        criterion_bandpass (tuple or None): See :func:`select_channels`.
        drop (sequence or None): Explicit channel indices to drop, bypassing
            ``criteria``.
        amplitude_kwargs (dict or None): See :func:`select_channels`.
        muap_band_kwargs (dict or None): See :func:`select_channels`.

    Returns:
        tuple:
            - ndarray: Preprocessed signal, (n_kept_channels, n_samples).
            - ndarray: Indices of kept channels, referring to ``sig`` rows.
            - ndarray: Indices of dropped channels.
            - dict: Detail from :func:`select_channels`, with
              ``common_mode_k`` and ``zscore`` added.
    """
    keep, dropped, info = select_channels(
        sig, fs, criteria=criteria, criterion_bandpass=criterion_bandpass, drop=drop,
        amplitude_kwargs=amplitude_kwargs, muap_band_kwargs=muap_band_kwargs)

    X = np.asarray(sig, float)[keep]
    X = X - X.mean(1, keepdims=True)
    X, _ = remove_common_mode(X, common_mode_k)
    if zscore:
        X = X / (X.std(1, keepdims=True) + 1e-9)

    info = dict(info, common_mode_k=int(common_mode_k), zscore=bool(zscore))
    return X, keep, dropped, info
