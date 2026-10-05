import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
from scipy.linalg import toeplitz
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from scipy.signal import butter, filtfilt

from ..evaluation.evaluate import *


def bandpass_signals(emg_data, fsamp, high_pass=20, low_pass=500, order=2):
    """
    Bandpass filter emg data using a butterworth filter

    Args:
        emg_data (ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        low_pass (float): Cut-off frequency for the low-pass filter
        high_pass (float): Cut-off frequency for the high-pass filter
        order (int): Order of the filter

    Returns:
        ndarray : filtered emg data (n_channels x n_samples)
    """

    b, a = butter(order, [high_pass, low_pass], fs=fsamp, btype="band")
    emg_data = filtfilt(b, a, emg_data, axis=1)

    return emg_data


def notch_signals(emg_data, fsamp, nfreq=50, dfreq=1, order=2, n_harmonics=3):
    """
    Notch filter emg data using a butterworth filter

    Args:
        emg_data (ndarray): emg data (n_channels x n_samples)
        fsamp (float): Sampling frequency
        nfreq (float): frequency to be filtered
        dfreq (float): width of the notch filter (plus/minus dfreq)
        order (int): Order of the filter
        n_harmonics: Number of harmonics to be filtered

    Returns:
        ndarray : filtered emg data (n_channels x n_samples)
    """

    harmonics = nfreq * np.arange(1, n_harmonics + 1)

    for i in np.arange(n_harmonics):
        b, a = butter(
            order,
            [harmonics[i] - dfreq, harmonics[i] + dfreq],
            fs=fsamp,
            btype="bandstop",
        )
        emg_data = filtfilt(b, a, emg_data, axis=1)

    return emg_data


def extension(Y, R):
    """
    Extend a multi-channel signal Y by an extension factor R
    using Toeplitz matrices.

    Parameters:
        Y (ndarray): Original signal (n_channels x n_samples)
        R (int): Extension factor (number of lags)

    Returns:
        eY (ndarray): Extended signal (n_channels * R x n_samples)
    """
    n_channels, n_samples = Y.shape
    eY = np.zeros((n_channels * R, n_samples))

    for i in range(n_channels):
        col = np.concatenate(([Y[i, 0]], np.zeros(R - 1)))
        row = Y[i, :]
        T = toeplitz(col, row)
        eY[i * R : (i + 1) * R, :] = T

    return eY


def whitening(Y, method="ZCA", backend="ed", regularization="auto", eps=1e-10,
              rank_trunc=None):
    """
    Adaptive whitening function using ZCA, PCA, or Cholesky.

    Parameters:
        Y (ndarray): Input signal (n_channels x n_samples)
        method (str): Whitening method: 'ZCA', 'PCA', 'Cholesky'
        backend (str): 'ed', or 'svd'
        regularization (str or float): 'auto', float value, or None
        eps (float): Small epsilon for numerical stability
        rank_trunc (float or None): Rank-truncated whitening. If a float in (0,1),
            eigen-components with eigenvalue <= rank_trunc * max_eigenvalue are
            DROPPED (their inverse-sqrt set to 0) instead of amplified. This is the
            fix for a near-singular extended-EMG covariance (see mu_xclean/diagnosis):
            the default 'auto' ridge is ~1e-8 and whitens the noise subspace to unit
            variance, drowning the motor units; a truncation of ~1e-3 keeps only the
            components carrying real variance. None (default) preserves the original
            behavior. Applies to both backends and to ZCA/PCA.

    Returns:
        wY (ndarray): Whitened signal
        Z (ndarray): Whitening matrix
    """
    n_channels, n_samples = Y.shape
    use_svd = backend == "svd"

    if method == "Cholesky":
        covariance = Y @ Y.T / (n_samples - 1)
        R = np.linalg.cholesky(covariance)
        Z = np.linalg.inv(R.T)
        wY = Z @ Y
        return wY, Z

    # Use SVD
    if use_svd:
        covariance = Y @ Y.T / (n_samples - 1)
        # covariance = np.cov(Y)
        # Decompose in float64 for stability; covariance is small (n_ch x n_ch)
        U, S, _ = np.linalg.svd(covariance.astype(np.float64, copy=False), full_matrices=False)
        if regularization == "auto":
            reg = np.mean(S[len(S) // 2 :] ** 2)
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1.0 / np.sqrt(S + reg + eps)
        if rank_trunc is not None:                       # drop near-null noise subspace
            S_inv = np.where(S > rank_trunc * S.max(), S_inv, 0.0)

        if method == "ZCA":
            Z = U @ np.diag(S_inv) @ U.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ U.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    # Use EIG
    else:
        covariance = Y @ Y.T / (n_samples - 1)
        S, V = np.linalg.eigh(covariance)

        if regularization == "auto":
            reg = np.mean(S[: len(S) // 2])
        elif isinstance(regularization, float):
            reg = regularization
        else:
            reg = 0
        S_inv = 1.0 / np.sqrt(S + reg + eps)
        if rank_trunc is not None:                       # drop near-null noise subspace
            S_inv = np.where(S > rank_trunc * S.max(), S_inv, 0.0)

        if method == "ZCA":
            Z = V @ np.diag(S_inv) @ V.T
        elif method == "PCA":
            Z = np.diag(S_inv) @ V.T
        else:
            raise ValueError("Unknown method.")
        wY = Z @ Y

    return wY, Z


def est_spike_times(sig, fsamp, cluster="kmeans", a=2, min_delay=0.01, centroids=None, outlier_percentile=99.9):
    """
    Estimate spike indices given a motor unit source signal and compute
    a silhouette-like metric for source quality quantification

    Args:
        sig (np.ndarray): Input signal (motor unit source)
        fsamp (float): Sampling rate in Hz
        cluster (string): Clustering method used to identify the spike indices.
            - 'kmeans': Use K-means clustering (default, for offline/initial decomposition)
            - 'kmedoids': Use K-medoids clustering (more robust to outliers)
            - 'centroid': Use pre-computed centroids (for real-time, requires centroids param)
        a (float): Exponent of assymetric power law
        min_delay (float): Minimum delay between peaks in seconds (default: 0.01)
        centroids (np.ndarray): Pre-computed centroids [noise_centroid, spike_centroid]
            Required when cluster='centroid'. The spike centroid should be the larger value.
        outlier_percentile (float): Percentile threshold for outlier removal (default: 99.9).
            Peaks above this percentile will be excluded from clustering.

    Returns:
        est_spikes (np.ndarray): Estimated spike indices
        sil (float): Silhouette-like score (0 = poor, 1 = strong separation)
        centroids (np.ndarray or None): Cluster centroids (shape: [2,]) or None if no peaks found
    """
    sig = np.asarray(sig)

    # Assymetric power law that can be useful for contrast enhancement
    sig = np.sign(sig) * np.abs(sig)**a

    # Detect peaks with minimum distance
    min_peak_dist = int(round(fsamp * min_delay))
    peaks, _ = find_peaks(sig, distance=min_peak_dist)

    if len(peaks) == 0:
        return np.array([]), 0.0, centroids

    # Get peak values
    peak_vals = sig[peaks].reshape(-1, 1)

    # Calculate outlier threshold before filtering
    threshold = None
    outlier_mask = None
    if outlier_percentile is not None and outlier_percentile < 100:
        threshold = np.percentile(peak_vals, outlier_percentile)
        outlier_mask = peak_vals.flatten() <= threshold

    # Remove outliers based on percentile threshold
    if outlier_mask is not None:
        # n_removed = np.sum(~outlier_mask)
        # if n_removed > 0:
            # print(f"[INFO] Removed {n_removed} outlier peaks above {outlier_percentile}th percentile (threshold: {threshold:.2f})")
        peaks_clean = peaks[outlier_mask]
        peak_vals_clean = peak_vals[outlier_mask]
    else:
        peaks_clean = peaks
        peak_vals_clean = peak_vals

    if len(peaks_clean) == 0:
        return np.array([]), 0.0, centroids
    
    """"
    # Debug plot of the source signal and detected peaks
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # Plot 1: Source signal with detected peaks (showing outliers)
    axes[0].plot(sig, 'b-', alpha=0.7, label='Source signal')
    if threshold is not None:
        # Plot kept peaks in green, outliers in red
        axes[0].scatter(peaks[outlier_mask], sig[peaks[outlier_mask]], c='g', s=20, zorder=5, label=f'Kept peaks (n={np.sum(outlier_mask)})')
        axes[0].scatter(peaks[~outlier_mask], sig[peaks[~outlier_mask]], c='r', s=40, marker='x', zorder=6, label=f'Outliers (n={np.sum(~outlier_mask)})')
        axes[0].axhline(threshold, color='orange', linestyle='--', linewidth=2, label=f'Threshold ({outlier_percentile}%ile): {threshold:.2f}')
    else:
        axes[0].scatter(peaks, sig[peaks], c='r', s=20, zorder=5, label=f'Peaks (n={len(peaks)})')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Source Signal with Detected Peaks')
    axes[0].legend()

    # Plot 2: Histogram of peak values (after outlier removal)
    axes[1].hist(peak_vals_clean, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(peak_vals_clean.mean(), color='r', linestyle='--', label=f'Mean: {peak_vals_clean.mean():.2f}')
    axes[1].set_xlabel('Peak Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Peak Values (after outlier removal)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    """

    # Update peaks and peak_vals to filtered versions
    peaks = peaks_clean
    peak_vals = peak_vals_clean



    # print(f"Peak values range: {peak_vals.min():.2f} to {peak_vals.max():.2f}")
    # if centroids is not None:
        # print(f"Centroids: {centroids}")

    if cluster == "kmeans" or cluster == "kmedoids":
        # K-means/K-medoids clustering to separate signal vs. noise
        if cluster == "kmeans":
            # print("[INFO] Using K-means clustering for spike detection.")
            model = KMeans(n_clusters=2, n_init=10, random_state=42)
        else:
            # print("[INFO] Using K-medoids clustering for spike detection.")
            model = KMedoids(n_clusters=2, random_state=42)
        labels = model.fit_predict(peak_vals)
        centroids = model.cluster_centers_.flatten()

        # Spikes are those in the cluster with the higher mean
        spike_cluster = np.argmax(centroids)
        est_spikes = peaks[labels == spike_cluster]

        # Compute within- and between-cluster distances
        D = model.transform(peak_vals)  # Distances to both centroids
        sumd = np.sum(
            D[labels == spike_cluster, spike_cluster] ** 2
        )  # Exponent 2 for obtaining the squared Euclidian distance
        between = np.sum(
            D[labels == spike_cluster, 1 - spike_cluster] ** 2
        )  # Exponent 2 for obtaining the squared Euclidian distance

        # Silhouette-inspired score
        denom = max(sumd, between)
        sil = (between - sumd) / denom if denom > 0 else 0.0

    elif cluster == "centroid":
        # Use pre-computed centroids for real-time spike detection
        # This avoids K-means overhead and provides consistent classification
        if centroids is None:
            raise ValueError("centroids must be provided when cluster='centroid'")

        centroids = np.asarray(centroids).flatten()
        if len(centroids) != 2:
            raise ValueError("centroids must have exactly 2 values: [noise_centroid, spike_centroid]")

        # Identify which centroid is the spike centroid (higher value)
        spike_cluster = np.argmax(centroids)
        noise_cluster = np.argmin(centroids)

        # print(f"[DEBUG] Using centroids: Noise = {centroids[noise_cluster]:.2f}, Spike = {centroids[spike_cluster]:.2f}")

        # Compute Euclidean distance from each peak to both centroids
        dist_to_spike = np.abs(peak_vals.flatten() - centroids[spike_cluster])
        dist_to_noise = np.abs(peak_vals.flatten() - centroids[noise_cluster])

        # Classify: spike if closer to spike centroid
        labels = (dist_to_spike < dist_to_noise).astype(int)
        # labels=1 means spike, labels=0 means noise
        est_spikes = peaks[labels == 1]

        # Compute silhouette-like score using the same formula as K-means
        if np.sum(labels == 1) > 0:
            sumd = np.sum(dist_to_spike[labels == 1] ** 2)
            between = np.sum(dist_to_noise[labels == 1] ** 2)
            denom = max(sumd, between)
            sil = (between - sumd) / denom if denom > 0 else 0.0
        else:
            sil = 0.0
    else:
        raise ValueError(f"Unknown cluster method: {cluster}. Use 'kmeans', 'kmedoids', or 'centroid'.")

    # print(f"Detected {len(est_spikes)} spikes.\n"
    #       f"Silhouette-like score: {sil:.4f} \n"
    #       f"Centroids: {centroids[0]:.2f}, {centroids[1]:.2f}")
    return est_spikes, sil, centroids


def gram_schmidt(w, B):
    """
    Stabilized Gram-Schmidt orthogonalization.

    Args:
        w (np.ndarray): Input vector to be orthogonalized (shape: [n,])
        B (np.ndarray): Matrix of orthogonal basis vectors in columns (shape: [n, k])

    Returns:
        u (np.ndarray): Orthogonalized vector
    """
    w = np.asarray(w, dtype=float)
    B = np.asarray(B, dtype=float)

    # Remove zero columns from B
    non_zero_cols = ~np.all(B == 0, axis=0)
    B = B[:, non_zero_cols]

    u = w.copy()
    for i in range(B.shape[1]):
        a = B[:, i]
        projection = (np.dot(u, a) / np.dot(a, a)) * a
        u = u - projection

    return u


def cov_isi(spike_indices, fsamp):
    """
    Coefficient of variation of the inter-spike intervals of one spike train.

    Args:
        - spike_indices (np.ndarray): Spike indices of one source
        - fsamp (float): Sampling rate in Hz

    Returns:
        - cov (float): Std over mean of the ISIs, np.inf for trains with fewer
          than three spikes or a non-positive mean ISI
    """
    spike_indices = np.asarray(spike_indices, dtype=int)
    if spike_indices.size <= 2:
        return np.inf
    isi = np.diff(spike_indices / fsamp)
    return np.std(isi) / np.mean(isi) if np.mean(isi) > 0 else np.inf


def duplicate_labels(spikes, n_samples, fsamp, max_shift=0.1, tol=0.001, threshold=0.3):
    """
    Group sources whose spike trains agree into duplicate clusters.

    Every pair is compared at the delay maximizing the cross-correlation of the
    binary trains within max_shift; the pair is a duplicate when the fraction of
    spikes matching within tol reaches the threshold. Grouping is transitive
    (single linkage): if source A matches B and B matches C, all three end up in
    one cluster even when A and C do not match each other directly.

    Args:
        - spikes (dict): Spiking instances of the motor neurons
        - n_samples (int): Number of time samples of the sources
        - fsamp (float): Sampling rate in Hz
        - max_shift (float): Maximal delay between two sources in seconds
        - tol (float): All spikes with a delay lower than tolerance (in seconds) are classified identical
        - threshold (float): Minimum fraction of common spikes to classify two sources as identical

    Returns:
        - labels (np.ndarray): Cluster label of each source (n_mu,). The label of
          a cluster is the smallest source index it contains.
    """
    n_source = len(spikes)
    parent = np.arange(n_source)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    trains = [get_bin_spikes(np.asarray(spikes[i], dtype=int), n_samples)
              for i in range(n_source)]

    for i in range(n_source):
        # Sources without spikes never match anything
        if len(spikes[i]) == 0:
            continue

        for j in range(i + 1, n_source):
            # Already in the same cluster, the comparison cannot change it
            if len(spikes[j]) == 0 or find(i) == find(j):
                continue
            # Compute the delay between source i and j
            _, shift = max_xcorr(trains[i], trains[j], max_shift=int(max_shift * fsamp))
            # Compute the number of common spikes
            tp, _, _ = match_spike_trains(trains[i], trains[j], shift=shift,
                                          tol=tol, fsamp=fsamp)
            # Calculate the matching rate and compare with threshold
            denom = max(len(spikes[i]), len(spikes[j]))
            match_score = tp / denom if denom > 0 else 0
            if match_score >= threshold:
                union(i, j)

    return np.array([find(i) for i in range(n_source)], dtype=int)


def duplicate_representatives(
    spikes, sil, n_samples, fsamp, max_shift=0.1, tol=0.001, threshold=0.3,
    covs=None, sil_th=None, min_firing_rate=None
):
    """
    Pick the source to keep for each duplicate cluster.

    Members of a cluster are ranked by, in order:
        1. clearing the quality gate, that is a silhouette of at least sil_th and
           a firing rate of at least min_firing_rate. A threshold left at None is
           not applied, so with neither one every member clears the gate,
        2. the lowest CoV-ISI, when covs is given,
        3. the highest silhouette, which is the only criterion when covs is None.
    Remaining ties go to the source found first.

    Args:
        - spikes (dict): Spiking instances of the motor neurons
        - sil (np.ndarray): Source quality metric
        - n_samples (int): Number of time samples of the sources
        - fsamp (float): Sampling rate in Hz
        - max_shift (float), tol (float), threshold (float): Passed to
          ``duplicate_labels``
        - covs (np.ndarray or None): CoV-ISI of each source
        - sil_th (float or None), min_firing_rate (float or None): Quality gate

    Returns:
        - keep (np.ndarray): Indices of the sources to keep, ascending, one per
          duplicate cluster
    """
    n_source = len(spikes)
    sil = np.asarray(sil, dtype=float)
    labels = duplicate_labels(spikes, n_samples, fsamp, max_shift=max_shift,
                              tol=tol, threshold=threshold)

    # Sources with an undefined CoV-ISI (too few spikes) rank last on that key
    if covs is None:
        rank_cov = np.zeros(n_source)
    else:
        covs = np.asarray(covs, dtype=float)
        rank_cov = np.where(np.isfinite(covs), -covs, -np.inf)

    gate = np.ones(n_source, dtype=bool)
    if sil_th is not None:
        gate &= sil >= sil_th
    if min_firing_rate is not None:
        duration = n_samples / fsamp
        fr = np.array([len(spikes[i]) / duration for i in range(n_source)])
        gate &= fr >= min_firing_rate

    keep = [max(np.flatnonzero(labels == label),
                key=lambda i: (gate[i], rank_cov[i], sil[i]))
            for label in np.unique(labels)]

    return np.array(sorted(keep), dtype=int)


def remove_duplicates(
    sources, spikes, sil, mu_filters, fsamp, max_shift=0.1, tol=0.001, threshold=0.3,
    covs=None, sil_th=None, min_firing_rate=None
):
    """
    Sort out source duplicates from a decomposition by clustering spike trains and
    keeping one representative source per cluster.

    Clustering is done by ``duplicate_labels`` and the representative is chosen
    by ``duplicate_representatives``: the lowest CoV-ISI among the members that
    clear the quality gate when covs is given, the highest silhouette otherwise.

    Args:
        - sources (np.ndarray): Original sources (n_mu x n_samples)
        - spikes (dict): Original ppiking instances of the motor neurons
        - sil (np.ndarray): Original source quality metric
        - mu_filters (np.ndarray): Original motor unit filters
        - fsamp (float): Sampling rate in Hz
        - max_shift (float): Maximal delay between two sources in seconds
        - tol (float): All spikes with a delay lower than tolerance (in seconds) are classified identical
        - theshold (float): Minimum fraction of common spikes to classify two sources as identical
        - covs (np.ndarray or None): CoV-ISI of each source, enables the CoV rule
        - sil_th (float or None), min_firing_rate (float or None): Quality gate
          used to rank cluster members

    Returns:
        - new_sources (np.ndarray): Updated sources (n_mu x n_samples)
        - new_spikes (dict): Updated spiking instances of the motor neurons
        - new_sil (np.ndarray): Updated source quality metric
        - new_filters (np.ndarray): Updated motor unit filters


    """
    keep = duplicate_representatives(
        spikes, sil, sources.shape[1], fsamp, max_shift=max_shift, tol=tol,
        threshold=threshold, covs=covs, sil_th=sil_th, min_firing_rate=min_firing_rate,
    )

    new_sources = sources[keep, :]
    new_spikes = {new: spikes[old] for new, old in enumerate(keep)}
    new_sil = np.asarray(sil, dtype=float)[keep]
    new_filters = mu_filters[:, keep]

    return new_sources, new_spikes, new_sil, new_filters


def good_source_mask(
    spikes, sil, n_samples, fsamp=None, threshold=0.9, min_firing_rate=1.0
):
    """
    Mask of the sources passing the silhouette and firing rate cuts.

    Args:
        - spikes (dict): Spiking instances of the motor neurons
        - sil (np.ndarray): Source quality metric
        - n_samples (int): Number of time samples of the sources
        - fsamp (float or None): Sampling frequency in Hz, required for the
          firing rate. Without it the firing rate is taken as 0.0
        - threshold (float): Sources with a SIL score below this will be rejected
        - min_firing_rate (float): Sources with a firing rate below this will be
          rejected (in Hz)

    Returns:
        - keep (np.ndarray): Boolean mask of the sources to keep (n_mu,)
    """
    sil = np.asarray(sil, dtype=float)
    duration = n_samples / fsamp if fsamp is not None else None

    keep = np.zeros(len(sil), dtype=bool)
    for i in range(len(sil)):
        firing_rate = len(spikes[i]) / duration if duration is not None else 0.0
        keep[i] = sil[i] >= threshold and firing_rate >= min_firing_rate

    return keep


def remove_bad_sources(
    sources, spikes, sil, mu_filters, centroids, covs, threshold=0.9, min_firing_rate=1.0, fsamp=None, max_cov=None
):
    """
    Reject sources with a silhoeutte score below a given threshold and that do not
    have a minimum firing rate.

    Args:
        - sources (np.ndarray): Original sources (n_mu x n_samples)
        - spikes (dict): Original ppiking instances of the motor neurons
        - sil (np.ndarray): Original source quality metric
        - mu_filters (np.ndarray): Original motor unit filters
        - centroids (np.ndarray): K-means centroids of each source
        - theshold (float): Sources with a SIL score below this theshold will be rejected
        - min_firing_rate (float): Sources with firing rate below this will be rejected (in Hz)
        - fsamp (float): Sampling frequency in Hz (required for firing rate calculation)

    Returns:
        - new_sources (np.ndarray): Updated sources (n_mu x n_samples)
        - new_spikes (dict): Updated spiking instances of the motor neurons
        - new_sil (np.ndarray): Updated source quality metric
        - new_filters (np.ndarray): Updated motor unit filters
        - new_centroids (np.ndarray): Updated K-means centroids of each source

    """

    keep = np.flatnonzero(good_source_mask(
        spikes, sil, sources.shape[1], fsamp=fsamp,
        threshold=threshold, min_firing_rate=min_firing_rate,
    ))

    new_sources = sources[keep, :]
    new_spikes = {new: spikes[old] for new, old in enumerate(keep)}
    new_sil = np.asarray(sil, dtype=float)[keep]
    new_filters = mu_filters[:, keep]
    new_centroids = {new: centroids[old] for new, old in enumerate(keep)}

    return new_sources, new_spikes, new_sil, new_filters, new_centroids


def map_source_from_window_to_global_time_idx(sources, spikes, win, n_time_samples):
    """
    TODO some description

    Args:
        sources (np.ndarray): Original sources
        spikes (dict): Original spikes
        win (tuple): Time indices of the window (start, end)
        n_time_samples (int): Number of time samples of the original recording

    Returns:
        new_sources (np.ndarray): Mapped sources
        spikes (float): (dict): Mapped spikes

    """

    # Initalize variables
    new_sources = np.zeros((sources.shape[0], n_time_samples))
    new_spikes = {i: [] for i in range(sources.shape[0])}

    for i in range(new_sources.shape[0]):
        new_sources[i, win[0] : win[1]] = sources[i, :]
        new_spikes[i] = spikes[i] + win[0]

    return new_sources, new_spikes


def spike_triggered_average(sig, spikes, win=0.02, fsamp=2048):
    """
    Calculate the spike triggered average given the spike times of a source

    Parameters:
        sig (2D np.array): signal [channels x time]
        spikes (1D array): Spike indices
        fsamp (float): Sampling frequency in Hz
        win (float): Window size in seconds for MUAP template (in seconds)

    Returns:
        waveform (2D np.array): Estimated impulse response of a given source

    """

    width = int(win * fsamp)
    waveform = np.zeros((sig.shape[0], 2 * width + 1))

    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]

    for i in np.arange(len(spikes)):
        waveform = waveform + sig[:, (spikes[i] - width) : (spikes[i] + width + 1)]

    waveform = waveform / len(spikes)

    return waveform


def peel_off(sig, spikes, win=0.02, fsamp=2048):
    """
    Peel off signal component based on spike triggered average.

    Parameters:
        sig (2D np.array): signal [channels x time]
        spikes (1D array): Spike indices
        fsamp (float): Sampling frequency in Hz
        win (float): Window size in seconds for MUAP template (in seconds)

    Returns:
        residual_sig (2D np.array): Residual signal after removing component
        comp_sig (2D np.array): Estimated contribution of the given source
    """

    waveform = spike_triggered_average(sig, spikes, win, fsamp)

    width = int(win * fsamp)
    spikes = spikes[(spikes >= width + 1) & (spikes < sig.shape[1] - width - 1)]
    firings = np.zeros(sig.shape[1])
    firings[spikes] = 1

    # Zero-pad waveform to match signal shape
    L = sig.shape[1]
    pad_len = L - waveform.shape[1]
    waveform_padded = np.pad(waveform, ((0, 0), (0, pad_len)), mode="constant")

    # FFT of firings (same for all channels)
    fft_firings = fft(firings)

    # FFT of waveform for each channel
    fft_waveform = fft(waveform_padded, axis=1)

    # Multiply in frequency domain (broadcasting firings to each channel)
    fft_product = fft_waveform * fft_firings

    # IFFT to get time domain component signal
    comp_sig = np.real(ifft(fft_product, axis=1))

    # Correct time shift due to FFT convolution (center of kernel)
    shift = (waveform.shape[1] - 1) // 2
    comp_sig = np.roll(comp_sig, -shift, axis=1)

    residual_sig = sig - comp_sig

    return residual_sig, comp_sig, waveform


def spike_dict_to_long_df(spike_dict, sort=True, fsamp=2048):
    """
    Convert a dictionary of spike instances into a long-formatted DataFrame.

    Parameters:
        spike_dict (dict): Keys are unit IDs, values are lists or arrays of spike times.
        sort (bool): Whether to sort the result by unit and spike time.
        fsamp (float): Sampling frequency to convert sample indices to time.

    Returns:
        pd.DataFrame: Long-formatted DataFrame with columns ['source_id', 'spike_time']
    """
    import pandas as pd

    rows = []
    for unit_id, spikes in spike_dict.items():
        for t in spikes:
            rows.append({"source_id": unit_id, "spike_time": t / fsamp})

    # If no spikes were found, create an empty DataFrame with the correct columns
    if not rows:
        return pd.DataFrame(columns=["source_id", "spike_time"])

    df = pd.DataFrame(rows)
    if sort and not df.empty:
        df = df.sort_values(by=["source_id", "spike_time"]).reset_index(drop=True)
    return df
