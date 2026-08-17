import numpy as np
import pickle as pkl
import os

from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from muniverse.algorithms.core import est_spike_times, extension



def design_filters(srate, lowcut=20.0, highcut=500.0, notch_freq=50.0):
    """Design bandpass and notch filters for EMG preprocessing.

    Returns SOS format filters and their initial conditions for stateful filtering.
    """
    nyq = srate / 2.0

    # Bandpass filter (4th order Butterworth)
    bp_sos = butter(4, [lowcut / nyq, highcut / nyq], btype='band', output='sos')

    # Notch filters for 50 Hz harmonics (up to Nyquist)
    notch_sos_list = []
    harmonic = notch_freq
    while harmonic < nyq:
        # Design notch filter (Q factor of 30 gives narrow notch)
        b, a = iirnotch(harmonic, Q=30, fs=srate)
        sos = tf2sos(b, a)
        notch_sos_list.append(sos)
        harmonic += notch_freq

    # Stack all notch filters into one SOS array
    if notch_sos_list:
        notch_sos = np.vstack(notch_sos_list)
    else:
        notch_sos = None

    return bp_sos, notch_sos


def init_filter_states(bp_sos, notch_sos, n_channels):
    """Initialize filter states for each channel (for stateful filtering)."""
    # Bandpass filter states: shape (n_sections, 2) per channel
    bp_zi_unit = sosfilt_zi(bp_sos)  # shape: (n_sections, 2)
    bp_zi = np.zeros((n_channels, bp_zi_unit.shape[0], 2))

    # Notch filter states
    if notch_sos is not None:
        notch_zi_unit = sosfilt_zi(notch_sos)
        notch_zi = np.zeros((n_channels, notch_zi_unit.shape[0], 2))
    else:
        notch_zi = None

    return bp_zi, notch_zi


def apply_filters(data, bp_sos, notch_sos, bp_zi, notch_zi):
    """Apply bandpass and notch filters to multi-channel data, updating filter states.

    Args:
        data: shape (n_channels, n_samples)
        bp_sos, notch_sos: filter coefficients in SOS format
        bp_zi, notch_zi: filter states per channel

    Returns:
        filtered_data, updated bp_zi, updated notch_zi
    """
    n_channels = data.shape[0]
    filtered = np.zeros_like(data)

    for ch in range(n_channels):
        # Apply bandpass filter
        y, bp_zi[ch] = sosfilt(bp_sos, data[ch], zi=bp_zi[ch])

        # Apply notch filters
        if notch_sos is not None:
            y, notch_zi[ch] = sosfilt(notch_sos, y, zi=notch_zi[ch])

        filtered[ch] = y

    return filtered, bp_zi, notch_zi




def calculate_firing_rates(spikes, n_mus, chunk_duration, spike_history):
    """Calculate firing rates for each motor unit over a rolling window of 5 chunks.

    Args:
        spikes: dict mapping MU index to array of spike indices for current chunk
        n_mus: number of motor units
        chunk_duration: duration of one chunk in seconds
        spike_history: list of lists, where spike_history[mu_idx] contains
                       spike counts from previous chunks (maintained externally)

    Returns:
        firing_rates: array of shape (n_mus,) with firing rates in Hz
        spike_history: updated history with current chunk's spike counts appended
    """
    window_size = 5

    for mu_idx in range(n_mus):
        # Get spike count for current chunk
        current_count = len(spikes[mu_idx]) if mu_idx in spikes else 0
        spike_history[mu_idx].append(current_count)

        # Keep only the last 5 chunks
        if len(spike_history[mu_idx]) > window_size:
            spike_history[mu_idx] = spike_history[mu_idx][-window_size:]

    # Calculate firing rates (spikes per second)
    firing_rates = np.zeros(n_mus)
    for mu_idx in range(n_mus):
        total_spikes = sum(spike_history[mu_idx])
        total_duration = len(spike_history[mu_idx]) * chunk_duration
        firing_rates[mu_idx] = total_spikes / total_duration if total_duration > 0 else 0.0

    return firing_rates, spike_history


def firing_rate_sliding_window(spike_times_samples, n_total_samples, fsamp, window_sec):
    """Compute instantaneous firing rate at each sample using a sliding window.

    Unlike ``calculate_firing_rates``, which works chunkwise over a rolling history
    for the online path, this operates on a whole recording at once.

    Returns an array of shape (n_total_samples,) with firing rate in Hz.
    """
    window_samples = int(window_sec * fsamp)
    half_win = window_samples // 2

    # Build a binary spike train
    spike_train = np.zeros(n_total_samples)
    valid = spike_times_samples[(spike_times_samples >= 0) & (spike_times_samples < n_total_samples)]
    spike_train[valid.astype(int)] = 1.0

    # Cumulative sum for fast window counting
    cumsum = np.cumsum(spike_train)
    cumsum = np.insert(cumsum, 0, 0)  # prepend 0 for indexing

    fr = np.zeros(n_total_samples)
    for t in range(n_total_samples):
        t_start = max(0, t - half_win)
        t_end = min(n_total_samples, t + half_win)
        actual_window = (t_end - t_start) / fsamp
        n_spikes_in_win = cumsum[t_end] - cumsum[t_start]
        fr[t] = n_spikes_in_win / actual_window if actual_window > 0 else 0.0

    return fr


def process_chunk(samples, mu_filters, Z, n_mus, fs, centroids, prev_tail=None, norm_factors=None, verbose=True):
    """Process a chunk of EMG data for online decomposition.

    Args:
        samples: EMG data (n_channels x n_samples)
        mu_filters: MU separation filters (n_ext_channels x n_mus)
        Z: Whitening matrix (n_ext_channels x n_ext_channels)
        n_mus: Number of motor units
        fs: Sampling frequency
        centroids: Dict mapping MU index to [noise_centroid, spike_centroid]
        prev_tail: Previous chunk's tail for continuity
        norm_factors: Dict mapping MU index to (offset, scale) for [0,1] normalization (Farina 2025)
                     If provided, sources are normalized before spike detection.
        verbose: Whether to print debug info (default True)

    Returns:
        sources_full: Source signals (n_mus x n_samples)
        spikes_full: Dict mapping MU index to spike times
        sil_full: Silhouette scores for each MU
        new_tail: Tail samples for next chunk
    """
    R = 16  # Extension factor

    if verbose:
        print(f"[INFO] Received chunk shape: {samples.shape} of seconds: {samples.shape[1]/fs}")

    # Prepend previous tail to avoid edge effects at chunk boundaries
    if prev_tail is not None:
        samples = np.hstack([prev_tail, samples])
        if verbose:
            print(f"[INFO] Prepended {prev_tail.shape[1]} samples from previous chunk")

    ext_sig = extension(samples, R)
    ext_sig -= np.mean(ext_sig, axis=1, keepdims=True)

    if verbose:
        print(f"[INFO] Extended signal shape: {ext_sig.shape}")

    white_sig_full = Z @ ext_sig

    if verbose:
        print(f"[INFO] Whitened signal shape: {white_sig_full.shape}")
        print(f"[INFO] Separation matrix (mu_filters) shape: {mu_filters.shape}")

    # Apply separation matrix: S = B^T @ Z (where B = mu_filters)
    # mu_filters is (n_channels x n_mus), so we need to transpose it for matrix multiplication
    sources_full = mu_filters.T @ white_sig_full

    if verbose:
        print(f"[INFO] Full sources shape: {sources_full.shape}")

    # Apply normalization before spike detection (Farina 2025)
    # This constrains pulse train amplitudes to [0, 1] range
    if norm_factors is not None:
        if verbose:
            print("[INFO] Applying normalization factors (Farina 2025)")
        for i in range(sources_full.shape[0]):
            if i in norm_factors:
                offset, scale = norm_factors[i]
                sources_full[i, :] = (sources_full[i, :] - offset) / scale
                sources_full[i, :] = np.clip(sources_full[i, :], 0, 1)

    # Step 3: Extract spike times from (normalized) sources
    spikes_full = {}
    sil_full = np.zeros(mu_filters.shape[1])
    if centroids is None:
        clusters = "kmeans"
        if verbose:
            print("[INFO] No centroids provided, using k-means clustering for spike detection.")
    else:
        clusters = "centroid"
        if verbose:
            print("[INFO] Using provided centroids for spike detection.")

    for i in range(mu_filters.shape[1]):
        spikes_full[i], sil_full[i], centroids[i] = est_spike_times(
            sources_full[i, :],
            fs,
            cluster=clusters,
            centroids=centroids[i],
        )

    if verbose:
        print(f"[INFO] Spike extraction complete, detected spikes for {len(spikes_full)} MUs.")
        print(f"[INFO] Detected spikes per MU: " + ", ".join([f"MU{mu}: {len(spikes_full[mu])} \n " for mu in spikes_full]))
        print(f"[INFO] SIL scores for MUs: {sil_full}")

    # Save the tail for next chunk (last R-1 samples of the original signal)
    new_tail = samples[:, -(R-1):]

    return sources_full, spikes_full, sil_full, new_tail