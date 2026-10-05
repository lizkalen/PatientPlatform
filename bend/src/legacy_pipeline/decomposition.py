import pickle as pkl
import numpy as np
import os 
import sys
import json
import numpy as np
from scipy.interpolate import interp1d

from muniverse.algorithms.decomposition import decompose_cbss
from muniverse.algorithms.cbss import CBSS
from muniverse.algorithms.core import est_spike_times


def recalibrate_filters_sta(white_sig, original_filters, fsamp, spike_window_samples=None):
    """
    Recalibrate MU filters using spike-triggered averaging on whitened raw EMG.

    This reintroduces temporal overlaps removed during offline peel-off,
    making filters suitable for online (non-peel-off) spike detection.

    As per Farina 2025: Original filters are applied to raw EMG to estimate
    discharge times, then spike-triggered averaging is performed on the raw
    EMG to recalculate filters capturing full temporal overlap structure.

    Args:
        white_sig: Whitened extended EMG signal (n_ext_channels x n_samples)
        original_filters: Original MU filters from offline decomposition (n_ext_channels x n_mus)
        fsamp: Sampling frequency
        spike_window_samples: Half-window size for STA in samples.
                             If None, defaults to ~10ms (fsamp // 100)

    Returns:
        recalibrated_filters: New MU filters (n_ext_channels x n_mus)
    """
    n_ext_channels, n_samples = white_sig.shape
    n_mus = original_filters.shape[1]

    if spike_window_samples is None:
        spike_window_samples = int(fsamp // 100)  # ~10ms half-window

    # Step 1: Apply original filters to get initial spike estimates
    initial_sources = original_filters.T @ white_sig  # (n_mus, n_samples)

    recalibrated_filters = np.zeros_like(original_filters)

    for mu_idx in range(n_mus):
        # Step 2: Detect spikes from initial source
        spikes_i, sil_i, _ = est_spike_times(
            initial_sources[mu_idx, :],
            fsamp,
            cluster="kmeans"
        )

        if len(spikes_i) < 5:
            # Not enough spikes for reliable STA, keep original filter
            recalibrated_filters[:, mu_idx] = original_filters[:, mu_idx]
            print(f"  MU {mu_idx}: Only {len(spikes_i)} spikes, keeping original filter")
            continue

        # Step 3: Spike-triggered averaging on whitened signal
        # Collect segments around each spike
        sta_sum = np.zeros(n_ext_channels)
        valid_spikes = 0

        for spike_sample in spikes_i:
            spike_sample = int(spike_sample)
            # Skip spikes too close to edges
            if spike_sample < spike_window_samples or spike_sample >= n_samples - spike_window_samples:
                continue

            # Extract segment centered on spike (using just the spike instant for filter)
            # The filter should capture the instantaneous spatial pattern
            sta_sum += white_sig[:, spike_sample]
            valid_spikes += 1

        if valid_spikes > 0:
            # Average to get STA filter
            sta_filter = sta_sum / valid_spikes
            # Normalize to unit norm (like original filters)
            norm = np.linalg.norm(sta_filter)
            if norm > 1e-10:
                sta_filter = sta_filter / norm
            recalibrated_filters[:, mu_idx] = sta_filter
        else:
            # Fallback to original
            recalibrated_filters[:, mu_idx] = original_filters[:, mu_idx]
            print(f"  MU {mu_idx}: No valid spikes for STA, keeping original filter")

    return recalibrated_filters


def compute_normalization_factors(sources, centroids):
    """
    Compute normalization factors to constrain pulse train amplitudes to [0, 1].

    For each MU: normalized = (source - noise_centroid) / (spike_centroid - noise_centroid)
    This maps noise to ~0 and spikes to ~1.

    Args:
        sources: MU source signals (n_mus x n_samples)
        centroids: Dict mapping MU index to [noise_centroid, spike_centroid]

    Returns:
        norm_factors: Dict mapping MU index to (offset, scale) tuple
                     where normalized = (source - offset) / scale
    """
    norm_factors = {}
    n_mus = sources.shape[0]

    for i in range(n_mus):
        if centroids is not None and i in centroids:
            noise_centroid, spike_centroid = centroids[i]
            offset = noise_centroid
            scale = spike_centroid - noise_centroid

            # Avoid division by zero
            if abs(scale) < 1e-10:
                # Fallback: use source statistics
                offset = np.median(sources[i, :])
                scale = np.percentile(sources[i, :], 99) - offset
                if abs(scale) < 1e-10:
                    scale = 1.0

            norm_factors[i] = (offset, scale)
        else:
            # No centroids available, use source statistics
            offset = np.median(sources[i, :])
            scale = np.percentile(sources[i, :], 99) - offset
            if abs(scale) < 1e-10:
                scale = 1.0
            norm_factors[i] = (offset, scale)

    return norm_factors


def normalize_sources(sources, norm_factors, clip=True):
    """
    Apply normalization factors to source signals.

    Args:
        sources: MU source signals (n_mus x n_samples)
        norm_factors: Dict mapping MU index to (offset, scale) tuple
        clip: If True, clip normalized values to [0, 1]

    Returns:
        normalized_sources: Normalized source signals (n_mus x n_samples)
    """
    normalized = sources.copy()
    n_mus = sources.shape[0]

    for i in range(n_mus):
        if norm_factors is not None and i in norm_factors:
            offset, scale = norm_factors[i]
            normalized[i, :] = (sources[i, :] - offset) / scale
            if clip:
                normalized[i, :] = np.clip(normalized[i, :], 0, 1)

    return normalized





def remove_spikes(emg, threshold_std=5):
    """
    Remove large spikes via simple thresholding and linear interpolation.
    
    Parameters
    ----------
    emg : ndarray, shape (n_channels, n_samples)
    threshold_std : float
        Number of standard deviations for threshold
    
    Returns
    -------
    emg_clean : ndarray
    spike_mask : ndarray, bool
    """



    emg_clean = emg.copy()
    spike_mask = np.zeros_like(emg, dtype=bool)
    
    for ch in range(emg.shape[0]):
        signal = emg[ch]
        threshold = threshold_std * np.std(signal)
        
        spikes = np.abs(signal) > threshold
        spike_mask[ch] = spikes
        
        if np.any(spikes):
            clean_idx = np.where(~spikes)[0]
            spike_idx = np.where(spikes)[0]
            
            if len(clean_idx) > 1:
                interp = interp1d(clean_idx, signal[clean_idx], 
                                  kind='linear', bounds_error=False, 
                                  fill_value='extrapolate')
                emg_clean[ch, spike_idx] = interp(spike_idx)
    
    return emg_clean, spike_mask


def decompose_emg(emg_data, config_path, output_folder, trial_name=None, show_plot=True):

    emg_raw = emg_data
    emg_clean, spikes = remove_spikes(emg_raw, threshold_std=8)



    with open(config_path, "r") as f:
        algo_cfg = json.load(f)

    if trial_name is None:
        trial_name = "trial"

    emg_clean = emg_clean[:, 24000: -6000]  #trim edges to avoid filter artifacts
    
    #plot emg before and after spike removal for first channel
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(emg_raw[0], label='Raw EMG', color='red')
        plt.title(f'Raw EMG Signal - Channel 0 - {trial_name}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(emg_clean[0], label='Cleaned EMG', color='blue')
        plt.title(f'Cleaned EMG Signal - Channel 0 - {trial_name}')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()

        plt.tight_layout()
        plt.show()
      
        
    results, metadata = decompose_cbss(
            data=emg_clean,
            algorithm_config=algo_cfg,
            show_config=False, 
    )



    #check if any in results is None
    if any(value is None for value in results.values()):
        raise RuntimeError(f"Decomposition failed for {trial_name}, skipping saving results ...")
        


    print(f"Decomposed {trial_name}, saving results ...")

    decomp_path = os.path.join(output_folder, f"results_{trial_name}.pkl")

    print(f"Saving results to {decomp_path} result keys: {list(results.keys())}")


    with open(decomp_path, "wb") as f:
        pkl.dump((results, metadata), f)

    return results, metadata, decomp_path 


def prompt_trial_selection(n_trials, all_metadata=None):
    """Prompt user to select which trials to use for multi-movement control.

    Args:
        n_trials: Total number of available trials
        all_metadata: Optional list of metadata dicts for each trial

    Returns:
        List of selected trial indices (0-indexed)
    """
    print(f"\n" + "=" * 60)
    print("TRIAL SELECTION FOR MULTI-MOVEMENT CONTROL")
    print("=" * 60)
    print(f"\nAvailable trials: {n_trials}")

    if all_metadata:
        for i, meta in enumerate(all_metadata):
            n_mus = meta.get('n_mus', 'N/A')
            print(f"  Trial {i + 1}: {n_mus} motor units")

    print(f"\nEnter trial numbers to combine (comma-separated, e.g., '1,2' or 'all'):")

    while True:
        response = input("Selection: ").strip().lower()

        if response == 'all':
            return list(range(n_trials))

        try:
            # Parse comma-separated trial numbers (1-indexed input, convert to 0-indexed)
            selected = [int(x.strip()) - 1 for x in response.split(',')]

            # Validate
            if all(0 <= idx < n_trials for idx in selected) and len(selected) > 0:
                print(f"Selected trials: {[i + 1 for i in selected]}")
                return selected
            else:
                print(f"Invalid selection. Enter numbers between 1 and {n_trials}")
        except ValueError:
            print("Invalid input. Enter comma-separated numbers or 'all'")

def compute_combined_whitening_and_centroids( recorded_trials, selected_indices, config_path,
                                                stacked_mu_filters, fsamp):
    """Compute new whitening matrix, recalibrate filters via STA, and compute normalization.

    This implements the Farina 2025 approach:
    1. Concatenate EMG from selected movements
    2. Compute unified whitening matrix
    3. Recalibrate filters via spike-triggered averaging (reintroduces temporal overlaps)
    4. Recalibrate centroids with recalibrated filters
    5. Compute normalization factors to constrain pulse trains to [0, 1]

    Args:
        recorded_trials: List of recorded EMG data dicts with 'data' key
        selected_indices: List of trial indices to combine
        stacked_mu_filters: Horizontally stacked MU filters (n_ext_channels x n_total_mus)
        fsamp: Sampling frequency in Hz
        config_path: Path to algorithm config JSON

    Returns:
        Z_combined: New whitening matrix computed from concatenated signals
        recalibrated_filters: STA-recalibrated MU filters (n_ext_channels x n_mus)
        recalibrated_centroids: Dict mapping MU index to [noise_centroid, spike_centroid]
        norm_factors: Dict mapping MU index to (offset, scale) for [0,1] normalization
        recalibrated_sil: Array of silhouette scores for each MU
        recalibrated_spikes: Dict mapping MU index to spike times
        sources: Source signals (n_mus x n_samples)
        stacked_mu_filters: Original filters (kept for comparison)
    """

    print(f"\n" + "-" * 40)
    print("Computing combined whitening matrix...")

    # Step 1: Concatenate raw EMG signals from selected trials
    selected_emg = [recorded_trials[i]["data"] for i in selected_indices]
    concatenated_emg = np.hstack(selected_emg)
    print(f"Concatenated EMG shape: {concatenated_emg.shape}")
    print(f"Total duration: {concatenated_emg.shape[1] / fsamp:.1f} seconds")

    # Step 2: Load config and create CBSS instance for preprocessing
    with open(config_path, "r") as f:
        algo_cfg = json.load(f)

    cbss = CBSS(**algo_cfg)

    # Step 3: Compute whitening matrix using just_preprocess=True
    # This does: bandpass -> notch -> extension -> whitening
    white_sig, Z_combined = cbss.decompose(concatenated_emg, fsamp, just_preprocess=True)
    print(f"New whitening matrix shape: {Z_combined.shape}")

    n_mus = stacked_mu_filters.shape[1]
    print(f"Processing {n_mus} motor units")

    # Step 4: Recalibrate filters via spike-triggered averaging (Farina 2025)
    # This reintroduces temporal overlaps removed during offline peel-off
    print("\nRecalibrating filters via spike-triggered averaging...")
    recalibrated_filters = recalibrate_filters_sta(
        white_sig=white_sig,
        original_filters=stacked_mu_filters,
        fsamp=fsamp
    )
    print(f"Filter recalibration complete")

    # Step 5: Apply recalibrated filters to get sources
    sources = recalibrated_filters.T @ white_sig
    print(f"Extracted {n_mus} source signals with recalibrated filters")

    # Step 6: Compute normalization factors from RAW sources (Farina 2025)
    # Normalize sources to [0, 1] range BEFORE computing centroids
    print("\nComputing normalization factors from raw sources...")
    norm_factors = {}
    for i in range(n_mus):
        # Use percentile-based normalization on raw source
        offset = np.median(sources[i, :])
        scale = np.percentile(sources[i, :], 99) - offset
        if abs(scale) < 1e-10:
            scale = 1.0
        norm_factors[i] = (offset, scale)
        print(f"  MU {i}: offset={offset:.3f}, scale={scale:.3f}")

    # Step 7: Normalize sources in place
    print("\nNormalizing sources to [0, 1] range...")
    sources_normalized = sources.copy()
    for i in range(n_mus):
        offset, scale = norm_factors[i]
        sources_normalized[i, :] = (sources[i, :] - offset) / scale
        sources_normalized[i, :] = np.clip(sources_normalized[i, :], 0, 1)

    # Step 8: Compute centroids on NORMALIZED sources
    # These centroids will now be in [0, 1] space and match streaming
    print("\nComputing centroids on normalized sources...")
    recalibrated_centroids = {}
    recalibrated_sil = np.zeros(n_mus)
    recalibrated_spikes = {}

    for i in range(n_mus):
        spikes_i, sil_i, centroids_i = est_spike_times(
            sources_normalized[i, :],  # Use normalized sources
            fsamp,
            cluster="kmeans"
        )
        recalibrated_centroids[i] = centroids_i
        recalibrated_sil[i] = sil_i
        recalibrated_spikes[i] = spikes_i

        # Calculate firing rate for logging
        fr = len(spikes_i) / (sources.shape[1] / fsamp) if len(spikes_i) > 0 else 0.0
        print(f"  MU {i}: SIL={sil_i:.3f}, FR={fr:.1f} Hz, {len(spikes_i)} spikes, centroids={centroids_i}")

    print(f"\nRecalibration complete!")
    print("-" * 40)

    return (Z_combined, recalibrated_filters, recalibrated_centroids, norm_factors,
            recalibrated_sil, recalibrated_spikes, sources, stacked_mu_filters)