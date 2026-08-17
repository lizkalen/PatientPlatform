
"""functions for loading pre-computed motor unit filters + whitening matrix and centroids
 for online decomposition.
 """

import pickle as pkl
import numpy as np

def load_pretrained_model(
    decomp_results_path: str,
    fs: float,
    n_mus: int = None
) -> tuple:
    """
    Loads pre-computed motor unit filters + whitening matrix, centroids, and normalization
    for online decomposition.

    Args:
        decomp_results_path: Path to pickle file with decomposition results
        fs: Sampling frequency
        n_mus: Number of MUs to load (None = all)

    Returns:
        tuple: (mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove)
            - mu_filters: (n_extended_channels, n_mus) - STA-recalibrated if available
            - Z: whitening matrix from offline decomposition (expected to be (n_extended_channels, n_extended_channels))
            - n_mus: number of motor units
            - centroids: Dict mapping MU index to [noise_centroid, spike_centroid] or None
            - norm_factors: Dict mapping MU index to (offset, scale) for [0,1] normalization, or None
            - channels_to_remove: List of channel indices that were removed during calibration
    """
    print(f"[INIT] Loading decomposition from: {decomp_results_path}")
    with open(decomp_results_path, "rb") as f:
        results, metadata = pkl.load(f)

    mu_filters = results["mu_filters"]  # Shape: (n_extended_channels, n_mus)

    Z = results.get("Z")
    if Z is None:
        raise KeyError(
            "Offline decomposition did not contain whitening matrix 'Z'. "
            "Expected keys: results['Z'] and results['mu_filters']."
        )

    # Limit number of MUs if requested
    if n_mus is not None:
        n_mus = min(n_mus, mu_filters.shape[1])
        mu_filters = mu_filters[:, :n_mus]
    else:
        n_mus = mu_filters.shape[1]

    print(f"[INIT] Loaded {n_mus} motor unit filters")
    print(f"[INIT] Whitening matrix Z: {Z.shape}")

    n_extended_channels, _ = mu_filters.shape
    if Z.shape[0] != n_extended_channels:
        raise ValueError(
            f"Shape mismatch: mu_filters has {n_extended_channels} extended channels, "
            f"but Z is {Z.shape}."
        )

    # Extract centroids if available (dict: {mu_idx: [noise_centroid, spike_centroid]})
    centroids = results.get("centroids")
    if centroids is not None:
        # If the caller requested a subset of MUs, drop centroids for MUs that are not loaded.
        centroids = {int(k): v for k, v in centroids.items() if int(k) < n_mus}
        for mu_idx, c in centroids.items():
            print(f"       MU{mu_idx}: noise={c[0]:.2f}, spike={c[1]:.2f}")
    else:
        print(f"[INIT] No centroids found - will use K-means for spike detection")

    # Extract normalization factors if available (Farina 2025)
    # Dict: {mu_idx: (offset, scale)} where normalized = (source - offset) / scale
    norm_factors = results.get("norm_factors")
    if norm_factors is not None:
        # If the caller requested a subset of MUs, drop norm_factors for MUs that are not loaded.
        norm_factors = {int(k): v for k, v in norm_factors.items() if int(k) < n_mus}
        print(f"[INIT] Loaded normalization factors for {len(norm_factors)} MUs")
        for mu_idx, (offset, scale) in norm_factors.items():
            print(f"       MU{mu_idx}: offset={offset:.3f}, scale={scale:.3f}")
    else:
        print(f"[INIT] No normalization factors found - sources will not be normalized")

    # Extract channels_to_remove if available (for filtering live stream data)
    channels_to_remove = results.get("channels_to_remove", [])
    if channels_to_remove:
        print(f"[INIT] Channels to remove from live stream: {channels_to_remove}")
    else:
        print(f"[INIT] No channels marked for removal - using all channels")

    print(f"[INIT] Fixed model initialization complete")

    return mu_filters.astype(np.float32), Z.astype(np.float32), n_mus, centroids, norm_factors, channels_to_remove
