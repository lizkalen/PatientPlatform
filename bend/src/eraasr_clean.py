"""ERAASR-based cleaning of continuous TENS-stimulated HD-EMG.

Adapts the batch ERAASR PCA-regression cleaner (the `pyeraasr` port) to a
continuous, periodic (TENS) stimulation artifact, as a drop-in alternative to
PARRM for the movement_disc_1307 pipeline.

Idea: within the stim-on region the artifact is a comb at the (fractional)
stim period P. We cut the stim-on signal into TENS cycles delimited by detected
pulses, resample each cycle to a fixed length L, stack them into a
[channels x L x n_cycles] tensor, and apply ERAASR's two artifact-specific
passes via `pyeraasr.clean_matrix_via_pca_regression`:
  * over-channels : removes the spatially-shared artifact component
  * over-pulses   : removes the component that repeats cycle-to-cycle (the comb)
The estimated artifact is mapped back to each cycle's original samples and
subtracted; non-stim samples pass through unchanged (same gating as PARRM).

Batch/offline by construction (uses all cycles) -- appropriate for offline MU
decomposition; NOT causal/streaming.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from pyeraasr.clean import clean_matrix_via_pca_regression


def _resample_to(x, L):
    """Linear-resample [C, n] -> [C, L] along time."""
    n = x.shape[1]
    if n == L:
        return x.astype(float, copy=True)
    t = np.linspace(0.0, 1.0, n)
    tt = np.linspace(0.0, 1.0, L)
    return interp1d(t, x, axis=1, kind="linear", fill_value="extrapolate")(tt)


def eraasr_clean_continuous(
    notchC,
    pulses,
    P,
    on,
    *,
    n_pc_channels=8,
    n_pc_pulses=4,
    omit_channels=1,
    omit_pulses=3,
    pca_only_omitted=False,
    cycle_len=None,
    min_frac=0.6,
    max_frac=1.6,
    clean_channels=True,
    clean_pulses=True,
    verbose=False,
):
    """Remove the TENS comb from a notch-filtered signal via ERAASR passes.

    Parameters
    ----------
    notchC : [C, N] float   notch-filtered good-channel signal.
    pulses : [K] int        stim pulse sample indices.
    P      : float          fractional stim period (samples).
    on     : [N] bool       stim-on gate (artifact removed only where True).
    n_pc_channels / n_pc_pulses : PCs kept for each pass (0 disables the pass).
    omit_channels / omit_pulses : leave-out bandwidth (total, incl. target).
    pca_only_omitted : ERAASR leave-out mode (False = fast: build PCs once,
        zero omitted loadings; True = refit PCA per variable, slow for many cycles).
    cycle_len : resample length per cycle (default round(P)).
    min_frac/max_frac : keep only cycles whose length is in [min_frac,max_frac]*P.

    Returns
    -------
    y : [C, N] float   cleaned signal (artifact subtracted in stim-on region).
    info : dict        diagnostics (n_cycles, cycle_len, artifact rms, ...).
    """
    notchC = np.asarray(notchC, float)
    C, N = notchC.shape
    on = np.asarray(on, bool)
    pulses = np.unique(np.asarray(pulses, int))
    pulses = pulses[(pulses >= 0) & (pulses < N)]
    L = int(round(P)) if cycle_len is None else int(cycle_len)

    # candidate cycles = consecutive pulse pairs that are stim-on and ~one period long
    if len(pulses) < 3:
        return notchC.copy(), {"n_cycles": 0, "reason": "too few pulses"}
    starts, stops = pulses[:-1], pulses[1:]
    lens = stops - starts
    keep = (
        (lens >= min_frac * P) & (lens <= max_frac * P)
        & on[starts] & on[np.clip(stops - 1, 0, N - 1)]
    )
    starts, stops = starts[keep], stops[keep]
    K = len(starts)
    if K < 8:
        return notchC.copy(), {"n_cycles": int(K), "reason": "too few cycles"}

    # cycle tensor A: [C, L, K]
    A = np.empty((C, L, K), float)
    for k in range(K):
        A[:, :, k] = _resample_to(notchC[:, starts[k]:stops[k]], L)

    art_total = np.zeros_like(A)
    cur = A

    if clean_channels and n_pc_channels > 0:
        mat = cur.transpose(1, 2, 0).reshape(L * K, C)                 # obs=(l,k) x channels
        _, _, art = clean_matrix_via_pca_regression(
            mat, n_pc_channels, omit=omit_channels, pca_only_omitted=pca_only_omitted)
        art1 = art.reshape(L, K, C).transpose(2, 0, 1)                 # [C,L,K]
        art_total = art_total + art1
        cur = A - art_total

    if clean_pulses and n_pc_pulses > 0:
        mat2 = cur.transpose(1, 0, 2).reshape(L * C, K)                # obs=(l,c) x cycles
        _, _, art = clean_matrix_via_pca_regression(
            mat2, n_pc_pulses, omit=omit_pulses, pca_only_omitted=pca_only_omitted)
        art2 = art.reshape(L, C, K).transpose(1, 0, 2)                 # [C,L,K]
        art_total = art_total + art2
        cur = A - art_total

    # map the artifact back to each cycle's native samples and subtract
    y = notchC.copy()
    for k in range(K):
        Lk = stops[k] - starts[k]
        art = _resample_to(art_total[:, :, k], Lk)
        y[:, starts[k]:stops[k]] -= art

    info = {
        "n_cycles": int(K), "cycle_len": int(L),
        "artifact_rms": float(np.sqrt((art_total ** 2).mean())),
        "signal_rms": float(np.sqrt((A ** 2).mean())),
    }
    if verbose:
        print(f"  ERAASR: {K} cycles (L={L}), artifact rms {info['artifact_rms']:.2f} "
              f"vs signal rms {info['signal_rms']:.2f}")
    return y, info


# --------------------------------------------------------------------------- #
# synthetic self-test: TENS comb + independent "MU" bumps -> comb removed,
# bumps preserved. Run: python eraasr_clean.py
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.default_rng(0)
    fs = 2048.0
    P = 67.4
    N = 60000
    C = 40
    t = np.arange(N)

    # periodic artifact: shared spatial pattern x stereotyped per-cycle waveform
    spatial = rng.standard_normal(C)
    phase = (t % P) / P
    art_wave = np.sin(2 * np.pi * phase) + 0.4 * np.sin(4 * np.pi * phase)
    artifact = 50.0 * np.outer(spatial, art_wave)

    # "neural": a few channel-local bumps NOT locked to the stim period
    neural = np.zeros((C, N))
    for _ in range(400):
        c = rng.integers(0, C)
        s = rng.integers(20, N - 20)
        neural[c, s - 5:s + 5] += rng.uniform(3, 6) * np.hanning(10)

    noise = 0.5 * rng.standard_normal((C, N))
    x = artifact + neural + noise

    pulses = np.arange(0, N - 1, P).astype(int)
    on = np.ones(N, bool)

    y, info = eraasr_clean_continuous(x, pulses, P, on, n_pc_channels=6, n_pc_pulses=4,
                                      verbose=True)

    # metrics: artifact power removed globally; neural preserved AT THE BUMPS
    art_before = np.sqrt(((x - neural - noise) ** 2).mean())
    art_after = np.sqrt(((y - neural - noise) ** 2).mean())      # residual artifact
    # neural preservation measured on bump support (where the MU-like signal lives)
    mask = np.abs(neural) > 0.5
    corr_bumps = np.corrcoef((y - noise)[mask], neural[mask])[0, 1]
    peak_ratio = np.median((y - noise)[mask] / neural[mask])
    print(f"  artifact rms before {art_before:.2f} -> residual after {art_after:.2f} "
          f"({100*(1-art_after/art_before):.1f}% removed)")
    print(f"  neural @ bumps: corr {corr_bumps:.3f}  median amp-ratio {peak_ratio:.3f} (want ~1)")
    assert art_after < 0.25 * art_before, "artifact not sufficiently removed"
    assert corr_bumps > 0.9, "neural signal not preserved at bumps"
    assert 0.8 < peak_ratio < 1.2, "neural amplitude distorted"
    print("  SELF-TEST PASSED")


if __name__ == "__main__":
    _selftest()
