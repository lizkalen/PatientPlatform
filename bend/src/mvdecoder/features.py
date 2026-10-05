"""Per-window feature extraction for the gate and gesture heads.

The decoder operates on the subset channels only (the muscle-coupled grids). For each
256 ms window it computes:
  * per-grid mean log-RMS (A_Gk), one per subset grid  -> feeds the gate
  * the amplitude deltas dA (A minus A a few windows earlier)  -> feeds the gate
  * the within-subset log-RMS map (lr - mean), grid energy shares, grid median
    frequencies, and the subset mean level A_sub  -> feed the gesture head

`FeatureExtractor.window_kernel` is the single per-window routine used by both the
streaming and offline paths. `AmplitudeHistory` computes dA from the sequence of A
vectors and is shared too, so the two paths produce identical vectors.
`vectors_from_cache` builds the same gate/gesture matrices from a prebuilt feat_v2
cache (178-d F + A), for training or comparison.

numpy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig


def active_subset_grids(cfg: DecoderConfig, good: np.ndarray) -> list:
    """Subset grids that actually contain at least one good channel, in order.
    Grids with no good channels are dropped so their (empty) per-grid features do not
    become NaN."""
    gridof = cfg.gridof(good)
    return [g for g in cfg.subset_grids if np.any(gridof == g)]


class AmplitudeHistory:
    """Running history of per-grid amplitude vectors, for the dA features.

    `push(A)` appends the current A vector and returns dA: for each used lag L,
    A_now - A_(L windows back), clamped to the first window (never reaching before the
    start, never resetting at bout edges). Concatenated in lag order."""

    def __init__(self, lags):
        self.lags = tuple(int(l) for l in lags)
        self.H: list[np.ndarray] = []

    def reset(self):
        self.H = []

    def push(self, A: np.ndarray) -> np.ndarray:
        self.H.append(np.asarray(A, float))
        cur = self.H[-1]
        L = len(self.H)
        if not self.lags:
            return np.zeros(0)
        return np.concatenate([cur - self.H[max(0, L - 1 - lag)] for lag in self.lags])


def deltas_offline(Ag: np.ndarray, E: np.ndarray, lags) -> np.ndarray:
    """Vectorized dA over a whole recording: sort windows by end sample E, take
    Ag - Ag[k back] for each lag (clamped to index 0), then restore the original order.
    Equivalent to AmplitudeHistory over the same windows in time order."""
    Ag = np.asarray(Ag, float)
    E = np.asarray(E, int)
    o = np.argsort(E)
    inv = np.argsort(o)
    As = Ag[o]
    idx = np.arange(len(As))
    return np.hstack([(As - As[np.maximum(idx - k, 0)])[inv] for k in lags])


class FeatureExtractor:
    """Per-window features on the subset channels. Construct with the config and the
    good-channel indices; it resolves the subset rows and per-grid positions."""

    def __init__(self, cfg: DecoderConfig, good: np.ndarray):
        self.cfg = cfg
        keep_pos, grid_pos = cfg.subset_positions(good)             # into good / into subset
        # keep only subset grids that carry good channels (drop empty ones)
        active = [k for k, gp in enumerate(grid_pos) if len(gp) > 0]
        self.active_grids = [cfg.subset_grids[k] for k in active]
        self.keep_pos = keep_pos
        self.grid_pos = [grid_pos[k] for k in active]
        self.n_amps = len(self.grid_pos)
        if self.n_amps == 0:
            raise ValueError("no subset grid contains good channels")
        if cfg.gate_lags > len(cfg.lags):
            raise ValueError(f"gate_lags={cfg.gate_lags} > available lags {len(cfg.lags)}")
        self.used_lags = cfg.lags[:cfg.gate_lags]
        self.freqs = np.fft.rfftfreq(cfg.win, 1.0 / cfg.fs)
        self.gate_dim = self.n_amps * (1 + cfg.gate_lags)
        self.gest_dim = len(self.keep_pos) + 2 * self.n_amps + 1

    def select_subset(self, good_signal: np.ndarray) -> np.ndarray:
        """Take the subset rows from a good-channel signal [n_good, N] -> [n_subset, N]."""
        return good_signal[self.keep_pos]

    def window_kernel(self, seg: np.ndarray):
        """Features for one window seg[n_subset, win].
        Returns (A, spatial, eg, mf, A_sub):
          A       per-grid mean log-RMS               (n_amps,)
          spatial log-RMS minus its subset mean       (n_subset,)
          eg      per-grid energy share (sum to 1)    (n_amps,)
          mf      per-grid median frequency           (n_amps,)
          A_sub   subset mean log-RMS                 scalar
        """
        seg = np.asarray(seg, float)
        rms = np.sqrt((seg ** 2).mean(1)) + 1e-9
        lr = np.log(rms)
        lev = float(lr.mean())
        A = np.array([lr[pos].mean() for pos in self.grid_pos])
        spatial = lr - lev
        Pw = np.abs(np.fft.rfft(seg, axis=1)) ** 2
        eg = np.array([(rms[pos] ** 2).sum() for pos in self.grid_pos])
        eg = eg / (eg.sum() + 1e-12)
        mf = np.empty(self.n_amps)
        for k, pos in enumerate(self.grid_pos):
            cs = np.cumsum(Pw[pos].mean(0))
            mf[k] = self.freqs[np.searchsorted(cs, cs[-1] / 2)]
        return A, spatial, eg, mf, lev

    def gate_vector(self, A: np.ndarray, dA: np.ndarray) -> np.ndarray:
        """Gate feature row: per-grid A followed by the used dA lags."""
        return np.concatenate([A, dA])

    def gesture_vector(self, spatial, eg, mf, A_sub) -> np.ndarray:
        """Gesture feature row: spatial map, grid energies, grid median freqs, A_sub."""
        return np.concatenate([spatial, eg, mf, [A_sub]])


def vectors_from_cache(F: np.ndarray, A: np.ndarray, E: np.ndarray,
                       good: np.ndarray, cfg: DecoderConfig):
    """Build (Xgate, Xgest) from a prebuilt feat_v2 cache.

    F is the per-window feature matrix [n, n_good + 2*n_grids] (per-channel log-RMS
    deviation, then per-grid energies, then per-grid median freqs); A is the mean
    log-RMS per window. Reconstructs the same gate/gesture vectors the from-raw path
    produces, for training or parity checks."""
    F = np.asarray(F, float)
    A = np.asarray(A, float)
    E = np.asarray(E, int)
    gridof = cfg.gridof(good)
    n_good = len(good)
    n_grids = len(cfg.grids)
    subset = active_subset_grids(cfg, good)          # non-empty subset grids
    lags = cfg.lags[:cfg.gate_lags]

    # gate: per-grid mean log-RMS (A + within-grid deviation mean) + dA
    AG = np.column_stack([A + F[:, np.flatnonzero(gridof == g)].mean(1) for g in subset])
    Xgate = np.hstack([AG, deltas_offline(AG, E, lags)]) if lags else AG

    # gesture: within-subset map + grid energy shares + grid median freqs + A_sub
    sub = np.flatnonzero(np.isin(gridof, subset))
    ch = F[:, sub]
    mu = ch.mean(1)
    eg = F[:, n_good:n_good + n_grids][:, subset]
    eg = eg / (eg.sum(1, keepdims=True) + 1e-12)
    mf = F[:, n_good + n_grids:][:, subset]
    Xgest = np.hstack([ch - mu[:, None], eg, mf, (A + mu)[:, None]])
    return Xgate, Xgest
