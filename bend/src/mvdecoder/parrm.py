"""PARRM artifact removal, offline and streaming.

PARRM estimates the artifact at sample t as the mean of the signal one or more exact
(fractional) periods back, skipping the nearest `skip` cycles, and subtracts it where
stimulation is running (the stim-on mask); other samples pass through unchanged. The
taps reach back 2P..25P, so the estimate needs about 25*P (~817 ms) of history.

`parrm_taps` builds the (floor, frac) tap table shared by both implementations, so
`StreamPARRM` matches `parrm_offline` once its history is full. `parrm_offline`
follows removers.rm_parrm, with the stim-on gate passed in rather than recomputed.

`parrm_offline_noncausal` is the offline-only variant: its taps straddle t, half the
cycles before and half after, so the estimate is centred rather than trailing. Use it
when the whole recording is in memory; use `parrm_offline` when the result has to
match the streaming path. numpy only.
"""
from __future__ import annotations

import numpy as np

from .config import DecoderConfig
from .pulses import stim_on_mask


def parrm_taps(P: float, m: int = 24, skip: int = 2):
    """The comb taps: for each cycle k in [skip, skip+m), the integer floor and the
    fractional remainder of k*P. Each tap reads t-floor and t-floor-1 (both in the
    past) and interpolates between them."""
    taps = []
    for k in range(skip, skip + m):
        d = k * P
        f = int(np.floor(d))
        taps.append((f, d - f))
    return taps


def parrm_offline(notchC: np.ndarray, P: float, on: np.ndarray,
                  m: int = 24, skip: int = 2) -> np.ndarray:
    """Whole-array PARRM on notch-filtered signal notchC[C, N]. `on` is the boolean
    stim-on gate [N]; the estimate is subtracted only where it is True. Matches
    removers.rm_parrm when on == stim_on_mask."""
    notchC = np.asarray(notchC, float)
    C, N = notchC.shape
    x = notchC
    acc = np.zeros((C, N))
    cnt = np.zeros(N)
    idx = np.arange(N)
    for f, frac in parrm_taps(P, m, skip):
        valid = idx - f - 1 >= 0
        if not valid.any():
            continue
        j0 = np.clip(idx - f, 0, N - 1)
        j1 = np.clip(idx - f - 1, 0, N - 1)
        src = np.zeros((C, N))
        src[:, valid] = (1 - frac) * x[:, j0[valid]] + frac * x[:, j1[valid]]
        acc[:, valid] += src[:, valid]
        cnt[valid] += 1
    est = np.zeros((C, N))
    nz = cnt > 0
    est[:, nz] = acc[:, nz] / cnt[nz]
    y = x - est
    on = np.asarray(on, bool)
    y[:, ~on] = x[:, ~on]                     # gate: pass clean/non-stim samples through
    return y


def parrm_taps_symmetric(P: float, m: int = 24, skip: int = 2):
    """Taps for the non-causal comb: for each cycle k in [skip, skip + m // 2), the
    (floor, frac, sign) of k*P in both directions. sign -1 reads backwards from t,
    sign +1 reads forwards. Half the cycles per side keeps the total tap count at m."""
    taps = []
    for k in range(skip, skip + max(1, m // 2)):
        d = k * P
        f = int(np.floor(d))
        frac = d - f
        taps.append((f, frac, -1))
        taps.append((f, frac, +1))
    return taps


def parrm_offline_noncausal(notchC: np.ndarray, P: float, on: np.ndarray,
                            m: int = 24, skip: int = 2) -> np.ndarray:
    """Non-causal PARRM: the artifact estimate averages taps on BOTH sides of t.

    `parrm_offline` reaches only backwards (t - kP), which is the most a streaming
    implementation can do. Offline both sides are available, so this takes m // 2
    cycles before and m // 2 after, k in [skip, skip + m // 2). The estimate is
    centred on t rather than trailing it, so it does not lag when the period drifts,
    and it needs about (skip + m // 2) * P of signal on each side instead of
    (skip + m) * P of history on one. Samples near either edge average whichever
    taps are in range, as in the causal version.

    Args:
        notchC (ndarray): Notch-filtered signal, (n_channels, n_samples).
        P (float): Stimulation period in samples, fractional.
        on (ndarray): Boolean stim-on gate, (n_samples,). The estimate is subtracted
            only where True; other samples pass through unchanged.
        m (int): Total number of comb taps, split evenly between the two sides.
        skip (int): Number of nearest cycles to skip on each side.

    Returns:
        ndarray: Cleaned signal, same shape as ``notchC``.
    """
    notchC = np.asarray(notchC, float)
    C, N = notchC.shape
    x = notchC
    acc = np.zeros((C, N))
    cnt = np.zeros(N)
    idx = np.arange(N)

    for f, frac, sign in parrm_taps_symmetric(P, m, skip):
        if sign < 0:
            valid = idx - f - 1 >= 0
            j0 = np.clip(idx - f, 0, N - 1)
            j1 = np.clip(idx - f - 1, 0, N - 1)
        else:
            valid = idx + f + 1 <= N - 1
            j0 = np.clip(idx + f, 0, N - 1)
            j1 = np.clip(idx + f + 1, 0, N - 1)
        if not valid.any():
            continue
        src = np.zeros((C, N))
        src[:, valid] = (1 - frac) * x[:, j0[valid]] + frac * x[:, j1[valid]]
        acc[:, valid] += src[:, valid]
        cnt[valid] += 1

    est = np.zeros((C, N))
    nz = cnt > 0
    est[:, nz] = acc[:, nz] / cnt[nz]
    y = x - est
    on = np.asarray(on, bool)
    y[:, ~on] = x[:, ~on]                     # gate: pass clean/non-stim samples through
    return y


def refine_period(notchC: np.ndarray, pulses: np.ndarray, P0: float, cfg: DecoderConfig,
                  half: float = 0.15, coarse: float = 0.02, fine: float = 0.004,
                  n_chan: int = 40, seg_s: float = 12.0, n_harm: int = 5) -> float:
    """Refine the PARRM period by minimising residual stim-harmonic power.

    With no trigger, P from the pulse spacing can be off by more than PARRM's tolerance.
    This runs PARRM on a stim-on slice for a grid of periods around P0 and returns the one
    that leaves the least power at the first `n_harm` stim harmonics (a two-stage coarse
    then fine search). Returns P0 unchanged if there is not enough stim-on data to judge."""
    from scipy.signal import welch
    pulses = np.asarray(pulses, int)
    N = notchC.shape[1]
    mid = int(pulses[len(pulses) // 3])
    warm = int(np.ceil((cfg.parrm_skip + cfg.parrm_m) * P0)) + 2000
    s0 = max(0, mid - warm)
    s1 = min(N, mid + int(seg_s * cfg.fs))
    plocal = pulses[(pulses >= s0) & (pulses < s1)] - s0
    if len(plocal) < 10 or s1 - s0 < warm + 4096:
        return P0
    sl = notchC[:, s0:s1]
    rms = np.sqrt((sl ** 2).mean(1))
    ch = np.argsort(rms)[-min(n_chan, sl.shape[0]):]            # loudest channels
    sub = sl[ch]
    ana = slice(warm, sub.shape[1])
    nper = min(4096, sub[:, ana].shape[1])

    def residual(P):
        on = stim_on_mask(sub.shape[1], plocal, P, cfg.parrm_pre, cfg.parrm_post)
        y = parrm_offline(sub, P, on, cfg.parrm_m, cfg.parrm_skip)
        f_, Pxx = welch(y[:, ana], fs=cfg.fs, nperseg=nper, axis=1)
        Pm = Pxx.mean(0); f0 = cfg.fs / P
        return sum(float(Pm[np.argmin(np.abs(f_ - f0 * h))]) for h in range(1, n_harm + 1))

    grid = np.arange(P0 - half, P0 + half + 1e-9, coarse)
    best = min(grid, key=residual)
    fgrid = np.arange(best - coarse, best + coarse + 1e-9, fine)
    return float(min(fgrid, key=residual))


class StreamPARRM:
    """Chunk-agnostic streaming PARRM. Keeps the last `need` samples of history so the
    estimate for any chunk length can reach the full 25*P back. `push(blk)` returns the
    artifact-removed chunk, ungated (the caller applies the stim-on gate). While the
    history is still filling, the missing samples read as zeros; after that it matches
    `parrm_offline`. Uses the same `parrm_taps` table as the offline path.
    """

    def __init__(self, C: int, P: float, m: int = 24, skip: int = 2):
        self.taps = parrm_taps(P, m, skip)
        self.need = max(f for f, _ in self.taps) + 2       # tap reach + interp neighbour
        self.hist = np.zeros((C, self.need), float)         # samples immediately before the block

    def push(self, blk: np.ndarray) -> np.ndarray:
        blk = np.asarray(blk, float)
        C, n = blk.shape
        full = np.concatenate([self.hist, blk], axis=1)     # [C, need + n]
        base = self.need                                    # index of blk[0] within `full`
        acc = np.zeros((C, n))
        cnt = 0
        for f, frac in self.taps:
            i0 = base - f                                   # >= 2, since need = fmax + 2
            acc += (1 - frac) * full[:, i0:i0 + n] + frac * full[:, i0 - 1:i0 - 1 + n]
            cnt += 1
        est = acc / cnt if cnt else np.zeros((C, n))
        self.hist = full[:, -self.need:]                    # roll history forward
        return blk - est
