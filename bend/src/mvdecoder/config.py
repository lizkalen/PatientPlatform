"""Static configuration for the two-stage movement decoder.

`DecoderConfig` holds the fixed parameters: window geometry, PARRM taps, filter
design, model size, feature selection and the decider's control thresholds. It also
carries the montage (grid spans and which grids the decoder uses). Per-session values
(good-channel mask, trigger threshold, PARRM period P) are supplied at calibration
time, not stored here.

`EXPERLANGEN` is the default preset (256-channel / 4-grid montage, decoder on
ch128-255 = grids 2&3). A different montage needs its own preset with the correct
geometry and `subset_grids`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FS = 2048.0


@dataclass(frozen=True)
class DecoderConfig:
    # ── acquisition / window geometry ───────────────────────────────────────────
    fs: float = FS
    win: int = 524                       # 256 ms window
    step: int = 131                      # 64 ms hop (75% overlap)

    # ── montage geometry: (grid name, start, stop) half-open channel spans ───────
    # subset_grids: grid indices whose channels the decoder uses. Only these run.
    grids: tuple = (("Grid1", 0, 64), ("Grid2", 64, 128),
                    ("Grid3", 128, 192), ("Grid4", 192, 256))
    subset_grids: tuple = (2, 3)         # ch128-255 = Grid3 + Grid4
    n_emg: int = 256                     # EMG channel count (grids only; trigger is separate)

    # ── artifact removal (PARRM) + pulse detection ──────────────────────────────
    parrm_m: int = 24                    # number of comb taps
    parrm_skip: int = 2                  # skip the nearest `skip` cycles -> taps 2P..25P
    parrm_pre: float = 1.0               # stim-on guard before a pulse (periods), offline only
    parrm_post: float = 2.0              # stim-on guard after a pulse (periods)
    refractory: int = 45                 # pulse-detection refractory (samples)
    trig_k: float = 5.0                  # estimate_trig_mid MAD multiplier

    # ── causal filters ──────────────────────────────────────────────────────────
    bp_lo: float = 20.0
    bp_hi: float = 450.0
    notch_base: float = 50.0             # notch every 50 Hz harmonic
    notch_q: float = 30.0

    # ── model ───────────────────────────────────────────────────────────────────
    n_trees: int = 50
    random_state: int = 0
    lags: tuple = (4, 8, 16)             # dA window-lags (256 / 512 / 1024 ms)
    gate_lags: int = 3                   # how many of `lags` the gate uses (3 -> full 8-d
    #   gate on 2 grids; 1 -> the leaner 4-d gate). Gate dim = n_amps * (1 + gate_lags),
    #   where n_amps is the number of non-empty subset grids.

    # ── smoothing / decision ────────────────────────────────────────────────────
    ema_alpha: float = 0.3               # per-head EMA on predict_proba
    ema_reset: bool = False              # reset EMA per contiguous run (gap > 1.5*step)?
    #   False: continuous EMA. True: reset at each contiguous run.
    ema_regime_reset: bool = True        # routed (two-head) path only: at a stim-on/stim-off
    #   change, restart the EMA of the head that takes over from the uniform prior, so a fresh
    #   regime does not inherit the other regime's smoothing. No effect without a clean head.

    # ── decider (move/rest gate), ported from MovementStimController.js ──────────
    on_thr: float = 0.5                  # rest -> move: START when p_move >= on_thr
    off_thr: float = 0.4                 # move -> rest: STOP when p_move < off_thr (hysteresis)
    latch_ms: float = 600.0              # onset latch: hold a fresh train at least this long

    # ── class labels ────────────────────────────────────────────────────────────
    class_names: tuple = ("fist", "trp", "ext", "rest")

    # -- derived helpers ---------------------------------------------------------
    @property
    def rest_idx(self) -> int:
        return len(self.class_names) - 1

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def grid_spans(self) -> list:
        """[(start, stop), ...] in grid order."""
        return [(a, z) for _, a, z in self.grids]

    def gridof(self, good: np.ndarray) -> np.ndarray:
        """Map each good channel index to its grid index 0..len(grids)-1."""
        spans = self.grid_spans
        good = np.asarray(good, int)
        return np.array([next(gi for gi, (a, z) in enumerate(spans) if a <= c < z)
                         for c in good], int)

    def subset_positions(self, good: np.ndarray):
        """Positions (into the `good` row order) of the channels the decoder uses,
        plus the per-grid position lists for the subset grids, in subset_grids order.

        Returns (keep_pos, grid_pos) where keep_pos indexes `good`, and grid_pos[k]
        indexes the KEPT-subset rows (0..len(keep_pos)-1) for subset_grids[k]."""
        go = self.gridof(good)
        keep_pos = np.flatnonzero(np.isin(go, self.subset_grids))
        kg = go[keep_pos]                                  # grid of each kept channel
        grid_pos = [np.flatnonzero(kg == g) for g in self.subset_grids]
        return keep_pos, grid_pos

    def with_(self, **kw) -> "DecoderConfig":
        """A copy with fields overridden (e.g. cfg.with_(gate_lags=1))."""
        return replace(self, **kw)


# Default preset (256-ch / 4-grid montage). DecoderConfig() already matches it.
EXPERLANGEN = DecoderConfig()

# Example preset for the 192-ch / 3-grid preerlplots (dai5) montage. With only 3 grids
# there is no Grid4; subset_grids here is a placeholder and should be set to whichever
# grids carry the muscle signal for that montage before use.
PREERLPLOTS_DAI5 = DecoderConfig(
    n_emg=192,
    grids=(("Grid1", 0, 64), ("Grid2", 64, 128), ("Grid3", 128, 192)),
    subset_grids=(2,),
    class_names=("close", "trp", "ext", "rest"),
)

# exp13072026: 384-EMG-channel montage, 6 grids of 64. Grids 1-3 (ch 0-191) are
# 13x5 / 4 mm; grids 4-6 (ch 192-383) are 8x8 / 10 mm. No trigger channel, so the stim
# time base comes from the saved stimpulses arrays (PulseTimesStimSource). The decoder
# uses all 6 grids; grids with no good channels are dropped automatically, so the gate
# dim adapts to however many subset grids actually carry data.
EXP13072026 = DecoderConfig(
    n_emg=384,
    grids=(("Grid1", 0, 64), ("Grid2", 64, 128), ("Grid3", 128, 192),
           ("Grid4", 192, 256), ("Grid5", 256, 320), ("Grid6", 320, 384)),
    subset_grids=(0, 1, 2, 3, 4, 5),
    class_names=("close", "trp", "ext", "rest"),
)
