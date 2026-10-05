"""Interactive good/bad channel reviewer for an EMG array.

Import and call `review_channels(...)` from another script, or run standalone on a saved
EMG array:

    python review_channels.py <emg.npy> --save <good_mask.npy> [--seed <seed_mask.npy>]
                               [--fs FS] [--label TEXT] [--no-grids]

`emg.npy` must hold a [n_channels, n_samples] array. With no --seed, the mask is seeded by
a per-grid RMS-outlier heuristic (pass --no-grids to seed all channels GOOD instead).

Controls
    g   mark GOOD and go to next
    b   mark BAD and go to next
    space / u   toggle current (no advance)
    n / right   next channel        p / left   previous channel
    s   save        q   quit
Bad channels are shown in red; the title tracks how many are currently bad.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt                     # interactive backend (do NOT force Agg)


# def _repo_root(start):
#     """Nearest ancestor of `start` that contains both bend/ and processing/ -- the repo
#     root. Found by search, not a fixed count of `..`, so this script keeps resolving
#     bend/src wherever it is moved inside the repo."""
#     d = os.path.abspath(start)
#     while d != os.path.dirname(d):
#         if os.path.isdir(os.path.join(d, "bend")) and os.path.isdir(os.path.join(d, "processing")):
#             return d
#         d = os.path.dirname(d)
#     raise RuntimeError("repo root not found (no ancestor with both bend/ and processing/)")


# HERE = os.path.dirname(os.path.abspath(__file__))
# REPO = _repo_root(HERE)
# sys.path.insert(0, os.path.join(REPO, "bend", "src"))
from src.mvdecoder import EXP13072026 as CFG


def auto_good_mask(rms, grids, lo=0.2, hi=5.0):
    """Per-grid robust RMS-outlier mask (True = GOOD channel): a channel is bad if its RMS
    is > hi x or < lo x its grid's median. Mirrors server.movement_training.bad_channel_mask;
    used to SEED the review when no seed mask is given. With no grids, every channel seeds
    GOOD (there is no per-grid median to compare against)."""
    rms = np.asarray(rms, float)
    if not grids:
        return np.ones(len(rms), bool)
    good = np.ones(len(rms), bool)
    for _, a, b in grids:
        seg = rms[a:b]
        med = np.median(seg)
        if med <= 0:
            good[a:b] = False
            continue
        ratio = seg / med
        good[a:b] = ~((ratio > hi) | (ratio < lo) | ~np.isfinite(ratio))
    return good


class ChannelReviewer:
    def __init__(self, emg, save_path, fs, seed_mask=None, grids=None, label=""):
        self.clean, self.fs = emg, fs
        self.nch = emg.shape[0]
        self.save_p = save_path
        self.label = label
        self.rms = np.sqrt((emg ** 2).mean(1))

        if seed_mask is not None:
            base = np.asarray(seed_mask, bool)
            self.good = (base[:self.nch] if base.shape[0] >= self.nch
                         else np.pad(base, (0, self.nch - base.shape[0]), constant_values=True)).copy()
            print(f"seed: given mask ({int(self.good.sum())}/{self.nch} good)")
        else:
            self.good = auto_good_mask(self.rms, grids)
            print(f"seed: auto RMS-outlier mask ({int(self.good.sum())}/{self.nch} good, "
                  f"bad {np.where(~self.good)[0].tolist()})")

        self.env = np.abs(emg).mean(0)                      # activity, to pick a zoom window
        self.i = 0
        self._save_note = ""
        self.grids = grids
        if grids:
            spans = [(a, b) for _, a, b in grids]
            self.grid_of = np.array([next((gi for gi, (a, b) in enumerate(spans) if a <= c < b), -1)
                                      for c in range(self.nch)])
            self.grid_names = [g for g, _, _ in grids] + ["?"]
        else:
            self.grid_of = None

        self.fig, (self.a0, self.a1) = plt.subplots(2, 1, figsize=(13, 6))
        self.fig.canvas.mpl_connect("key_press_event", self._key)
        self._draw()
        plt.show(block=True)

    def _key(self, e):
        if e.key in ("g", "b"):
            self.good[self.i] = (e.key == "g"); self.i = min(self.i + 1, self.nch - 1)
        elif e.key in (" ", "u"):
            self.good[self.i] = not self.good[self.i]
        elif e.key in ("n", "right"):
            self.i = min(self.i + 1, self.nch - 1)
        elif e.key in ("p", "left"):
            self.i = max(self.i - 1, 0)
        elif e.key == "s":
            np.save(self.save_p, self.good)
            self._save_note = (f"SAVED: {os.path.basename(self.save_p)}, "
                               f"{int(self.good.sum())}/{self.nch} good")
            print(self._save_note)
        elif e.key == "q":
            plt.close(self.fig); return
        else:
            return
        self._draw()

    def _draw(self):
        y = self.clean[self.i]
        t = np.arange(len(y)) / self.fs
        good = bool(self.good[self.i])
        col = "0.15" if good else "tab:red"
        self.a0.clear(); self.a1.clear()
        self.a0.plot(t, y, color=col, lw=0.3)
        lim = (np.nanpercentile(np.abs(y), 99.7) or 1.0)
        self.a0.set_ylim(-1.3 * lim, 1.3 * lim); self.a0.set_xlim(0, t[-1])
        self.a0.set_ylabel("full")
        # zoom on the most active 2 s
        c = int(np.argmax(np.convolve(self.env, np.ones(int(self.fs)) / self.fs, "same")))
        lo, hi = max(0, c - int(self.fs)), min(len(y), c + int(self.fs))
        self.a1.plot(t[lo:hi], y[lo:hi], color=col, lw=0.6)
        self.a1.set_xlim(t[lo], t[hi - 1]); self.a1.set_xlabel("time (s)"); self.a1.set_ylabel("zoom")
        gname = f" ({self.grid_names[self.grid_of[self.i]]})" if self.grids else ""
        prefix = f"{self.label}  " if self.label else ""
        saved = f"   [{self._save_note}]" if self._save_note else ""
        self.a0.set_title(
            f"{prefix}ch {self.i}/{self.nch-1}{gname}  RMS {self.rms[self.i]:.2f}  "
            f"[{'GOOD' if good else 'BAD'}]   bad so far: {int((~self.good).sum())}   "
            f"(g good · b bad · space toggle · n/p or arrows · s save · q quit){saved}",
            fontsize=9, color=("tab:green" if self._save_note else ("black" if good else "tab:red")))
        self._save_note = ""
        self.fig.canvas.draw_idle()


def review_channels(emg, save_path, fs=None, seed_mask=None, grids=None, label=""):
    """Step through every row of `emg` ([n_channels, n_samples]) in an interactive GUI,
    seeded from `seed_mask` (or an auto RMS-outlier mask if none is given). Saves the
    good/bad mask to `save_path` on 's'. `grids` is an optional
    list of (name, start, stop) channel spans -- pass CFG.grids when `emg`'s row order
    matches the recording's absolute channel indices, or leave it None for an arbitrary
    channel subset (no per-grid grouping in the title / auto-seed)."""
    ChannelReviewer(emg, save_path, fs if fs is not None else CFG.fs, seed_mask, grids, label)


def _load_array(path, key=None):
    if path.endswith(".npz"):
        d = np.load(path)
        if key is None:
            key = "emg" if "emg" in d.files else d.files[0]
        return d[key]
    return np.load(path)


def main():
    ap = argparse.ArgumentParser(description="Review a good/bad channel mask for a given EMG array.")
    ap.add_argument("emg", help="path to a .npy (or .npz) file holding [n_channels, n_samples]")
    ap.add_argument("--save", required=True, help="path to write the reviewed good_mask.npy to")
    ap.add_argument("--seed", default=None, help="path to a seed good_mask.npy (default: auto RMS-outlier)")
    ap.add_argument("--key", default=None, help="array key to use, if --emg is a .npz")
    ap.add_argument("--fs", type=float, default=CFG.fs, help=f"sample rate (default: CFG.fs={CFG.fs})")
    ap.add_argument("--label", default="", help="text shown in the plot title")
    ap.add_argument("--no-grids", action="store_true", help="disable per-grid grouping (use CFG.grids by default)")
    args = ap.parse_args()

    emg = np.asarray(_load_array(args.emg, args.key), dtype=np.float32)
    seed = np.load(args.seed).astype(bool) if args.seed else None
    grids = None if args.no_grids else CFG.grids

    print(f"{os.path.basename(args.emg)}: {emg.shape}. Review channels; g/b to mark, s to save.")
    review_channels(emg, args.save, fs=args.fs, seed_mask=seed, grids=grids, label=args.label)


if __name__ == "__main__":
    main()
