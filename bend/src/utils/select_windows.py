"""Interactively pick movement/exclude windows on any EMG array.

CLI usage
    python select_windows_generic.py <emg.npy> --fs 2000
    python select_windows_generic.py <emg.npz>
        npz keys: emg [n_channels, n_samples] (required), fs, channels, on (optional)

    By default the masks save next to the input file as
    <stem>_mask.npy and <stem>_exclude.npy. Override with --mask/--exclude.

As a library
    from select_windows_generic import select_windows
    select_windows(emg, fs, channels=None, on=None,
                   mask_path="mask.npy", exclude_path="exclude.npy", label="")

Controls
    left-drag      add a span in the current mode (bout / exclude)
    right-click    remove the span under the cursor
    e              toggle mode: movement bout  <->  exclude zone
    n / p          next / previous channel
    a              cycle amplitude scale
    s              save masks
    q              quit (matplotlib close also works)
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

plt.rcParams["toolbar"] = "none"


class WindowSelector:
    def __init__(self, emg, fs, channels=None, on=None, mask_path="mask.npy",
                 exclude_path="exclude.npy", label=""):
        self.emg = np.asarray(emg, float)
        self.fs = float(fs)
        self.N = self.emg.shape[1]
        self.channels = np.asarray(channels if channels is not None else np.arange(self.emg.shape[0]))
        self.on = np.zeros(self.N, bool) if on is None else np.asarray(on, bool)
        self.label = label
        self.mask_p, self.excl_p = mask_path, exclude_path
        self.order = np.argsort(np.sqrt((self.emg ** 2).mean(1)))[::-1]
        self.ci = 0
        self.scale = 1.0
        self.mode = "bout"
        self._save_note = ""
        self.bouts = self._load_runs(self.mask_p)
        self.excludes = self._load_runs(self.excl_p)

        self.fig, self.ax = plt.subplots(figsize=(14, 5))
        self.span = SpanSelector(self.ax, self._onselect, "horizontal", useblit=True,
                                 props=dict(alpha=0.2, facecolor="tab:blue"), interactive=False)
        self.fig.canvas.mpl_connect("button_press_event", self._onclick)
        self.fig.canvas.mpl_connect("key_press_event", self._onkey)
        self._draw()
        plt.show(block=True)

    def _load_runs(self, path):
        if not os.path.exists(path):
            return []
        m = np.load(path).astype(bool)
        m = m[:self.N] if m.shape[0] >= self.N else np.pad(m, (0, self.N - m.shape[0]))
        d = np.diff(np.r_[0, m.view(np.int8), 0])
        return [[int(s), int(e)] for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]

    def _to_mask(self, runs):
        m = np.zeros(self.N, bool)
        for s, e in runs:
            m[max(0, s):min(self.N, e)] = True
        return m

    def _onselect(self, xmin, xmax):
        s, e = int(xmin * self.fs), int(xmax * self.fs)
        if e - s < int(0.2 * self.fs):
            return
        (self.bouts if self.mode == "bout" else self.excludes).append([s, e])
        self._draw()

    def _onclick(self, event):
        if event.button != 3 or event.xdata is None:      # right-click removes
            return
        x = event.xdata * self.fs
        for runs in (self.bouts, self.excludes):
            for i, (s, e) in enumerate(runs):
                if s <= x <= e:
                    runs.pop(i); self._draw(); return

    def _onkey(self, event):
        if event.key == "e":
            self.mode = "exclude" if self.mode == "bout" else "bout"
        elif event.key == "n":
            self.ci = (self.ci + 1) % len(self.order)
        elif event.key == "p":
            self.ci = (self.ci - 1) % len(self.order)
        elif event.key == "a":
            self.scale = {1.0: 2.0, 2.0: 0.5, 0.5: 1.0}[self.scale]
        elif event.key == "s":
            self.save(); return
        elif event.key == "q":
            plt.close(self.fig); return
        else:
            return
        self._draw()

    def _draw(self):
        self.ax.clear()
        ch = self.order[self.ci]
        y = self.emg[ch]
        t = np.arange(self.N) / self.fs
        for s, e in _runs(self.on):
            self.ax.axvspan(s / self.fs, e / self.fs, color="0.90", lw=0)
        self.ax.plot(t, y, color="0.15", lw=0.3)
        for s, e in self.bouts:
            self.ax.axvspan(s / self.fs, e / self.fs, color="tab:blue", alpha=0.25)
        for s, e in self.excludes:
            self.ax.axvspan(s / self.fs, e / self.fs, color="tab:red", alpha=0.25, hatch="//")
        lim = (np.nanpercentile(np.abs(y), 99.7) or 1.0) / self.scale
        self.ax.set_ylim(-1.3 * lim, 1.3 * lim)
        self.ax.set_xlim(0, t[-1])
        self.ax.set_xlabel("time (s)")
        title = f"{self.label}  " if self.label else ""
        saved = f"   [{self._save_note}]" if self._save_note else ""
        self.ax.set_title(
            f"{title}ch{self.channels[ch]} ({self.ci+1}/{len(self.order)})   "
            f"mode=[{self.mode}]   bouts={len(self.bouts)} excl={len(self.excludes)}   "
            f"(left-drag add · right-click remove · e mode · n/p ch · a scale · s save · q quit){saved}",
            fontsize=9, color=("tab:green" if self._save_note else "black"))
        self._save_note = ""
        self.fig.canvas.draw_idle()

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.mask_p)), exist_ok=True)
        np.save(self.mask_p, self._to_mask(self.bouts))
        if self.excludes:
            np.save(self.excl_p, self._to_mask(self.excludes))
        elif os.path.exists(self.excl_p):
            os.remove(self.excl_p)
        self._save_note = (f"SAVED: {len(self.bouts)} bouts -> {os.path.basename(self.mask_p)}"
                            + (f", {len(self.excludes)} excl" if self.excludes else ""))
        print(self._save_note)
        self._draw()


def _runs(mask):
    m = np.asarray(mask, bool)
    d = np.diff(np.r_[0, m.view(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def select_windows(emg, fs, channels=None, on=None, mask_path="mask.npy",
                   exclude_path="exclude.npy", label=""):
    """Open the interactive window selector on emg [n_channels, n_samples]."""
    WindowSelector(emg, fs, channels=channels, on=on, mask_path=mask_path,
                   exclude_path=exclude_path, label=label)


def _load_emg(path, fs_arg):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = np.load(path)
        if "emg" not in d:
            raise SystemExit(f"{path}: npz has no 'emg' key (found {list(d.keys())})")
        emg = d["emg"]
        fs = fs_arg if fs_arg is not None else float(d["fs"]) if "fs" in d else None
        channels = d["channels"] if "channels" in d else None
        on = d["on"] if "on" in d else None
    else:
        emg = np.load(path)
        fs, channels, on = fs_arg, None, None
    if fs is None:
        raise SystemExit("no sampling rate found; pass --fs")
    return np.asarray(emg), float(fs), channels, on


def main():
    ap = argparse.ArgumentParser(description="Interactively pick movement/exclude windows on an EMG array.")
    ap.add_argument("emg_path", help=".npy [n_channels, n_samples], or .npz with an 'emg' key")
    ap.add_argument("--fs", type=float, default=None, help="sampling rate in Hz (required for .npy input)")
    ap.add_argument("--mask", default=None, help="output path for the bout mask (default: <stem>_mask.npy)")
    ap.add_argument("--exclude", default=None, help="output path for the exclude mask (default: <stem>_exclude.npy)")
    ap.add_argument("--label", default=None, help="title label (default: input file name)")
    args = ap.parse_args()

    emg, fs, channels, on = _load_emg(args.emg_path, args.fs)
    stem = os.path.splitext(args.emg_path)[0]
    mask_path = args.mask or f"{stem}_mask.npy"
    exclude_path = args.exclude or f"{stem}_exclude.npy"
    label = args.label if args.label is not None else os.path.basename(args.emg_path)

    print(f"{args.emg_path}: emg {emg.shape}, fs={fs:g}. Draw bouts, press s to save, q to quit.")
    select_windows(emg, fs, channels=channels, on=on, mask_path=mask_path,
                   exclude_path=exclude_path, label=label)


if __name__ == "__main__":
    main()
