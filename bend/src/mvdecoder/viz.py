"""Plotting helpers for decoder results.

matplotlib is imported lazily inside the functions, so importing the rest of the
package never requires it. All functions save to `save` if given and return the figure.

  confusion_heatmap    - one confusion matrix as a row-normalised (recall) heatmap
  transfer_summary_bars - grouped macro-F1 bars across metric x (family, path)
  recording_timeline   - truth vs decoder over time, with p_move and the gate thresholds
  stages_timeline      - one class-strip row per head/stage, each drawn only where it ran
"""
from __future__ import annotations

import numpy as np

_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def confusion_heatmap(M, class_names, title="", ax=None, save=None):
    """Row-normalised (recall) confusion heatmap with raw counts annotated."""
    plt = _plt()
    M = np.asarray(M, float)
    row = M.sum(1, keepdims=True)
    norm = M / np.where(row > 0, row, 1)
    own = ax is not None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
    else:
        fig = ax.figure
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("truth")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{norm[i, j]*100:.0f}\n{int(M[i, j])}", ha="center", va="center",
                    fontsize=8, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="recall")
    if not own:
        fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


def transfer_summary_bars(records, title="", save=None):
    """Grouped bars of macro-F1 across metrics.
    records: list of dicts {"family","path","metric","value"} (value in [0,1])."""
    plt = _plt()
    metrics = list(dict.fromkeys(r["metric"] for r in records))
    groups = list(dict.fromkeys((r["family"], r["path"]) for r in records))
    lut = {(r["family"], r["path"], r["metric"]): r["value"] for r in records}

    fig, ax = plt.subplots(figsize=(max(7, 1.4 * len(metrics)), 4.2))
    x = np.arange(len(metrics))
    w = 0.8 / max(1, len(groups))
    for gi, (fam, path) in enumerate(groups):
        vals = [lut.get((fam, path, m), np.nan) * 100 for m in metrics]
        ax.bar(x + gi * w, vals, w, label=f"{fam} / {path}", color=_PALETTE[gi % len(_PALETTE)])
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylabel("macro-F1 (%)"); ax.set_ylim(0, 100); ax.axhline(50, ls=":", c="gray", lw=0.8)
    ax.set_title(title, fontsize=10); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


# Class-strip colours matching the analysis figures on the drive:
#   movement classes = blue / orange / green, stim-rest = grey, clean rest = light gray.
_MOVE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
_STIM_REST = "0.55"          # grey  (rest while stim is on)
_CLEAN_REST = "0.86"         # light gray (rest while stim is off)
_STIM_SHADE = "0.90"         # light band behind the EMG where stim is on


def _window_color(cls, stim_on, rest_idx):
    if cls == rest_idx:
        return _STIM_REST if stim_on else _CLEAN_REST
    return _MOVE_COLORS[cls % len(_MOVE_COLORS)]


def _strip(ax, t, cls, stim, rest_idx, step_t):
    """Draw a horizontal class-strip: contiguous same-colour runs as spans."""
    cols = [_window_color(int(c), bool(s), rest_idx) for c, s in zip(cls, stim)]
    i, n = 0, len(t)
    while i < n:
        j = i
        while j < n and cols[j] == cols[i]:
            j += 1
        right = (t[j] if j < n else t[-1] + step_t)
        ax.axvspan(t[i], right, color=cols[i], lw=0)
        i = j
    ax.set_yticks([]); ax.set_ylim(0, 1)


def recording_timeline(E, gt, pred, stim_win, class_names, fs=2048.0, rest_idx=3,
                       clean_ch=None, clean_on=None, ch_label="", title="", save=None,
                       rest_label="rest"):
    """Cleaned EMG channel + truth vs decoder class-strips over one recording.

    Top: the cleaned signal of one channel in time, with stim-on lightly shaded.
    Middle/bottom: truth and decoder class-strips (movement colours; grey = rest under
    stim, light gray = rest with stim off), matching the drive figures.
    stim_win : per-window bool (window overlaps stimulation)."""
    plt = _plt()
    from matplotlib.patches import Patch
    E = np.asarray(E)
    order = np.argsort(E)
    E, gt, pred, stim_win = E[order], np.asarray(gt)[order], np.asarray(pred)[order], np.asarray(stim_win)[order]
    t = E / fs
    step_t = float(np.median(np.diff(t))) if len(t) > 1 else 0.064

    fig, (aE, aT, aD) = plt.subplots(3, 1, figsize=(12, 4.6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.4, 0.7, 0.7]))
    # cleaned EMG channel (the actual signal, not RMS)
    if clean_ch is not None:
        y = np.asarray(clean_ch, float)
        ts = np.arange(len(y)) / fs
        if clean_on is not None:
            for s, e in _runs(ts, np.asarray(clean_on, bool)):
                aE.axvspan(s, e, color=_STIM_SHADE, lw=0)
        aE.plot(ts, y, color="0.15", lw=0.3)
        lim = float(np.nanpercentile(np.abs(y), 99.7)) or 1.0
        aE.set_ylim(-1.3 * lim, 1.3 * lim)
    aE.set_ylabel(f"cleaned EMG\n{ch_label}", fontsize=9)
    aE.set_title(title, fontsize=10)

    _strip(aT, t, gt, stim_win, rest_idx, step_t); aT.set_ylabel("truth", fontsize=9)
    _strip(aD, t, pred, stim_win, rest_idx, step_t); aD.set_ylabel("decoder", fontsize=9)
    aD.set_xlabel("time (s)")

    handles = [Patch(color=_MOVE_COLORS[c % len(_MOVE_COLORS)], label=class_names[c])
               for c in range(rest_idx)]
    handles += [Patch(color=_STIM_REST, label=f"{rest_label} (stim on)"),
                Patch(color=_CLEAN_REST, label=f"{rest_label} (stim off)")]
    aE.legend(handles=handles, fontsize=8, loc="upper right", ncol=len(handles))
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig


_GATE_MOVE = "#d62728"       # red   (gate says move)
_GATE_REST = "#7f7f7f"       # grey  (gate says rest)
GATE_PALETTE = {0: _GATE_MOVE, 1: _GATE_REST}
_STIM_BAND = "#e2ebf7"       # cool tint behind the rows where stim runs. Deliberately not
#   `_STIM_SHADE`: on a stages plot the neutral grey of clean rest sits next to it, and two
#   neutral greys read as one band.


def class_palette(rest_idx=3):
    """Default value -> colour map for a class strip. Move classes take the movement
    colours, `rest_idx` is stim-rest grey, and `rest_idx + 1` is clean-rest light gray,
    so a routed decoder can show which regime produced each rest."""
    pal = {c: _MOVE_COLORS[c % len(_MOVE_COLORS)] for c in range(rest_idx)}
    pal[rest_idx] = _STIM_REST
    pal[rest_idx + 1] = _CLEAN_REST
    return pal


def stages_timeline(E, rows, class_names, rest_idx=3, stim=None, fs=2048.0, step=131,
                    clean_ch=None, clean_on=None, ch_label="", extra_legend=None,
                    title="", save=None):
    """One class-strip row per head/stage, over the windows of a recording.

    E     : window end samples, one per window
    rows  : list of (name, values) or (name, values, palette). `values` holds one class
            per window and -1 where that row does not apply, so a head draws only where
            it ran. `palette` maps value -> colour; the default is `class_palette`.
            Use GATE_PALETTE for a move/rest gate row.
    stim  : per-window bool. Where True the rows get a cool band behind them.
    clean_ch : the cleaned signal of one channel, at the sample rate. Given, it becomes a
            trace above the rows, so the strips can be read against the EMG that produced
            them. `clean_on` is its sample-level stim mask, `ch_label` its axis label.
    extra_legend : more (label, colour) pairs to append to the legend.

    Values of `rest_idx + 1` mean "rest, decided with the stimulation off", which the
    default palette draws light gray against the grey of stim-rest."""
    plt = _plt()
    from matplotlib.patches import Patch
    E = np.asarray(E, int)
    order = np.argsort(E)
    E = E[order]
    t = E / fs
    w = step / fs * 1.05
    default = class_palette(rest_idx)
    rows = [(r[0], np.asarray(r[1])[order], r[2] if len(r) > 2 else default) for r in rows]
    stim = None if stim is None else np.asarray(stim, bool)[order]

    nr = len(rows)
    h = 0.5 * nr + 1.4
    if clean_ch is None:
        fig, ax = plt.subplots(figsize=(13.5, h))
        aE = None
    else:
        fig, (aE, ax) = plt.subplots(2, 1, figsize=(13.5, h + 2.0), sharex=True,
                                     gridspec_kw=dict(height_ratios=[2.0, 0.5 * nr]))
    top = aE if aE is not None else ax

    if aE is not None:                       # the cleaned signal behind the strips
        y = np.asarray(clean_ch, float)
        ts = np.arange(len(y)) / fs
        if clean_on is not None:
            for a, b in _runs(ts, np.asarray(clean_on, bool)):
                aE.axvspan(a, b, color=_STIM_BAND, lw=0, zorder=0)
        aE.plot(ts, y, color="0.15", lw=0.3, zorder=2)
        lim = float(np.nanpercentile(np.abs(y), 99.7)) or 1.0
        aE.set_ylim(-1.3 * lim, 1.3 * lim)
        aE.set_ylabel(f"cleaned EMG\n{ch_label}", fontsize=9)
        for sp in ("top", "right"):
            aE.spines[sp].set_visible(False)

    if stim is not None:
        for a, b in _runs(t, stim):
            ax.axvspan(a, b, color=_STIM_BAND, lw=0, zorder=0)
    for i, (name, vals, pal) in enumerate(rows):
        y0 = (nr - 1 - i) + 0.1
        for k, c in pal.items():
            m = vals == k
            if m.any():
                ax.bar(t[m], 0.8, bottom=y0, width=w, color=c, linewidth=0, zorder=2)

    ax.set_yticks([(nr - 1 - i) + 0.5 for i in range(nr)])
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(0, nr + 0.05); ax.set_xlim(t.min(), t.max())
    ax.set_xlabel("time (s)")
    top.set_title(title, loc="left", fontsize=10, pad=18)
    handles = [Patch(color=default[c], label=class_names[c]) for c in range(rest_idx)]
    handles += [Patch(color=_STIM_REST, label=f"{class_names[rest_idx]} (stim on)"),
                Patch(color=_CLEAN_REST, label=f"{class_names[rest_idx]} (stim off)")]
    handles += [Patch(color=c, label=l) for l, c in (extra_legend or [])]
    if stim is not None or clean_on is not None:
        handles += [Patch(color=_STIM_BAND, label="stim on")]
    top.legend(handles=handles, ncol=len(handles), loc="lower center",
               bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=8,
               columnspacing=1.1, handlelength=1.1)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=170, bbox_inches="tight")
    return fig


def _runs(t, on):
    """Contiguous (t_start, t_end) spans where the boolean `on` is True."""
    on = np.asarray(on)
    spans, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            spans.append((t[i], t[min(j, n - 1)])); i = j
        else:
            i += 1
    return spans
