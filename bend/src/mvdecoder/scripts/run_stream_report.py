"""Accuracy and figures for a routed streaming run: reads `_run_stream.npz` from
run_stream.py, matches it to the ground truth of the online feature cache by window end,
and reports each head on the windows it actually ran plus the routed system as a whole.

Writes to figs_run_stream/ beside the analysis caches:
    fig_stages.png      truth, each head where it ran, and the final routed decision
    fig_confusion.png   CLEAN head / gate / gesture / routed 4-class / close-vs-release

Paths follow the same MVDEC_* env vars as run_stream.py.

    python -u run_stream_report.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # bend/src

from mvdecoder import EXPERLANGEN, load_feat_cache
from mvdecoder import eval as mve
from mvdecoder import viz

BASE = os.environ.get("MVDEC_BASE", r"E:\PatientGUI Output\experlangen")
ANALYSIS = os.environ.get("MVDEC_ANALYSIS", os.path.join(BASE, "analysis", "artifact_removal"))
RECORD = os.environ.get("MVDEC_RECORD", os.path.join(ANALYSIS, "_run_stream.npz"))
TESTCACHE = os.path.join(ANALYSIS, "feat_v2_parrm_good.npz")
FIGS = os.environ.get("MVDEC_FIGS", os.path.join(ANALYSIS, "figs_run_stream"))


def head_report(title, y, p, names):
    """Print one head's confusion and per-class recall. Returns (M, acc, macroF1)."""
    labels = list(range(len(names)))
    M = mve.confusion(y, p, labels)
    pr = mve.prf(y, p, labels)
    acc = float((np.asarray(y) == np.asarray(p)).mean() * 100)
    f1 = float(np.mean([pr[c][2] for c in labels]) * 100)
    print(f"\n{title}   n={len(y)}  acc {acc:.1f}%  macroF1 {f1:.1f}%")
    print("        " + "".join(f"{n:>7s}" for n in names) + "  | recall")
    for i, n in enumerate(names):
        print(f"  {n:>5s} " + "".join(f"{v:7d}" for v in M[i]) + f"  | {pr[i][1]*100:5.1f}%")
    return M, acc, f1


def main():
    cfg = EXPERLANGEN
    os.makedirs(FIGS, exist_ok=True)
    d = np.load(RECORD)
    E, stim, on_clean = d["E"], d["stim"], d["on_clean"]
    gate, gest, clean, final = d["gate"], d["gest"], d["clean"], d["final"]
    fs, step = float(d["fs"]), int(d["step"])
    rest, names = cfg.rest_idx, list(cfg.class_names)

    # ground truth: the offline cache of the same recording, matched by window end
    te = load_feat_cache(TESTCACHE)
    pos = {int(x): i for i, x in enumerate(te["E"].astype(int))}
    idx = np.array([pos.get(int(x), -1) for x in E])
    ok = idx >= 0
    gt = np.full(len(E), -1, int)
    gt[ok] = te["gt"].astype(int)[idx[ok]]
    S_off = np.zeros(len(E), bool)
    S_off[ok] = te["S"].astype(bool)[idx[ok]]
    agree = float((stim[ok] == S_off[ok]).mean() * 100)
    print(f"matched {int(ok.sum())}/{len(E)} streamed windows to truth | "
          f"streamed stim {int(stim.sum())} clean {int((~stim).sum())} | "
          f"causal routing agrees with the offline stim flag {agree:.2f}%")

    mc = ok & on_clean                              # windows the clean head ran on
    mg = ok & ~on_clean                             # windows the stim heads ran on
    mgg = mg & (gt < rest) & (gt >= 0)              # true-movement stim windows
    print("\n" + "=" * 78 + "\nPER HEAD (each on the windows it ran)\n" + "=" * 78)
    panels = []
    if mc.any():
        M, a, f = head_report("[CLEAN head]   all classes, stim-OFF windows",
                              gt[mc], clean[mc], names)
        panels.append((f"CLEAN head\n(stim-OFF)\nacc {a:.1f}%  mF1 {f:.1f}%", M, names))
    M, a, f = head_report("[STIM gate]    move/rest, stim-ON windows",
                          (gt[mg] == rest).astype(int), gate[mg], ["move", "rest"])
    panels.append((f"STIM gate\n(stim-ON)\nacc {a:.1f}%  mF1 {f:.1f}%", M, ["move", "rest"]))
    M, a, f = head_report("[STIM gesture] move classes, stim-ON true-movement windows",
                          gt[mgg], gest[mgg], names[:rest])
    panels.append((f"STIM gesture\n(stim-ON move)\nacc {a:.1f}%  mF1 {f:.1f}%", M, names[:rest]))

    print("\n" + "=" * 78 + "\nROUTED SYSTEM (both heads, every matched window)\n" + "=" * 78)
    to2 = lambda v: (np.asarray(v) >= rest - 1).astype(int)     # close vs release
    M, a, f = head_report("[ROUTED] all classes", gt[ok], final[ok], names)
    panels.append((f"ROUTED\n(all windows)\nacc {a:.1f}%  mF1 {f:.1f}%", M, names))
    M, a, f = head_report("[ROUTED] close vs release", to2(gt[ok]), to2(final[ok]),
                          ["close", "release"])
    panels.append((f"ROUTED close-vs-release\nacc {a:.1f}%  mF1 {f:.1f}%", M,
                   ["close", "release"]))
    print("\n[ROUTED] release-3 macroF1 "
          f"{mve.summary(gt[ok], final[ok], cfg.n_classes)['macroF1_release3']*100:.1f}%")

    # ---- figures ----------------------------------------------------------------
    plt = viz._plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.4))
    for ax, (t, M, labs) in zip(np.atleast_1d(axes), panels):
        viz.confusion_heatmap(M, labs, title=t, ax=ax)
    fig.suptitle("Routed streamed decoder, causal, cross-recording", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIGS, "fig_confusion.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[fig] {os.path.join(FIGS, 'fig_confusion.png')}")

    # clean-regime rest draws light gray, stim-regime rest grey (class rest + 1)
    pale = lambda v, m: np.where(m & (v == rest), rest + 1, v)
    hide = lambda v, m: np.where(m, v, -1)
    rows = [("truth", pale(np.where(ok, gt, -1), ok & ~stim)),
            ("CLEAN head", pale(hide(clean, mc), mc)),
            ("gate", np.where(mg, gate, -1), viz.GATE_PALETTE),
            ("gesture", hide(gest, mg)),
            ("final", pale(np.where(ok, final, -1), ok & on_clean))]
    viz.stages_timeline(
        E, rows, names, rest_idx=rest, stim=stim, fs=fs, step=step,
        clean_ch=d["clean_ch"] if "clean_ch" in d else None,
        clean_on=d["clean_on"] if "clean_on" in d else None,
        ch_label=f"ch{int(d['clean_ch_idx'])}" if "clean_ch_idx" in d else "",
        extra_legend=[("gate: move", viz.GATE_PALETTE[0])],
        title="Routed streamed decoder — each head drawn only where it ran",
        save=os.path.join(FIGS, "fig_stages.png"))
    plt.close("all")
    print(f"[fig] {os.path.join(FIGS, 'fig_stages.png')}")


if __name__ == "__main__":
    main()
