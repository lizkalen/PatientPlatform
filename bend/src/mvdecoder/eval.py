"""Evaluation metrics for the decoder output.

Provides confusion matrices, per-class precision/recall/F1 and macro-F1 at three
framings used in this project: 4-class (fist/trp/ext/rest), release-3 (ext and rest
merged) and close-vs-release (grasp = classes 0-1 vs release = classes 2-3). Also an
event-latency helper: how long after a release the gate flips to rest.

numpy only.
"""
from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred, labels) -> np.ndarray:
    idx = {c: i for i, c in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), int)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        M[idx[int(t)], idx[int(p)]] += 1
    return M


def prf(y_true, y_pred, labels) -> dict:
    """Per-class (precision, recall, f1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c in labels:
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[c] = (p, r, f)
    return out


def macro_f1(y_true, y_pred, labels) -> float:
    res = prf(y_true, y_pred, labels)
    return float(np.mean([res[c][2] for c in labels]))


def _release3(y):
    y = np.asarray(y)
    return np.where(y >= 2, 2, y)


def _close_release(y):
    return (np.asarray(y) >= 2).astype(int)


def summary(y_true, y_pred, n_classes=4) -> dict:
    """Macro-F1 at the three framings."""
    labels4 = list(range(n_classes))
    return {
        "macroF1_4class": macro_f1(y_true, y_pred, labels4),
        "macroF1_release3": macro_f1(_release3(y_true), _release3(y_pred), [0, 1, 2]),
        "macroF1_close_vs_release": macro_f1(_close_release(y_true), _close_release(y_pred), [0, 1]),
    }


def report(y_true, y_pred, class_names=("fist", "trp", "ext", "rest")) -> dict:
    """Print the three macro-F1 numbers, the 4-class confusion and per-class P/R/F1.
    Returns the summary dict."""
    n = len(class_names)
    s = summary(y_true, y_pred, n)
    print(f"4-class macroF1        {s['macroF1_4class']*100:5.1f}%")
    print(f"release-3 macroF1      {s['macroF1_release3']*100:5.1f}%")
    print(f"close-vs-release F1    {s['macroF1_close_vs_release']*100:5.1f}%")
    print("\nconfusion (rows = truth, cols = pred):")
    M = confusion(y_true, y_pred, list(range(n)))
    head = "        " + " ".join(f"{c:>6s}" for c in class_names)
    print(head)
    for i, c in enumerate(class_names):
        print(f"{c:>7s} " + " ".join(f"{v:6d}" for v in M[i]))
    print("\nper-class precision / recall / f1:")
    pr = prf(y_true, y_pred, list(range(n)))
    for i, c in enumerate(class_names):
        p, r, f = pr[i]
        print(f"  {c:>6s}  P {p*100:5.1f}  R {r*100:5.1f}  F1 {f*100:5.1f}")
    return s


def stage_report(y_true, gate_is_rest, gest_pred, S, rest_idx=3) -> dict:
    """Per-stage numbers on stim windows: the gate (move vs rest) over all stim windows,
    and the gesture head over the true-move stim windows.
    gate_is_rest : bool per window (gate decided rest)
    gest_pred    : gesture argmax per window (0..rest_idx-1)
    """
    S = np.asarray(S, bool)
    gt = np.asarray(y_true)
    gate_true_rest = (gt == rest_idx)[S]
    gate_pred_rest = np.asarray(gate_is_rest, bool)[S]
    gate = macro_f1(gate_true_rest.astype(int), gate_pred_rest.astype(int), [0, 1])
    move = S & (gt < rest_idx)
    gest = macro_f1(gt[move], np.asarray(gest_pred)[move], list(range(rest_idx)))
    return {"gate_macroF1": gate, "gesture_macroF1": gest}


def event_latency(dec_E, dec_is_rest, E, S, gt, boutid, fs=2048.0, win=524,
                  hold=3) -> dict:
    """Time from each bout's release (start of its stim-rest tail) to the gate first
    flipping to rest and holding for `hold` decisions. Returns per-bout latencies (ms)
    and their median. Follows the event-latency measure in stream_e2e."""
    dec_E = np.asarray(dec_E, int)
    dec_is_rest = np.asarray(dec_is_rest, bool)
    S = np.asarray(S, bool)
    gt = np.asarray(gt, int)
    boutid = np.asarray(boutid, int)
    order = np.argsort(dec_E)
    dE, dR = dec_E[order], dec_is_rest[order]
    rows = []
    for b in sorted(set(boutid[boutid >= 0].tolist())):
        m = S & (gt == 3) & (boutid == b)
        if m.sum() < hold:
            continue
        t_rel = (E[m].min() - (win - 1) / 2.0) / fs
        cand = np.flatnonzero((dE / fs >= t_rel) & dR)
        flip = None
        for i in cand:
            if i + hold <= len(dR) and dR[i:i + hold].all():
                flip = i
                break
        if flip is not None:
            rows.append((b, (dE[flip] / fs - t_rel) * 1000.0))
    lat = np.array([r[1] for r in rows]) if rows else np.array([])
    return {"per_bout": rows, "median_ms": float(np.median(lat)) if lat.size else None,
            "n": len(rows)}
