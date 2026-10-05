"""Example driver: fit the decoder, run it on a recording, and report accuracy.

Fits the two heads from the stim-train feature cache, calibrates on the online run's
trigger, then evaluates two ways:
  * streamed  - the online run fed through StreamingDecoder frame by frame;
  * offline   - the same run extracted whole-array and decided in one pass.
Also checks feature parity (A_G3) between the offline extraction and the cache.

Paths default to the experlangen data on the USB; override with the MVDEC_* env vars.

    python -u run_offline.py
"""
import os
import sys
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # bend/src -> import mvdecoder

from mvdecoder import (EXPERLANGEN, TwoStageDecoder, StreamingDecoder,
                       TriggerStimSource, load_recording, load_feat_cache,
                       load_good_indices, extract_offline, decide_offline)
from mvdecoder import eval as mve
from mvdecoder.features import vectors_from_cache

BASE = os.environ.get("MVDEC_BASE", r"D:\PatientGUI Output\experlangen")
ANALYSIS = os.environ.get("MVDEC_ANALYSIS", os.path.join(BASE, "analysis", "artifact_removal"))
ONLINE = os.environ.get("MVDEC_ONLINE",
                        os.path.join(BASE, "mv_Default-Training-Sequence_online_20260714_172405.raw"))
STIMTRAIN = os.path.join(ANALYSIS, "feat_v2_stimtrain_parrm_good.npz")
TESTCACHE = os.path.join(ANALYSIS, "feat_v2_parrm_good.npz")
GOODMASK = os.path.join(ANALYSIS, "good_mask.npy")
MODEL = os.path.join(BASE, "movement_model.pkl")
CHUNK = int(os.environ.get("MVDEC_CHUNK", "4096"))


def main():
    cfg = EXPERLANGEN
    good = load_good_indices(GOODMASK)
    keep_pos, _ = cfg.subset_positions(good)
    print(f"good channels: {len(good)}   subset (decoder) channels: {len(keep_pos)}")

    with open(MODEL, "rb") as f:
        art = pickle.load(f)
    trig_ch = int(art["trig_ch"])
    print(f"trigger channel: {trig_ch}")

    # 1) fit from the stim-train cache
    tr = load_feat_cache(STIMTRAIN)
    dec = TwoStageDecoder(cfg, good)
    dec.fit_from_cache(tr["F"], tr["A"], tr["E"], tr["gt"], tr["S"])
    print(f"fit: gate on {int(tr['S'].astype(bool).sum())} stim windows, "
          f"gesture on {int((tr['S'].astype(bool) & (tr['gt'] < cfg.rest_idx)).sum())} move windows")

    # 2) load the online run + calibrate on its trigger
    data, trig, meta = load_recording(ONLINE, trig_ch=trig_ch)
    print(f"online run: {meta['n_channels']} ch x {meta['n_samples']} samp "
          f"({meta['n_samples'] / cfg.fs:.1f}s)")
    src = TriggerStimSource(trig_ch, cfg)
    thr, P = src.calibrate(trig)
    dec.set_calibration(thr, P)
    print(f"calibration: thr={thr:.1f}  P={P:.4f} ({cfg.fs / P:.2f} Hz)")

    # ground truth (from the test cache: window ends, labels, stim flags)
    te = load_feat_cache(TESTCACHE)
    E, gt, S = te["E"].astype(int), te["gt"].astype(int), te["S"].astype(bool)

    # 3) STREAMED evaluation
    stream = StreamingDecoder(dec, src)
    events = []
    for s in range(0, data.shape[1], CHUNK):
        events += stream.process(data[:, s:s + CHUNK])
    pos = {int(e): i for i, e in enumerate(E)}
    pred = np.full(len(E), -1, int)
    for ev in events:
        i = pos.get(int(ev["end"]))
        if i is not None:
            pred[i] = ev["final"]
    m = (pred >= 0) & S
    print(f"\n=== STREAMED (matched stim windows: {int(m.sum())}) ===")
    mve.report(gt[m], pred[m], cfg.class_names)

    # 4) OFFLINE evaluation (whole-array extraction at the cache window ends)
    src2 = TriggerStimSource(trig_ch, cfg)
    src2.calibrate(trig)
    ext = extract_offline(data, src2, cfg, good, ends=E)
    dr = decide_offline(dec, ext["Xgate"], ext["Xgest"], E, cfg)
    pred_off = np.array([d["final"] for d in dr["decisions"]])
    print(f"\n=== OFFLINE (stim windows: {int(S.sum())}) ===")
    mve.report(gt[S], pred_off[S], cfg.class_names)

    # 5) feature parity: offline A_G3 vs the cache's A_G3
    Xg_c, _ = vectors_from_cache(te["F"], te["A"], te["E"], good, cfg)
    d = np.abs(ext["Xgate"][:, 0] - Xg_c[:, 0])
    print(f"\nfeature parity A_G3 (offline vs cache): median|d| stim {np.median(d[S]):.2e}  "
          f"max {d[S].max():.2e}")


if __name__ == "__main__":
    main()
