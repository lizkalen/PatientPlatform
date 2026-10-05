"""Real-time streaming trial: feed the online run frame by frame against a wall clock
and report per-frame compute time and decision counts.

Runs the full routed decoder: the stim gate + gesture heads on the stimulated windows
and the clean (no-stim) head on the rest, picked per window by the causal stim mask.
The clean head comes from the no-stim training cache; if that cache is absent the run
falls back to the stim heads alone and says so.

Saves the per-window record to `_run_stream.npz` next to the analysis caches. Accuracy
and figures come from run_stream_report.py, which reads that file. Reuses the fit +
calibration from run_offline. Paths follow the same MVDEC_* env vars.

    python -u run_stream.py
    python -u run_stream_report.py
"""
import os
import sys
import time
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # bend/src

from mvdecoder import (EXPERLANGEN, TwoStageDecoder, StreamingDecoder,
                       TriggerStimSource, load_recording, load_feat_cache,
                       load_good_indices)

BASE = os.environ.get("MVDEC_BASE", r"E:\PatientGUI Output\experlangen")
ANALYSIS = os.environ.get("MVDEC_ANALYSIS", os.path.join(BASE, "analysis", "artifact_removal"))
ONLINE = os.environ.get("MVDEC_ONLINE",
                        os.path.join(BASE, "mv_Default-Training-Sequence_online_20260714_172405.raw"))
STIMTRAIN = os.path.join(ANALYSIS, "feat_v2_stimtrain_parrm_good.npz")
NOSTIMTRAIN = os.environ.get("MVDEC_NOSTIM",
                             os.path.join(ANALYSIS, "feat_v2_nostimtrain_parrm_good.npz"))
RECORD = os.environ.get("MVDEC_RECORD", os.path.join(ANALYSIS, "_run_stream.npz"))
FRAME = int(os.environ.get("MVDEC_FRAME", "128"))          # device frame size (62.5 ms)
REALTIME = os.environ.get("MVDEC_REALTIME", "0") == "1"     # sleep to a wall clock?
SAVECLEAN = os.environ.get("MVDEC_SAVECLEAN", "1") == "1"   # keep one cleaned channel for
#   the report's EMG trace. It copies each cleaned frame inside the timed section, which
#   costs a few microseconds per frame; set to 0 for timing with nothing else in the loop.


def main():
    cfg = EXPERLANGEN
    good = load_good_indices(os.path.join(ANALYSIS, "good_mask.npy"))
    with open(os.path.join(BASE, "movement_model.pkl"), "rb") as f:
        trig_ch = int(pickle.load(f)["trig_ch"])
    tr = load_feat_cache(STIMTRAIN)
    dec = TwoStageDecoder(cfg, good).fit_from_cache(tr["F"], tr["A"], tr["E"], tr["gt"], tr["S"])

    # clean head: no-stim training block, same gesture features, all classes
    if os.path.exists(NOSTIMTRAIN):
        nt = load_feat_cache(NOSTIMTRAIN)
        dec.fit_clean_from_cache(nt["F"], nt["A"], nt["E"], nt["gt"], nt["S"])
        gt_c, S_c = nt["gt"].astype(int), nt["S"].astype(bool)
        m = (S_c & (gt_c < cfg.rest_idx)) | (~S_c & (gt_c == cfg.rest_idx))
        print(f"clean head: {int(m.sum())} no-stim windows "
              f"{np.bincount(gt_c[m], minlength=cfg.n_classes).tolist()} {cfg.class_names}")
    else:
        print(f"no clean cache at {NOSTIMTRAIN} -> stim heads only, no routing")

    data, trig, meta = load_recording(ONLINE, trig_ch=trig_ch)
    src = TriggerStimSource(trig_ch, cfg)
    thr, P = src.calibrate(trig)
    dec.set_calibration(thr, P)

    stream = StreamingDecoder(dec, src, keep_clean=SAVECLEAN)
    budget_ms = FRAME / cfg.fs * 1000.0
    frame_ms, rec = [], []
    N = data.shape[1]
    t0 = time.perf_counter()
    for k, s in enumerate(range(0, N, FRAME)):
        if REALTIME:
            due = t0 + (k + 1) * FRAME / cfg.fs
            while time.perf_counter() < due:
                pass
        a = time.perf_counter()
        evs = stream.process(data[:, s:s + FRAME])
        frame_ms.append((time.perf_counter() - a) * 1000.0)
        for ev in evs:
            # each head's own smoothed call, -1 on the windows where it did not run
            clean = int(ev["clean_ema"].argmax()) if ev["head"] == "clean" else -1
            gate = int(ev["gate_ema"].argmax()) if ev["head"] == "stim" else -1
            gest = int(ev["gest_ema"].argmax()) if ev["head"] == "stim" else -1
            rec.append((ev["end"], ev["stim"], ev["head"] == "clean", gate, gest, clean,
                        ev["final"], ev["p_move"]))
    frame_ms = np.array(frame_ms)
    rec = np.array(rec, float)
    over = int((frame_ms > budget_ms).sum())
    print(f"frames: {len(frame_ms)} of {FRAME} samples (budget {budget_ms:.1f} ms)")
    print(f"compute/frame: mean {frame_ms.mean():.2f}  p95 {np.percentile(frame_ms, 95):.2f}  "
          f"max {frame_ms.max():.2f} ms   over budget: {over}")
    print(f"decisions emitted: {len(rec)}")
    on_clean = rec[:, 2].astype(bool)
    if dec.has_clean:
        print(f"routed: stim heads {int((~on_clean).sum())} windows  |  "
              f"clean head {int(on_clean.sum())} windows")

    out = dict(E=rec[:, 0].astype(int), stim=rec[:, 1].astype(bool), on_clean=on_clean,
               gate=rec[:, 3].astype(int), gest=rec[:, 4].astype(int),
               clean=rec[:, 5].astype(int), final=rec[:, 6].astype(int),
               p_move=rec[:, 7], frame_ms=frame_ms, fs=np.float64(cfg.fs),
               step=np.int64(cfg.step), has_clean=np.bool_(dec.has_clean))
    if SAVECLEAN:                       # loudest cleaned channel, for the report's trace
        sig, on, keep = stream.clean_signal()
        ci = int(np.argmax(np.sqrt((sig ** 2).mean(1))))
        out.update(clean_ch=sig[ci], clean_on=on, clean_ch_idx=np.int64(keep[ci]))
        print(f"cleaned trace: ch{int(keep[ci])} (loudest of {len(keep)} subset channels)")
    np.savez_compressed(RECORD, **out)
    print(f"record -> {RECORD}")


if __name__ == "__main__":
    main()
