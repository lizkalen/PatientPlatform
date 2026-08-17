"""Parity and invariance checks.

  1. StreamPARRM matches parrm_offline once its history is full (synthetic, no data).
  2. AmplitudeHistory matches the vectorized deltas_offline (synthetic).
  3. With the experlangen data present:
     - offline feature extraction matches the feat_v2 cache (A_G3);
     - the streamed decisions are invariant to how the stream is chunked.

Run:  python -u test_parity.py
Data-dependent checks are skipped with a message if the recording is not found.
"""
import os
import sys
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # bend/src

from mvdecoder import (EXPERLANGEN, TwoStageDecoder, StreamingDecoder,
                       TriggerStimSource, load_recording, load_feat_cache,
                       load_good_indices)
from mvdecoder.parrm import parrm_offline, StreamPARRM
from mvdecoder.features import AmplitudeHistory, deltas_offline

BASE = os.environ.get("MVDEC_BASE", r"D:\PatientGUI Output\experlangen")
ANALYSIS = os.environ.get("MVDEC_ANALYSIS", os.path.join(BASE, "analysis", "artifact_removal"))
ONLINE = os.path.join(BASE, "mv_Default-Training-Sequence_online_20260714_172405.raw")


def test_parrm_stream_matches_offline():
    rng = np.random.default_rng(0)
    C, N, P = 8, 4000, 67.4017
    x = rng.standard_normal((C, N))
    on = np.ones(N, bool)
    y_off = parrm_offline(x, P, on, m=24, skip=2)
    sp = StreamPARRM(C, P, m=24, skip=2)
    outs = []
    i = 0
    for step in (37, 128, 500, 91, 1000):          # varied chunk sizes
        while i < N and step:
            take = min(step, N - i)
            outs.append(sp.push(x[:, i:i + take]))
            i += take
            break
    # feed the rest in fixed chunks
    while i < N:
        take = min(256, N - i)
        outs.append(sp.push(x[:, i:i + take]))
        i += take
    y_str = np.concatenate(outs, axis=1)
    warm = sp.need
    d = np.abs(y_off[:, warm:] - y_str[:, warm:]).max()
    assert d < 1e-9, f"PARRM stream vs offline max|d|={d:.2e}"
    print(f"[PASS] PARRM stream==offline after warm-up (max|d|={d:.2e}, warm={warm})")


def test_amplitude_history_matches_offline():
    rng = np.random.default_rng(1)
    n = 300
    A = rng.standard_normal((n, 2))
    E = np.arange(n) * 131                          # uniform window ends, already sorted
    lags = (4, 8, 16)
    dA_off = deltas_offline(A, E, lags)
    ah = AmplitudeHistory(lags)
    dA_str = np.array([ah.push(A[i]) for i in range(n)])
    d = np.abs(dA_off - dA_str).max()
    assert d < 1e-12, f"dA stream vs offline max|d|={d:.2e}"
    print(f"[PASS] AmplitudeHistory==deltas_offline (max|d|={d:.2e})")


def _fit_decoder():
    good = load_good_indices(os.path.join(ANALYSIS, "good_mask.npy"))
    tr = load_feat_cache(os.path.join(ANALYSIS, "feat_v2_stimtrain_parrm_good.npz"))
    dec = TwoStageDecoder(EXPERLANGEN, good)
    dec.fit_from_cache(tr["F"], tr["A"], tr["E"], tr["gt"], tr["S"])
    return dec, good


def test_feature_parity_and_chunk_invariance():
    if not os.path.exists(ONLINE):
        print("[SKIP] experlangen online run not found; data-dependent checks skipped")
        return
    from mvdecoder import extract_offline
    from mvdecoder.features import vectors_from_cache
    cfg = EXPERLANGEN
    with open(os.path.join(BASE, "movement_model.pkl"), "rb") as f:
        trig_ch = int(pickle.load(f)["trig_ch"])
    dec, good = _fit_decoder()
    data, trig, _ = load_recording(ONLINE, trig_ch=trig_ch)
    src = TriggerStimSource(trig_ch, cfg)
    thr, P = src.calibrate(trig)
    dec.set_calibration(thr, P)

    # feature parity: offline A_G3 vs cache A_G3 on stim windows
    te = load_feat_cache(os.path.join(ANALYSIS, "feat_v2_parrm_good.npz"))
    E, S = te["E"].astype(int), te["S"].astype(bool)
    src2 = TriggerStimSource(trig_ch, cfg); src2.calibrate(trig)
    ext = extract_offline(data, src2, cfg, good, ends=E)
    Xg_c, _ = vectors_from_cache(te["F"], te["A"], te["E"], good, cfg)
    d = np.abs(ext["Xgate"][:, 0] - Xg_c[:, 0])
    med = float(np.median(d[S]))
    assert med < 1e-4, f"A_G3 parity median|d|={med:.2e}"
    print(f"[PASS] feature parity A_G3 offline-vs-cache median|d|={med:.2e} (stim)")

    # chunk-invariance on a subrange: one chunk vs fixed 128 vs bursty
    T = min(60 * int(cfg.fs), data.shape[1])
    sub = data[:, :T]

    def run(chunks):
        st = StreamingDecoder(dec, src); st.reset()
        evs = []
        for c in chunks:
            evs += st.process(c)
        return [(int(e["end"]), int(e["final"])) for e in evs]

    one = run([sub])
    fixed = run([sub[:, i:i + 128] for i in range(0, T, 128)])
    rng = np.random.default_rng(3)
    bursts, i = [], 0
    while i < T:
        k = int(rng.integers(1, 4)) * 128
        bursts.append(sub[:, i:i + k]); i += k
    burst = run(bursts)
    assert one == fixed == burst, "chunk-invariance failed"
    print(f"[PASS] chunk-invariance: {len(one)} decisions identical (1-chunk / 128 / bursty)")


def test_decider_hysteresis_and_latch():
    from mvdecoder.decider import Decider, START, STOP, HOLD
    cfg = EXPERLANGEN.with_(ema_alpha=1.0)          # no smoothing, so p_move == input
    d = Decider(cfg)
    move, rest = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    g = np.array([1.0, 0.0, 0.0])

    # rising through on_thr starts stim
    assert d.update(rest, g, 0.0)["stim_cmd"] == HOLD
    assert d.update(move, g, 100.0)["stim_cmd"] == START
    # a brief rest inside the latch window does NOT stop it
    assert d.update(rest, g, 200.0)["stim_cmd"] == HOLD
    # after the latch, a confirmed rest stops it
    assert d.update(rest, g, 100.0 + cfg.latch_ms + 1)["stim_cmd"] == STOP
    # hysteresis: a value in the [off, on) band neither starts nor stops
    d2 = Decider(cfg)
    assert d2.update(np.array([0.45, 0.55]), g, 0.0)["stim_cmd"] == HOLD
    print("[PASS] decider hysteresis + onset latch")


if __name__ == "__main__":
    test_parrm_stream_matches_offline()
    test_amplitude_history_matches_offline()
    test_decider_hysteresis_and_latch()
    test_feature_parity_and_chunk_invariance()
    print("\nall parity checks done")
