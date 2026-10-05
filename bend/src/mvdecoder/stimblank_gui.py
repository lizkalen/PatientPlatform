"""Interactive GUI to select the 30 Hz stim-pulse times for blanking, plus its pulse
helpers. Reconstructed as we last had it.

Blanking, NOT template subtraction: on build each pulse's [pre, post] window is REPLACED
by a linear interpolation across it (_blank_regions) — the artifact samples are discarded,
never modelled.

External names this module expects (wire these to wherever they now live):
  constants : FS, STIM_HZ, OUT, DATASET, RECS
  colours   : COL (movement -> colour), used only for bout tinting is optional here
  io/dsp    : load_emg(name)->(ch,n), notch_only(x), bandpass(x), envelope(x),
              channel_rms(x), _blank_regions(sig, times, half, N, fill)
  channels  : load_good_mask()->(good_bool, src),  _which_grid(ch)->str
  segment   : segment_recording(key, cond, fname, xg)->(bouts, src)
  gui       : _use_gui()->backend_name,  log(*a)
"""
from __future__ import annotations

import os

import numpy as np
from scipy.signal import find_peaks, sosfilt

from .config import EXP13072026 as CFG
from .dsp import design_filters
from .io import load_recording

# ── session wiring ───────────────────────────────────────────────────────────────
# The GUI + pulse helpers below read these module-level names. Montage-derived values
# (FS, N_EMG, the filters, the grid map) come from the mvdecoder config; the session
# paths + tuning are pushed in by the driver via `configure()` (see
# processing/scripts/movement_disc_1307/{0-mark_stim_pulses.py, stimblank_config.py}).
# These helpers were lost in the refactor and reconstructed here; if equivalents
# resurface elsewhere in the package, import them instead.
FS = float(CFG.fs)                 # 2048 Hz
N_EMG = CFG.n_emg                  # 384 EMG channels (grids only)
STIM_HZ = 30.0                     # nominal; overridden by config (measured ~30.57)
_BP_SOS, _NOTCH_SOS = design_filters(CFG)

STIM_DIR = None                    # recordings dir (…/cache/preerlplots_exp13072026/stim)
OUT_BASE = None                    # out root (…/cache/out); OUT is per-family
OUT = None                         # current family's out dir (set by run_family)
RECS = []                          # [(key, cond, fname, movement), …] for the family
DATASET = ""
FAMILIES = {}

BLANK_MS = 15.0                    # blank window per pulse (preview overlay)
WIN_MS = 100.0                     # zoom-window width
SNAP = 6                           # click-snap radius (samples) to the local peak
ENV_SMOOTH_MS = 50.0               # stim-on envelope smoothing
ENV_K = 5.0                        # stim-on threshold = median + ENV_K * MAD
ENV_MERGE_S = 0.3                  # bridge stim-on gaps shorter than this
ENV_MIN_S = 0.3                    # drop stim-on regions shorter than this


def configure(**kw):
    """Push session constants from the driver's config file into this module's namespace
    before running (FS, STIM_HZ, STIM_DIR, OUT_BASE, FAMILIES, BLANK_MS, ENV_*, …)."""
    g = globals()
    for k, v in kw.items():
        if k not in g:
            raise KeyError(f"unknown stimblank config key: {k!r}")
        g[k] = v


def log(*a):
    print(*a, flush=True)


def _use_gui():
    """Switch matplotlib to an interactive backend for the marking window."""
    import matplotlib
    for bk in ("QtAgg", "Qt5Agg", "TkAgg"):
        try:
            matplotlib.use(bk, force=True)
            return bk
        except Exception:
            continue
    return matplotlib.get_backend()


def load_emg(fname):
    """The N_EMG grid channels of a recording (channels x samples)."""
    data, _, _ = load_recording(os.path.join(STIM_DIR, f"{fname}_raw.pkl"))
    return np.asarray(data[:N_EMG], float)


def notch_only(x):
    """Notch at 50 Hz + harmonics (the montage config's notch cascade)."""
    return sosfilt(_NOTCH_SOS, np.asarray(x, float), axis=1)


def bandpass(x):
    """Band-pass (the montage config's bp_lo–bp_hi)."""
    return sosfilt(_BP_SOS, np.asarray(x, float), axis=1)


def channel_rms(x):
    return np.sqrt((np.asarray(x, float) ** 2).mean(axis=1))


def envelope(x, smooth_ms=None):
    """Across-channel mean |x|, moving-average smoothed — the activity/stim envelope."""
    x = np.asarray(x, float)
    r = np.mean(np.abs(x), axis=0) if x.ndim == 2 else np.abs(x)
    w = max(1, int((ENV_SMOOTH_MS if smooth_ms is None else smooth_ms) / 1000.0 * FS))
    return np.convolve(r, np.ones(w) / w, mode="same")


def _which_grid(ch):
    for name, a, z in CFG.grids:
        if a <= ch < z:
            return name
    return "?"


def load_good_mask():
    """Good-channel bool mask (length N_EMG) for the current family, or all-True if none
    exists yet — this GUI runs BEFORE good_mask is created, so it must not require it."""
    p = os.path.join(OUT, "good_mask.npy") if OUT else None
    if p and os.path.exists(p):
        m = np.load(p).astype(bool)
        m = (m[:N_EMG] if m.shape[0] >= N_EMG
             else np.pad(m, (0, N_EMG - m.shape[0]), constant_values=True))
        return m, "good_mask.npy"
    return np.ones(N_EMG, bool), "all channels (no good_mask yet)"


def _runs(on):
    d = np.diff(np.r_[0, np.asarray(on, bool).view(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def segment_recording(key, cond, fname, xg):
    """Stim-on regions for a triggerless run: where the across-channel |signal| envelope
    stays above the resting floor (median + ENV_K*MAD), gaps bridged / short runs dropped.
    `xg` is the band-passed EMG. Returns (regions, src); whole recording if nothing crosses."""
    env = envelope(xg)
    base = float(np.median(env))
    mad = float(np.median(np.abs(env - base))) + 1e-9
    gap, minlen = int(ENV_MERGE_S * FS), int(ENV_MIN_S * FS)
    merged = []
    for s, e in _runs(env > base + ENV_K * mad):
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    regs = [(s, e) for s, e in merged if e - s >= minlen]
    if not regs:
        return [(0, xg.shape[1])], "whole recording (no stim-on detected)"
    return regs, "stim-envelope"


def _blank_regions(sig, times, half, N, fill="linear"):
    """Artifact component removed by linear-interpolation blanking: within each pulse's
    [t-half, t-half+N) window, the samples minus the straight line across it (0 elsewhere),
    so `sig - _blank_regions(...)` is the blanked signal. GUI preview only."""
    sig = np.asarray(sig, float)
    _, n = sig.shape
    comp = np.zeros_like(sig)
    N = int(N)
    for t in np.atleast_1d(np.asarray(times, int)):
        a = max(0, int(t) - int(half))
        b = min(n, a + N)
        if b - a < 2:
            continue
        ramp = np.linspace(0.0, 1.0, b - a)[None, :]
        line = sig[:, a][:, None] + (sig[:, b - 1] - sig[:, a])[:, None] * ramp
        comp[:, a:b] = sig[:, a:b] - line
    return comp


def run_family(family, movements=None):
    """Set OUT/RECS/DATASET for a family and open the marking GUI for its recordings."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {list(FAMILIES)}")
    out_dir, recs = FAMILIES[family]
    global OUT, RECS, DATASET
    OUT = os.path.join(OUT_BASE, out_dir)
    os.makedirs(OUT, exist_ok=True)
    movs = [m for m in (movements or list(recs)) if m in recs]
    RECS = [(f"stim_{m}", "stim", recs[m], m) for m in movs]
    DATASET = f"exp13072026/{family}"
    stage_stimblank(blank_ms=BLANK_MS, win_ms=WIN_MS, snap=SNAP)


# ── pulse-time helpers ─────────────────────────────────────────────────────────────
def stim_pulse_path(key):
    return os.path.join(OUT, f"stimpulses_{key}.npy")


def load_stim_pulses(key):
    """Saved stim pulse-centre samples for a recording, or None if none marked."""
    p = stim_pulse_path(key) if key else None
    return np.load(p) if (p and os.path.exists(p)) else None


def _blank_pre_post(ms):
    """(pre, post) blank half-windows in samples for a total window of `ms`, split with
    the same ~4ms-before / ~11ms-after ratio as the tuned 15 ms default (pre=8/post=23)."""
    span = ms / 1000.0 * FS
    return int(round(span * 8 / 31)), int(round(span * 23 / 31))


def stim_ref(xn):
    """Pulse-detection signal: mean |notched EMG| across channels. The stim artifact
    hits many electrodes at once, so this spikes sharply at every pulse."""
    return np.mean(np.abs(xn.astype(np.float64)), axis=0)


def snap_to_peaks(centers, ref, snap=6):
    """Nudge each centre to the largest sample of `ref` within ±snap. Pass the SIGNED
    viewed channel so this snaps to the upward peak you see (not the larger |lobe|)."""
    n = len(ref); out = []
    for c in np.atleast_1d(np.asarray(centers, int)):
        lo, hi = max(0, int(c) - snap), min(n, int(c) + snap + 1)
        if hi > lo:
            out.append(lo + int(np.argmax(ref[lo:hi])))
    return np.unique(out)


def pulse_grid(regions, ref, anchors=None, fs=None, stim_hz=None, snap=6):
    """Full 30 Hz pulse grid inside each (start, end) region, phase-locked to any anchors
    that fall in the region (least-squares fit if ≥2, single-anchor phase if 1, else the
    region's strongest peak), then snapped to the local |artifact| peak. Kept for
    reference; the GUI's Auto-fill uses detect_pulses (peak-following, rate-robust)."""
    fs = fs or FS; stim_hz = stim_hz or STIM_HZ
    period = fs / stim_hz
    anchors = np.asarray(anchors if anchors is not None else [], int)
    centers = []
    for rs, re in regions:
        rs, re = int(rs), int(re)
        a = np.sort(anchors[(anchors >= rs) & (anchors <= re)])
        if len(a) >= 2:                              # fit phase + exact period
            k = np.round((a - a[0]) / period)
            per, base = np.polyfit(k, a.astype(float), 1)
        elif len(a) == 1:
            per, base = period, float(a[0])
        elif re > rs:
            per, base = period, float(rs + int(np.argmax(ref[rs:re])))
        else:
            continue
        j0, j1 = int(np.floor((rs - base) / per)), int(np.ceil((re - base) / per))
        centers += [int(round(base + j * per)) for j in range(j0, j1 + 1)]
    centers = sorted({c for c in centers if 0 <= c < len(ref)
                      and any(rs <= c <= re for rs, re in regions)})
    return snap_to_peaks(np.array(centers, int), ref, snap) if centers else np.array([], int)


def detect_pulses(regions, ref, anchors=None, fs=None, stim_hz=None):
    """Locate stim artifacts as the prominent PEAKS (local maxima) of the 1-D signal
    `ref` inside each region — pass the VIEWED channel (signed), so 'peak' means the
    upward spike you actually see, not the larger lobe of |signal|. Follows the real
    peaks, so it's robust to the rate drifting off exactly stim_hz (unlike a rigid grid).
    Your manually-placed `anchors` calibrate the detector and are always kept: their
    median spacing sets the minimum inter-pulse distance and their median height sets the
    amplitude/prominence threshold. With no anchors it falls back to a robust median+MAD
    threshold and the nominal period for spacing."""
    fs = fs or FS; stim_hz = stim_hz or STIM_HZ
    period = fs / stim_hz
    anchors = np.asarray(sorted({int(a) for a in (anchors if anchors is not None else [])}), int)
    if len(anchors) >= 3:                            # spacing from your marks (rate-agnostic)
        g = np.diff(anchors); g = g[(g > 0.3 * period) & (g < 3 * period)]
        base_gap = float(np.median(g)) if len(g) else period
    else:
        base_gap = period
    min_dist = max(1, int(0.55 * base_gap))
    if len(anchors):                                 # amplitude gate from your marks
        h = float(np.median(ref[anchors])); height, prom = 0.5 * h, 0.4 * h
    else:
        base = float(np.median(ref)); mad = float(np.median(np.abs(ref - base))) + 1e-9
        height, prom = base + 4 * mad, 3 * mad
    found = []
    for rs, re in regions:
        rs, re = int(rs), int(re)
        if re - rs < 3:
            continue
        pk, _ = find_peaks(ref[rs:re], distance=min_dist, height=height, prominence=prom)
        found += [rs + int(p) for p in pk]
    # union with your anchors, thinning near-duplicates within half the min distance
    # (a collision keeps your anchor over a detected peak)
    anchset = set(int(a) for a in anchors); kept = []
    for p in sorted(anchset | set(found)):
        if kept and p - kept[-1] < min_dist * 0.5:
            if p in anchset and kept[-1] not in anchset:
                kept[-1] = p                         # prefer the manual anchor
            continue
        kept.append(p)
    return np.array(kept, int)


# ── the GUI ─────────────────────────────────────────────────────────────────────────
def stage_stimblank(keys=None, snap=6, blank_ms=15.0, win_ms=100.0):
    """GUI (one window per stim run) to place the 30 Hz stim-pulse times for blanking.

    The view is a SINGLE channel shown `win_ms` (default 100 ms) at a time and autoscaled,
    so the stim spikes are large and clickable. Pulses are auto-DETECTED as prominent
    peaks inside the bouts (detect_pulses — follows the real peaks, robust to rate drift);
    page through to verify/correct:
      • ←/→ = page the window   ·  ↑/↓ = change channel   ·  n/p = jump next/prev bout
      • +/- = widen/narrow the window   ·  click the overview (top) to jump there
      • left-click (zoom) = add a pulse (snapped to the channel's local peak)
      • right-click (zoom) = remove the nearest pulse
      • Auto-fill = detect ALL peaks, calibrated by your manual marks (spacing + height);
        so mark a few real artifacts first, then Auto-fill   ·  Clear = drop all
      • b / Blank-preview = overlay the blanked channel (green) so you see the spike go
    Each pulse's blank window is shaded (red) so you see exactly what will be cut; the
    last-touched pulse is green. Save -> stimpulses_<key>.npy; Skip leaves it un-blanked."""
    bk = _use_gui()
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button
    for km in ("keymap.back", "keymap.forward", "keymap.home", "keymap.pan", "keymap.zoom",
               "keymap.save", "keymap.grid", "keymap.yscale", "keymap.xscale",
               "keymap.fullscreen"):
        plt.rcParams[km] = []                       # free ←/→/↑/↓/n/p/b/+/-
    log(f"stim-blank GUI (backend={bk}, blank window {blank_ms:.1f} ms)")
    good, _ = load_good_mask(); gidx = np.where(good)[0]
    pre, post = _blank_pre_post(blank_ms)           # what the build/xstim blanking will cut

    todo = [r for r in RECS if r[1] == "stim" and (not keys or r[0] in keys)]
    if not todo:
        log("stim-blank: no stim runs selected"); return
    for key, cond, fname, mov in todo:
        xn = notch_only(load_emg(fname)[good]); n = xn.shape[1]
        t = np.arange(n) / FS
        decO = max(1, n // 4000)                     # overview decimation
        bouts, _ = segment_recording(key, cond, fname, bandpass(xn))
        bouts = [tuple(b) for b in bouts]
        ch0 = int(np.argmax(channel_rms(xn)))        # detect on the viewed channel (peaks = maxima)
        saved = load_stim_pulses(key)
        pulses = set(int(c) for c in (saved if saved is not None
                                      else detect_pulses(bouts, xn[ch0])))
        v = {"ch": ch0, "win": win_ms / 1000.0, "blank": False,
             "center": (bouts[0][0] + bouts[0][1]) // 2 if bouts else n // 2,
             "bout": 0, "last": -1}

        fig, (axO, axZ) = plt.subplots(2, 1, figsize=(15, 7),
                                       gridspec_kw=dict(height_ratios=[1, 2.4]))
        plt.subplots_adjust(bottom=0.13, hspace=0.34)

        def centers():
            return np.array(sorted(pulses), int)

        def in_rate():
            dur = sum(e - s for s, e in bouts) / FS
            return len(pulses) / dur if dur > 1e-9 else 0.0

        def zbounds():
            half = max(1, int(v["win"] / 2 * FS))
            c = int(np.clip(v["center"], half, max(half, n - half)))
            return max(0, c - half), min(n, c + half)

        def redraw():
            ch = v["ch"]; sig = xn[ch]; z0, z1 = zbounds(); cc = centers()
            # ── overview (whole run, decimated) ──
            axO.clear()
            axO.plot(t[::decO], sig[::decO], color="0.6", lw=0.4)
            for s, e in bouts:
                axO.axvspan(s / FS, e / FS, color="tab:blue", alpha=0.12)
            axO.axvspan(z0 / FS, z1 / FS, color="tab:orange", alpha=0.40)
            axO.set_xlim(0, n / FS); axO.set_yticks([])
            axO.set_title(f"[{DATASET}] {key} · {mov} · ch {gidx[ch]} ({_which_grid(gidx[ch])}) "
                          f"— {len(pulses)} pulses (~{in_rate():.0f}/s in-bout, expect {STIM_HZ}) "
                          f"· click to jump · orange = current {v['win']*1000:.0f} ms window",
                          fontsize=9)
            # ── zoom (single channel, win_ms, autoscaled) ──
            axZ.clear()
            seg = sig[z0:z1].astype(np.float64)
            for s, e in bouts:                      # faint bout tint
                if e > z0 and s < z1:
                    axZ.axvspan(max(s, z0) / FS, min(e, z1) / FS, color="tab:blue", alpha=0.05)
            ccz = cc[(cc >= z0) & (cc < z1)]
            for c in ccz:                           # blank window + centre marker (selection feedback)
                hot = (c == v["last"])
                axZ.axvspan((c - pre) / FS, (c + post) / FS,
                            color="tab:green" if hot else "tab:red", alpha=0.22)
                axZ.axvline(c / FS, color="tab:green" if hot else "tab:red", lw=1.0)
                axZ.plot(c / FS, sig[c], marker="v", color="k", ms=6, zorder=5)
            axZ.plot(t[z0:z1], seg, color="tab:purple", lw=0.9, zorder=4)
            if v["blank"] and len(ccz):             # blanked channel overlay (green)
                z0p, z1p = max(0, z0 - 64), min(n, z1 + 64)
                comp = _blank_regions(xn[:, z0p:z1p], cc[(cc >= z0p) & (cc < z1p)] - z0p,
                                      half=pre, N=pre + post, fill="linear")
                axZ.plot(t[z0p:z1p], xn[ch, z0p:z1p] - comp[ch], color="tab:green", lw=1.1, zorder=6)
            m0, m1 = float(seg.min()), float(seg.max()); pad = 0.12 * (m1 - m0 + 1e-9)
            axZ.set_xlim(z0 / FS, z1 / FS); axZ.set_ylim(m0 - pad, m1 + pad)
            axZ.set_xlabel("Time [s]"); axZ.set_ylabel(f"ch {gidx[ch]} (mV, notched)")
            axZ.set_title(f"{z0/FS:.3f}–{z1/FS:.3f}s · {len(ccz)} pulses in view"
                          f"{' · BLANK preview (green)' if v['blank'] else ''}\n"
                          "←/→ page · ↑/↓ channel · n/p bout · +/- width · "
                          "L-click add · R-click remove · red = blank window", fontsize=9)
            fig.canvas.draw_idle()

        def on_click(event):
            if event.xdata is None:
                return
            if getattr(fig.canvas, "toolbar", None) and fig.canvas.toolbar.mode:
                return                              # a zoom/pan tool is active — ignore clicks
            c = int(event.xdata * FS)
            if event.inaxes is axO:
                v["center"] = c; redraw(); return
            if event.inaxes is axZ:
                if event.button == 1:               # add, snapped to this channel's PEAK (max)
                    p = int(snap_to_peaks([c], xn[v["ch"]], snap)[0])
                    pulses.add(p); v["last"] = p; redraw()
                elif event.button == 3 and pulses:  # remove nearest within ~10 ms
                    arr = centers(); j = int(np.argmin(np.abs(arr - c)))
                    if abs(arr[j] - c) < int(0.010 * FS):
                        pulses.discard(int(arr[j])); v["last"] = -1; redraw()
        fig.canvas.mpl_connect("button_press_event", on_click)

        def toggle_blank():
            v["blank"] = not v["blank"]; redraw()

        def page(d):
            v["center"] += int(d * v["win"] * FS); redraw()

        def chan(d):
            v["ch"] = int(np.clip(v["ch"] + d, 0, len(gidx) - 1)); redraw()

        def zoom(f):
            v["win"] = float(np.clip(v["win"] * f, 0.01, n / FS)); redraw()

        def bout(d):
            if bouts:
                v["bout"] = int(np.clip(v["bout"] + d, 0, len(bouts) - 1))
                s, e = bouts[v["bout"]]; v["center"] = (s + e) // 2; redraw()

        def on_key(e):
            k = e.key
            if k == "left": page(-1)
            elif k == "right": page(1)
            elif k == "up": chan(1)
            elif k == "down": chan(-1)
            elif k in ("+", "="): zoom(0.6)
            elif k == "-": zoom(1 / 0.6)
            elif k == "n": bout(1)
            elif k == "p": bout(-1)
            elif k == "b": toggle_blank()
        fig.canvas.mpl_connect("key_press_event", on_key)

        def autofill():                              # detect PEAKS on the viewed channel, your-mark-calibrated
            cc = detect_pulses(bouts, xn[v["ch"]], anchors=centers())
            pulses.clear(); pulses.update(int(c) for c in cc); v["last"] = -1; redraw()

        def clearp():
            pulses.clear(); v["last"] = -1; redraw()

        def save():
            np.save(stim_pulse_path(key), centers())
            print(f"{key}: saved {len(pulses)} stim pulses -> stimpulses_{key}.npy", flush=True)
            plt.close(fig)

        def skip():
            print(f"{key}: skipped (run left un-blanked)", flush=True); plt.close(fig)

        def _btn(x0, w, label, fn):
            b = Button(plt.axes([x0, 0.02, w, 0.055]), label); b.on_clicked(lambda _: fn())
            return b
        _b = [_btn(0.05, 0.10, "◀ Bout", lambda: bout(-1)), _btn(0.16, 0.10, "Bout ▶", lambda: bout(1)),
              _btn(0.28, 0.11, "Auto-fill", autofill), _btn(0.40, 0.08, "Clear", clearp),
              _btn(0.49, 0.12, "Blank prev.", toggle_blank),
              _btn(0.72, 0.10, "Save", save), _btn(0.84, 0.10, "Skip", skip)]
        redraw()
        _keep = _b                                   # keep widgets alive
        plt.show()
