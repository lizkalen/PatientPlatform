"""Validity scoring and acceptance for motor units.

Independent of the decomposition that produced the units. The input is a spike
train and the channel-space signal it was extracted from; the source signal and
the separation filter are not used. Any decomposer that yields spike times on a
known signal can be scored here.

Scores are computed from the spike-triggered average (STA) over channels and
from the inter-spike intervals:

- reproducibility: correlation between STAs built from alternate spikes
- spatial extent: how many channels carry the action potential, and how much of
  the array does
- waveform: signal-to-noise, biphasic symmetry, deflection duration
- timing: fraction of inter-spike intervals below the refractory period

:func:`hardened_validity` combines them into an accept or reject decision. The
spatial test is two-sided: a unit is rejected both when the potential is spread
across the array (common mode) and when it appears on too few channels (single
channel artefact).

Thresholds live in :data:`DEFAULT_GATE` and can be overridden per call. They were
set on 64-channel HD-sEMG grids. The two spatial thresholds constrain the same
quantity from opposite sides and are jointly unsatisfiable on arrays of 25
channels or fewer; :func:`gate_is_satisfiable` reports this and
:func:`validate_units` warns when it holds.
"""
from __future__ import annotations

import warnings

import numpy as np

DEFAULT_GATE = {
    "min_spikes": 20,          # discharges required to score a unit at all
    "min_split_half_r": 0.6,   # correlation between alternate-spike STAs
    "min_wf_snr": 6.0,         # STA peak-to-peak over the STA baseline spread
    "min_n_focal_ch": 3,       # channels above half the maximum peak-to-peak
    "max_focality_frac": 0.12, # those channels as a fraction of the array
    "min_peak_floor": 6.0,     # maximum peak-to-peak over the median channel
    "min_biphasic": 0.25,      # smaller lobe over larger lobe on the peak channel
    "max_defl_ms": 6.0,        # time between the negative and positive peak
    "max_refrac_viol": 0.10,   # fraction of inter-spike intervals under 20 ms
}

REFRACTORY_S = 0.02


def spike_triggered_average(sig, spikes, half):
    """Average the signal over windows centred on each spike.

    Spikes closer than ``half`` samples to either edge are skipped.

    Args:
        sig (ndarray): Signal, (n_channels, n_samples).
        spikes (sequence): Spike sample indices.
        half (int): Half window length in samples.

    Returns:
        tuple:
            - ndarray: STA, (n_channels, 2 * half + 1). All zeros when no spike
              is usable.
            - int: Number of spikes averaged.
    """
    sig = np.asarray(sig, float)
    n_ch, n_samples = sig.shape
    sp = np.asarray(spikes, int)
    sp = sp[(sp >= half) & (sp + half + 1 <= n_samples)]
    if sp.size == 0:
        return np.zeros((n_ch, 2 * half + 1)), 0
    acc = np.zeros((n_ch, 2 * half + 1))
    for s in sp:
        acc += sig[:, s - half:s + half + 1]
    return acc / sp.size, int(sp.size)


def mu_validity(sig, spikes, fs, half_ms=10.0, top_m=16, min_spikes=None):
    """Score one spike train against the signal it was extracted from.

    Args:
        sig (ndarray): Channel-space signal the spikes index into,
            (n_channels, n_samples). This is the signal fed to the decomposer,
            not the source or the whitened signal.
        spikes (sequence): Spike sample indices.
        fs (float): Sampling frequency in Hz.
        half_ms (float): Half length of the STA window in milliseconds.
        top_m (int): Number of highest peak-to-peak channels used for the
            split-half correlation.
        min_spikes (int or None): Discharges required to score. Below this the
            scores stay NaN. None uses ``DEFAULT_GATE["min_spikes"]``.

    Returns:
        dict: Keys and meaning:
            - ``n`` (int): spikes inside the scoreable range
            - ``split_half_r`` (float): correlation between the STA of
              even-indexed and odd-indexed spikes, over ``top_m`` channels
            - ``focality_frac`` (float): fraction of channels whose STA
              peak-to-peak exceeds half the array maximum
            - ``n_focal_ch`` (int): count behind ``focality_frac``
            - ``peak_floor`` (float): maximum peak-to-peak over the median
              channel peak-to-peak
            - ``wf_snr`` (float): peak-to-peak on the dominant channel over the
              standard deviation of the STA window edges
            - ``biphasic`` (float): smaller lobe over larger lobe on the
              dominant channel, in [0, 1]
            - ``defl_ms`` (float): time between the negative and positive peak
              on the dominant channel
            - ``refrac_viol`` (float): fraction of inter-spike intervals below
              20 ms
            - ``dom`` (int): dominant channel, indexing ``sig`` rows; -1 when
              the unit was not scored
            - ``spikes`` (ndarray): the spike indices that were scored
    """
    gate_min = DEFAULT_GATE["min_spikes"] if min_spikes is None else min_spikes
    sig = np.asarray(sig, float)
    half = int(half_ms / 1000 * fs)
    n_samples = sig.shape[1]
    sp = np.sort(np.asarray(spikes, int))
    sp = sp[(sp >= half) & (sp + half + 1 <= n_samples)]

    out = dict(n=int(sp.size), split_half_r=np.nan, focality_frac=np.nan, n_focal_ch=0,
               peak_floor=np.nan, wf_snr=np.nan, biphasic=np.nan, defl_ms=np.nan,
               refrac_viol=np.nan, dom=-1, spikes=sp)
    if sp.size < gate_min:
        return out

    full, _ = spike_triggered_average(sig, sp, half)
    p2p = full.max(1) - full.min(1)
    dom = int(np.argmax(p2p))
    topch = np.argsort(p2p)[::-1][:top_m]

    even, _ = spike_triggered_average(sig, sp[0::2], half)
    odd, _ = spike_triggered_average(sig, sp[1::2], half)
    ve, vo = even[topch].ravel(), odd[topch].ravel()
    out["split_half_r"] = float(np.corrcoef(ve, vo)[0, 1]) if ve.std() > 0 and vo.std() > 0 else 0.0

    out["focality_frac"] = float(np.mean(p2p > 0.5 * p2p.max()))
    out["n_focal_ch"] = int(np.sum(p2p > 0.5 * p2p.max()))
    out["peak_floor"] = float(p2p.max() / (np.median(p2p) + 1e-12))

    edge_w = max(1, half // 2)
    edge = np.c_[full[:, :edge_w], full[:, -edge_w:]]
    out["wf_snr"] = float((full[dom].max() - full[dom].min()) / (np.median(edge.std(1)) + 1e-12))

    wf = full[dom]
    lo_i, hi_i = int(wf.argmin()), int(wf.argmax())
    out["biphasic"] = float(min(abs(wf[lo_i]), abs(wf[hi_i]))
                            / (max(abs(wf[lo_i]), abs(wf[hi_i])) + 1e-12))
    out["defl_ms"] = float(abs(hi_i - lo_i) / fs * 1000)

    isi = np.diff(sp) / fs
    out["refrac_viol"] = float(np.mean(isi < REFRACTORY_S)) if isi.size else np.nan
    out["dom"] = dom
    return out


def validity_failures(v, gate=None):
    """List the criteria a scored unit fails.

    Args:
        v (dict): Output of :func:`mu_validity`.
        gate (dict or None): Overrides for :data:`DEFAULT_GATE`. Keys not given
            fall back to the default.

    Returns:
        list[str]: Names of failed criteria, using the :data:`DEFAULT_GATE`
        keys. Empty when the unit passes. A unit scored below ``min_spikes``
        returns ``["min_spikes"]``.
    """
    g = dict(DEFAULT_GATE, **(gate or {}))
    if v["n"] < g["min_spikes"] or v["dom"] < 0:
        return ["min_spikes"]

    checks = [
        ("min_split_half_r", v["split_half_r"] > g["min_split_half_r"]),
        ("min_wf_snr", v["wf_snr"] > g["min_wf_snr"]),
        ("min_n_focal_ch", v["n_focal_ch"] >= g["min_n_focal_ch"]),
        ("max_focality_frac", v["focality_frac"] < g["max_focality_frac"]),
        ("min_peak_floor", v["peak_floor"] > g["min_peak_floor"]),
        ("min_biphasic", v["biphasic"] > g["min_biphasic"]),
        ("max_defl_ms", v["defl_ms"] < g["max_defl_ms"]),
        ("max_refrac_viol", v["refrac_viol"] < g["max_refrac_viol"]),
    ]
    return [name for name, ok in checks if not ok]


def gate_is_satisfiable(n_channels, gate=None):
    """Report whether any unit can pass the gate on an array of this size.

    ``min_n_focal_ch`` and ``max_focality_frac`` constrain the same quantity from
    opposite sides: the first is a count, the second is that count divided by the
    channel total. They admit a solution only when
    ``min_n_focal_ch / n_channels < max_focality_frac``. With the defaults, 3 and
    0.12, that needs more than 25 channels. Below that every unit is rejected
    regardless of its data, so raise ``max_focality_frac`` or lower
    ``min_n_focal_ch`` for smaller montages.

    Args:
        n_channels (int): Channels in the signal units are scored against, after
            channel selection.
        gate (dict or None): Overrides for :data:`DEFAULT_GATE`.

    Returns:
        bool: True when the two spatial criteria admit a solution.
    """
    g = dict(DEFAULT_GATE, **(gate or {}))
    return bool(g["min_n_focal_ch"] / max(1, n_channels) < g["max_focality_frac"])


def hardened_validity(v, gate=None):
    """Accept or reject a scored unit.

    Args:
        v (dict): Output of :func:`mu_validity`.
        gate (dict or None): Overrides for :data:`DEFAULT_GATE`.

    Returns:
        bool: True when the unit fails no criterion.
    """
    return len(validity_failures(v, gate)) == 0


def summarize_validity(units, gate=None, n_channels=None):
    """Count how many units the gate accepted and which criterion refused which.

    A unit that fails several criteria is listed under each of them, so the
    per-criterion counts sum to more than the number of refused units.

    Args:
        units (list[dict]): Output of :func:`validate_units`. Entries carrying a
            ``source_index`` key are reported under that value, so a summary
            built after filtering still refers to the pre-filter numbering.
        gate (dict or None): Overrides for :data:`DEFAULT_GATE`. Pass the same
            value used to score the units.
        n_channels (int or None): Channels the units were scored against. When
            given, ``satisfiable`` reports :func:`gate_is_satisfiable`.

    Returns:
        dict:
            - ``n_before`` (int): units scored
            - ``n_after`` (int): units passing every criterion
            - ``passed`` (list): indices of the accepted units
            - ``refused`` (list): indices of the refused units
            - ``refused_by`` (dict): criterion name to the indices it refused.
              Every key of :data:`DEFAULT_GATE` is present, empty when that
              criterion refused nothing.
            - ``n_refused_by`` (dict): criterion name to count
            - ``gate`` (dict): thresholds applied
            - ``satisfiable`` (bool or None): whether any unit could pass on this
              channel count; None when ``n_channels`` was not given
    """
    refused_by = {k: [] for k in DEFAULT_GATE}
    passed, refused = [], []
    for u in units:
        uid = u.get("source_index", u["unit"])
        for f in u["failures"]:
            refused_by.setdefault(f, []).append(uid)
        (passed if u["hardened"] else refused).append(uid)

    return {
        "n_before": len(units),
        "n_after": len(passed),
        "passed": passed,
        "refused": refused,
        "refused_by": refused_by,
        "n_refused_by": {k: len(v) for k, v in refused_by.items()},
        "gate": dict(DEFAULT_GATE, **(gate or {})),
        "satisfiable": None if n_channels is None else gate_is_satisfiable(n_channels, gate),
    }


def validate_units(sig, spikes, fs, gate=None, half_ms=10.0, top_m=16, channel_index=None):
    """Score and gate a set of units.

    Args:
        sig (ndarray): Channel-space signal the spikes index into,
            (n_channels, n_samples).
        spikes (dict or sequence): Spike indices per unit. A dict is iterated in
            sorted key order; a sequence is iterated in order.
        fs (float): Sampling frequency in Hz.
        gate (dict or None): Overrides for :data:`DEFAULT_GATE`.
        half_ms (float): Half length of the STA window in milliseconds.
        top_m (int): Channels used for the split-half correlation.
        channel_index (sequence or None): Maps ``sig`` rows back to the channel
            numbering of the original recording, as returned by
            ``preprocessing.select_channels``. When given, each unit gains a
            ``dom_channel`` key holding the original channel number.

    Returns:
        list[dict]: One entry per unit, in input order. Each entry holds the
        keys from :func:`mu_validity` plus:
            - ``unit`` (int or key): position in the input, or the dict key
            - ``hardened`` (bool): result of :func:`hardened_validity`
            - ``failures`` (list[str]): criteria failed
            - ``dom_channel`` (int): present only when ``channel_index`` is given
    """
    if not gate_is_satisfiable(np.shape(sig)[0], gate):
        g = dict(DEFAULT_GATE, **(gate or {}))
        warnings.warn(
            f"No unit can pass the gate on {np.shape(sig)[0]} channels: "
            f"min_n_focal_ch={g['min_n_focal_ch']} needs more than "
            f"{g['min_n_focal_ch'] / g['max_focality_frac']:.0f} channels at "
            f"max_focality_frac={g['max_focality_frac']}. Raise max_focality_frac "
            f"or lower min_n_focal_ch.",
            RuntimeWarning, stacklevel=2)

    items = ([(k, spikes[k]) for k in sorted(spikes)] if isinstance(spikes, dict)
             else list(enumerate(spikes)))

    units = []
    for key, sp in items:
        v = mu_validity(sig, sp, fs, half_ms=half_ms, top_m=top_m,
                        min_spikes=(gate or {}).get("min_spikes"))
        v["unit"] = key
        v["failures"] = validity_failures(v, gate)
        v["hardened"] = len(v["failures"]) == 0
        if channel_index is not None and v["dom"] >= 0:
            v["dom_channel"] = int(np.asarray(channel_index)[v["dom"]])
        units.append(v)
    return units
