"""Loading recordings, feature caches and channel masks.

`load_recording` returns EMG as [channels, samples] from either format used in this
project:
  * a `.raw` file (float32, channel-interleaved) with a `.pkl` sidecar giving `shape`
    and `raw_file`;
  * a `<name>_raw.pkl` dict with an `'emg'` matrix (the preerlplots format).

`load_feat_cache` reads a feat_v2 `.npz` (F, A, E, S, gt, boutid, repid).
`load_good_indices` reads a good_mask `.npy` into channel indices.

numpy only.
"""
from __future__ import annotations

import os
import pickle

import numpy as np


def _load_raw_with_sidecar(raw_path: str, side: dict) -> np.ndarray:
    """Read a channel-interleaved float32 .raw into [channels, samples] using the
    channel count from its sidecar dict."""
    nch = int(np.asarray(side["shape"]).ravel()[0])
    flat = np.memmap(raw_path, dtype=np.float32, mode="r")
    m = flat.size // nch
    return np.asarray(flat[:m * nch]).reshape(m, nch).T


def load_recording(path, trig_ch: int | None = None):
    """Load a recording. Returns (data[channels, samples], trig or None, meta).
    `trig_ch` selects the trigger row; pass None if the recording has no trigger."""
    path = str(path)
    meta = {"path": path}

    if path.endswith(".raw"):
        side_path = path[:-4] + ".pkl"
        with open(side_path, "rb") as f:
            side = pickle.load(f)
        data = _load_raw_with_sidecar(path, side)
        meta["format"] = "raw+sidecar"
    elif path.endswith(".pkl"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "emg" in obj:
            data = np.asarray(obj["emg"])
            meta["format"] = "pkl-emg"
        elif isinstance(obj, dict) and "raw_file" in obj:
            raw_path = os.path.join(os.path.dirname(path), os.path.basename(obj["raw_file"]))
            data = _load_raw_with_sidecar(raw_path, obj)
            meta["format"] = "pkl-sidecar"
        else:
            data = np.asarray(obj)
            meta["format"] = "pkl-array"
    elif path.endswith(".npy"):
        data = np.load(path)
        meta["format"] = "npy"
    else:
        raise ValueError(f"unrecognised recording format: {path}")

    data = np.asarray(data)
    meta["n_channels"], meta["n_samples"] = int(data.shape[0]), int(data.shape[1])
    trig = None
    if trig_ch is not None:
        if trig_ch >= data.shape[0]:
            raise ValueError(f"trig_ch={trig_ch} out of range ({data.shape[0]} channels)")
        trig = np.asarray(data[trig_ch], float)
    return data, trig, meta


def load_feat_cache(path) -> dict:
    """Load a feat_v2 .npz into a dict with F, A, E, S, gt, boutid, repid."""
    d = np.load(str(path), allow_pickle=True)
    out = {}
    for k in ("F", "A", "E", "S", "gt", "boutid", "repid"):
        if k in d:
            out[k] = d[k]
    return out


def load_good_indices(path) -> np.ndarray:
    """Load a good_mask .npy (boolean per channel) into good-channel indices."""
    mask = np.load(str(path))
    return np.where(mask)[0]
