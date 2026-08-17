"""
Recording spool recovery
========================

Turns an orphaned recording *spool* - the file the server appends to while a
recording is running - into a standard ``.pkl`` recording file.

Usage::

    python -m server.recover_recording <spool-file> [-o OUTPUT.pkl]
    python -m server.recover_recording --list <folder>

Why spools exist (architecture audit P5): recordings used to accumulate in RAM
as a list of chunks and were pickled exactly once, at stop. At Quattrocento
rates that is ~2 MB/s (~7 GB/hour) of resident memory, and any process death -
a closed console window, a crash, a power cut, a `taskkill` - lost **100%** of
the block. The server now appends every chunk to a spool as it arrives, so the
worst case is losing the last partial chunk rather than the whole session, and
memory no longer grows with recording length.

Spool format
------------
Two files per recording, side by side in the output folder::

    emg_recording_<subject>_<session>_<start-ts>.spool       raw samples
    emg_recording_<subject>_<session>_<start-ts>.spool.pkl   sidecar (metadata)

The spool is a headerless stream of fixed-width samples in **samples-major**
order - exactly what ``chunk.T`` produces, appended chunk after chunk::

    sample 0: [ch0, ch1, ... chN-1]
    sample 1: [ch0, ch1, ... chN-1]
    ...

so the whole recording is::

    np.fromfile(spool, dtype).reshape(-1, n_channels).T

There is deliberately no per-chunk framing: chunk boundaries carry no meaning
once the stream is reassembled, and a headerless stream stays readable even if
the writer died mid-chunk - `load_spool` simply discards a trailing partial
sample. The pre-trigger buffer is written FIRST, as part of the same stream, so
the reassembled array is identical to what the old in-RAM path produced.

This is the same raw+sidecar pattern the movement online-run writer already
used (``_open_online_writer`` in websocket_server.py), generalised so both the
server and this tool share one implementation instead of two that can drift.

The sidecar is a pickled dict carrying everything needed to interpret the spool
plus the recording metadata. It is written **at start**, not at close, so an
orphan left behind by a crash is always reconstructable. It is rewritten twice
more: once on the first chunk (the sample dtype is only knowable once a chunk
exists - and a spool with no chunk yet has nothing to recover), and once at
close with the final sample count.

What recovery cannot restore
----------------------------
Two things live only in the server's RAM and die with it:

  * ``decomposition`` - per-chunk firing rates / spikes / sources / SIL scores,
    when a decomposition model was running;
  * ``session_timeline`` - the phase timeline the frontend sends along with the
    ``stop_recording`` command.

A recovered file therefore carries ``recovered_from_spool: True`` and omits
those two keys. Everything else - ``data``, ``srate``, ``n_channels``,
``filtered``, ``pre_trigger_seconds``, ``timestamp``, ``stream_type``,
``trial_metadata`` - is identical to a normally saved recording, because the
server builds its own file with the very same functions below.
"""

import argparse
import os
import pickle as pkl
import sys
from datetime import datetime
from typing import Optional

import numpy as np

SPOOL_SUFFIX = ".spool"
SPOOL_KIND = "emg_recording_spool"
SPOOL_LAYOUT = ("samples-major: np.fromfile(spool, dtype).reshape(-1, n_channels).T")


# ── paths ────────────────────────────────────────────────────────────────────

def sidecar_path(spool_path: str) -> str:
    """``<name>.spool`` -> ``<name>.spool.pkl``."""
    return spool_path + ".pkl"


def safe_name(s, default: str = "x") -> str:
    """Filesystem-safe token for building informative recording filenames.

    Lives here rather than on a server class so both WebSocket servers and this
    tool agree on how a subject/session token is spelled on disk.
    """
    s = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(s or default))
    return s.strip("-. ") or default


def unique_path(path: str) -> str:
    """First free name in the ``x.pkl`` / ``x_2.pkl`` / ``x_3.pkl`` series.

    Recording filenames are second-resolution timestamps, so two stops inside
    one second used to silently overwrite each other (audit P5). Suffixing is
    preferred over adding more timestamp digits because it keeps the name
    readable and the ordering obvious.
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    for n in range(2, 1000):
        candidate = f"{stem}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
    # Absurd, but never return a name we know collides: fall back to microseconds.
    return f"{stem}_{datetime.now().strftime('%H%M%S%f')}{ext}"


# ── durable writes ───────────────────────────────────────────────────────────

def write_pickle_atomic(payload: dict, path: str) -> str:
    """Pickle to ``<path>.partial``, fsync, then rename into place.

    The rename is what makes "delete the spool only after the .pkl is durably
    written" a real guarantee rather than a hope: a reader never sees a
    half-written recording, and a crash mid-write leaves the spool plus an
    obvious ``.partial``, not a truncated file that looks complete.

    The default pickle protocol is used deliberately - the saved format must
    stay byte-compatible with what downstream analysis already loads.
    """
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        pkl.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def write_sidecar(spool_path: str, info: dict) -> str:
    """Write (or rewrite) the spool sidecar atomically."""
    path = sidecar_path(spool_path)
    tmp = path + ".partial"
    with open(tmp, "wb") as f:
        pkl.dump(info, f)
    os.replace(tmp, path)
    return path


def read_sidecar(spool_path: str) -> dict:
    """Load and sanity-check the sidecar next to a spool."""
    path = sidecar_path(spool_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no sidecar beside {os.path.basename(spool_path)} (expected "
            f"{os.path.basename(path)}). Without it the channel count and dtype "
            f"are unknown and the spool cannot be interpreted.")
    with open(path, "rb") as f:
        info = pkl.load(f)
    if not isinstance(info, dict) or info.get("kind") != SPOOL_KIND:
        raise ValueError(f"{path} is not a recording spool sidecar")
    return info


# ── reassembly ───────────────────────────────────────────────────────────────

def spool_sample_count(spool_path: str, info: dict) -> int:
    """Complete samples currently in the spool, from its size on disk."""
    n_channels = int(info.get("n_channels") or 0)
    dtype = np.dtype(info.get("dtype") or "float32")
    if n_channels <= 0:
        return 0
    return os.path.getsize(spool_path) // (n_channels * dtype.itemsize)


def load_spool(spool_path: str, info: dict) -> np.ndarray:
    """Reassemble a spool into the ``(n_channels, n_samples)`` recording array.

    Returns a C-contiguous array, matching the layout the old
    ``np.hstack`` path produced, so the pickled result is identical.
    """
    n_channels = int(info.get("n_channels") or 0)
    if n_channels <= 0:
        raise ValueError(f"sidecar reports n_channels={n_channels!r}; cannot "
                         f"interpret the spool")
    dtype = np.dtype(info.get("dtype") or "float32")

    flat = np.fromfile(spool_path, dtype=dtype)
    if flat.size == 0:
        return np.empty((n_channels, 0), dtype=dtype)

    complete = (flat.size // n_channels) * n_channels
    if complete != flat.size:
        # The writer died part-way through a sample. Drop the fragment rather
        # than refuse the whole recording over the last few bytes.
        print(f"[recover] discarding {flat.size - complete} trailing value(s): "
              f"the spool ends mid-sample (the writer was interrupted)")
        flat = flat[:complete]
    return np.ascontiguousarray(flat.reshape(-1, n_channels).T)


def build_recording_payload(data, info: dict, timestamp: Optional[str] = None,
                            timeline=None, decomposition=None,
                            recovered: bool = False) -> dict:
    """Build the standard recording dict.

    Key order matters as much as key names here: this is the format downstream
    analysis already loads, and the server and the recovery tool must produce
    the same thing. Optional keys are appended, never interleaved.
    """
    payload = {
        "data": data,
        "srate": info.get("srate"),
        "n_channels": info.get("n_channels"),
        "filtered": info.get("filtered"),
        "pre_trigger_seconds": info.get("pre_trigger_seconds"),
        "timestamp": timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"),
        "stream_type": info.get("stream_type"),
        "trial_metadata": info.get("trial_metadata"),
        "session_timeline": timeline,
    }
    if decomposition is not None:
        payload["decomposition"] = decomposition
    if info.get("spool_write_errors"):
        # Only present when something actually went wrong, so a clean recording
        # is byte-identical to one saved by the pre-spool code.
        payload["spool_write_errors"] = info["spool_write_errors"]
    if recovered:
        payload["recovered_from_spool"] = True
    return payload


# ── the tool ─────────────────────────────────────────────────────────────────

def recover(spool_path: str, out_path: Optional[str] = None) -> str:
    """Reassemble one orphaned spool into a ``.pkl``. Returns the written path."""
    spool_path = os.path.abspath(spool_path)
    if not os.path.exists(spool_path):
        raise FileNotFoundError(spool_path)

    info = read_sidecar(spool_path)
    data = load_spool(spool_path, info)
    if data.shape[1] == 0:
        raise ValueError(f"{os.path.basename(spool_path)} contains no samples; "
                         f"nothing to recover")

    if out_path is None:
        base = (spool_path[:-len(SPOOL_SUFFIX)] if spool_path.endswith(SPOOL_SUFFIX)
                else spool_path)
        out_path = unique_path(base + "_recovered.pkl")

    payload = build_recording_payload(data, info, recovered=True)
    written = write_pickle_atomic(payload, out_path)

    srate = info.get("srate") or 0
    duration = data.shape[1] / srate if srate else 0.0
    print(f"[recover] {os.path.basename(spool_path)} -> {os.path.basename(written)}")
    print(f"[recover]   {data.shape[0]} channels, {data.shape[1]} samples "
          f"({duration:.1f}s @ {srate or '?'} Hz), dtype {data.dtype}")
    if info.get("spool_write_errors"):
        print(f"[recover]   WARNING: the server logged spool write errors during "
              f"this recording: {info['spool_write_errors']}")
    print("[recover]   note: decomposition results and the session timeline are "
          "not recoverable (they only ever existed in the server's memory)")
    return written


def find_spools(folder: str) -> list:
    """Every ``*.spool`` in a folder, newest first."""
    if not os.path.isdir(folder):
        raise NotADirectoryError(folder)
    found = [os.path.join(folder, n) for n in os.listdir(folder)
             if n.endswith(SPOOL_SUFFIX)]
    return sorted(found, key=os.path.getmtime, reverse=True)


def _print_listing(folder: str) -> int:
    spools = find_spools(folder)
    if not spools:
        print(f"No orphaned spools in {folder}")
        return 0
    print(f"{len(spools)} spool(s) in {folder}:")
    for path in spools:
        try:
            info = read_sidecar(path)
            n = spool_sample_count(path, info)
            srate = info.get("srate") or 0
            duration = f"{n / srate:.1f}s" if srate else f"{n} samples"
            meta = info.get("trial_metadata") or {}
            who = f"{meta.get('subjectId') or '?'}/{meta.get('sessionId') or '?'}"
            print(f"  {os.path.basename(path):<60s} {duration:>10s}  {who}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {os.path.basename(path):<60s} UNREADABLE: {exc}")
    print("\nRecover one with:\n  python -m server.recover_recording <spool-file>")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m server.recover_recording",
        description="Rebuild a standard .pkl recording from an orphaned spool "
                    "left behind by a crashed or killed backend.")
    parser.add_argument("spool", nargs="?",
                        help="path to the .spool file to recover")
    parser.add_argument("-o", "--output",
                        help="output .pkl path (default: <spool>_recovered.pkl)")
    parser.add_argument("--list", metavar="FOLDER", dest="list_folder",
                        help="list orphaned spools in a recordings folder")
    args = parser.parse_args(argv)

    if args.list_folder:
        return _print_listing(args.list_folder)
    if not args.spool:
        parser.error("give a spool file to recover, or --list FOLDER")

    try:
        recover(args.spool, args.output)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[recover] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
