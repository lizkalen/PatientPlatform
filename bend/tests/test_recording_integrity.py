"""Tests for recording durability: the spool, recovery, and save failures.

Layer under test: the recording path of both WebSocket servers plus
`server.recover_recording`. Before this work a recording lived as a list of
chunks in RAM and was pickled exactly once, at stop (architecture audit P5):

  * ~7 GB/hour of resident memory at Quattrocento rates;
  * any process death lost 100% of the block;
  * a `pkl.dump` failure escaped, skipped the state reset, and the *next*
    `start_recording` then wiped the chunks it had left behind;
  * the multi-GB pickle ran on the event loop, stalling every queued command
    behind it, `stimulate_stop` included (audit P6);
  * filenames were second-resolution timestamps with no subject token, so two
    stops in one second silently overwrote each other.

Samples now stream to an on-disk spool as they arrive, the final `.pkl` is
reassembled in a worker thread, and the spool is deleted only once that file is
durably written. These tests pin all of that, plus the `_mv_paused` latch that
used to make later recordings silently empty (audit B2).

Most checks need real array I/O, so they run only where numpy is installed and
print a SKIP otherwise - the same convention `bend/src/mvdecoder/tests/
test_parity.py` uses for its data-dependent checks. On the `patientgui` conda
environment everything runs.

Run:  py -3 bend/tests/test_recording_integrity.py
      (or `pytest bend/tests/test_recording_integrity.py`)
"""

import asyncio
import os
import pickle
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _stim_stubs import (                                        # noqa: E402
    Sock, arun, ensure_src_on_path, import_servers, ok, run_suite,
)

ensure_src_on_path()

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:                                              # bare Python
    HAVE_NUMPY = False

RippleWebSocketServer, SimulatedWebSocketServer, _ConnectionClosed = import_servers()

import server.recover_recording as rr                            # noqa: E402


def _skip(what):
    print(f"  [skip] {what} (numpy is not installed in this interpreter)")


class _Workspace:
    """A throwaway output folder."""

    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="recording_test_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def _server(output, n_channels=4, srate=100.0, **kwargs):
    """A server wired for recording only: no device, no sockets, no broadcast."""
    srv = RippleWebSocketServer(output_folder=output, **kwargs)
    srv.sample_rate = srate
    srv.n_channels = n_channels
    srv.sent = []

    async def record_broadcast(message):
        srv.sent.append(message)

    srv.broadcast = record_broadcast
    return srv


def _chunk(n_channels, n_samples, start=0.0, dtype="float64"):
    """A recognisable (n_channels, n_samples) block."""
    values = np.arange(start, start + n_channels * n_samples, dtype=dtype)
    return values.reshape(n_channels, n_samples)


def _spools(folder):
    return [n for n in os.listdir(folder) if n.endswith(rr.SPOOL_SUFFIX)]


def _pkls(folder):
    return sorted(n for n in os.listdir(folder) if n.endswith(".pkl")
                  and not n.endswith(rr.SPOOL_SUFFIX + ".pkl"))


META = {"subjectId": "P 07", "sessionId": "sess/2"}


# ── the spool ────────────────────────────────────────────────────────────────

def test_spool_is_written_incrementally_and_nothing_is_kept_in_ram():
    """Every chunk must reach disk as it arrives, and RAM must not grow."""
    if not HAVE_NUMPY:
        return _skip("spool incremental write")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)

        ok(len(_spools(output)) == 1, "a spool file exists as soon as recording starts")
        spool = os.path.join(output, _spools(output)[0])
        ok(os.path.exists(rr.sidecar_path(spool)),
           "the sidecar is written AT START, so a crash leaves it recoverable")

        ok(not hasattr(srv.recording, "recorded_chunks"),
           "the in-RAM chunk list is gone from RecordingState entirely")

        sizes = []
        for i in range(5):
            srv._spool_write(_chunk(4, 10, start=i * 40))
            sizes.append(os.path.getsize(spool))

        # Growth measured ON DISK, not in the writer's buffer: a killed process
        # keeps only what was actually flushed.
        per_chunk = 4 * 10 * np.dtype("float64").itemsize
        ok(sizes == [per_chunk * (i + 1) for i in range(5)],
           f"the file on disk grows by exactly one chunk each time ({sizes})")
        ok(srv.recording.spool_samples == 50,
           f"50 samples spooled, state says {srv.recording.spool_samples}")

        # RAM proxy: no attribute on the recording state holds the samples.
        held = [name for name, value in vars(srv.recording).items()
                if isinstance(value, list) and value]
        ok(not held, f"no sample list is retained on the recording state ({held})")

        await srv.stop_recording(None)

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_saved_file_matches_the_pre_spool_format():
    """The .pkl must be what downstream analysis already loads."""
    if not HAVE_NUMPY:
        return _skip("saved-file format")

    async def scenario(output):
        srv = _server(output)
        pre = _chunk(4, 7, start=1000.0)
        srv.rolling_buffer = [pre[:, i] for i in range(pre.shape[1])]

        await srv.start_recording(META)
        chunks = [_chunk(4, 10, start=i * 40) for i in range(3)]
        for chunk in chunks:
            srv._spool_write(chunk)
        await srv.stop_recording({"phases": ["A", "B"]})

        files = _pkls(output)
        ok(len(files) == 1, f"exactly one .pkl was written ({files})")
        with open(os.path.join(output, files[0]), "rb") as f:
            saved = pickle.load(f)

        ok(list(saved.keys())[:9] == [
            "data", "srate", "n_channels", "filtered", "pre_trigger_seconds",
            "timestamp", "stream_type", "trial_metadata", "session_timeline"],
           f"the key order is unchanged ({list(saved.keys())})")

        expected = np.hstack([pre] + chunks)
        ok(saved["data"].shape == expected.shape,
           f"shape {saved['data'].shape} == hstack shape {expected.shape}")
        ok(np.array_equal(saved["data"], expected),
           "the reassembled samples are identical to the old hstack result, "
           "pre-trigger first")
        ok(saved["data"].dtype == expected.dtype,
           f"dtype preserved ({saved['data'].dtype}), not silently narrowed")
        ok(saved["data"].flags["C_CONTIGUOUS"],
           "the array is C-contiguous, as hstack produced")
        ok(saved["trial_metadata"] == META and saved["n_channels"] == 4
           and saved["session_timeline"] == {"phases": ["A", "B"]},
           "metadata, channel count and timeline round-trip")
        ok("recovered_from_spool" not in saved,
           "a normally saved file is not marked as recovered")

        ok(not _spools(output),
           "the spool is deleted once the .pkl is durably written")
        ok("P-07" in files[0] and "sess-2" in files[0],
           f"the filename carries sanitized subject/session tokens ({files[0]})")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_crash_leaves_a_spool_that_recovers_to_the_same_recording():
    """The whole point: a killed backend must not cost the session."""
    if not HAVE_NUMPY:
        return _skip("crash recovery")

    async def scenario(output):
        chunks = [_chunk(4, 10, start=i * 40) for i in range(3)]

        # 1. A clean recording, saved normally.
        clean = _server(output)
        await clean.start_recording(META)
        for chunk in chunks:
            clean._spool_write(chunk)
        await clean.stop_recording(None)
        with open(os.path.join(output, _pkls(output)[0]), "rb") as f:
            direct = pickle.load(f)

        # 2. The same recording, but the process "dies": the file handle is
        #    never closed and stop_recording is never reached. Nothing is
        #    flushed on the way out - only the per-chunk flush saves this.
        crashed = _server(output)
        await crashed.start_recording(META)
        for chunk in chunks:
            crashed._spool_write(chunk)
        orphan = crashed.recording.spool_path
        leaked_handle = crashed.recording.spool_file  # deliberately NOT closed
        crashed.recording.is_recording = False

        ok(os.path.exists(orphan), "the orphaned spool survives the crash")
        ok(os.path.exists(rr.sidecar_path(orphan)),
           "so does its sidecar, written back at start")

        recovered_path = rr.recover(orphan)
        with open(recovered_path, "rb") as f:
            recovered = pickle.load(f)

        ok(np.array_equal(recovered["data"], direct["data"]),
           "recovered samples are identical to the directly-saved ones")
        ok(recovered["data"].dtype == direct["data"].dtype,
           "recovered dtype matches")
        for key in ("srate", "n_channels", "filtered", "pre_trigger_seconds",
                    "stream_type", "trial_metadata"):
            ok(recovered[key] == direct[key], f"recovered '{key}' matches")
        ok(recovered.get("recovered_from_spool") is True,
           "the recovered file is honestly flagged as recovered")
        ok(recovered["session_timeline"] is None,
           "the timeline is None: it only ever existed in the dead process")

        leaked_handle.close()                         # so the temp dir can go

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_recover_cli_entry_point():
    """`python -m server.recover_recording <spool>` must work end to end."""
    if not HAVE_NUMPY:
        return _skip("recover CLI")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)
        srv._spool_write(_chunk(4, 10))
        orphan = srv.recording.spool_path
        srv.recording.spool_file.close()
        srv.recording.is_recording = False

        code = rr.main([orphan])
        ok(code == 0, "the CLI exits 0")
        ok(any(n.endswith("_recovered.pkl") for n in os.listdir(output)),
           "the CLI wrote a recovered .pkl")

        listing = rr.main(["--list", output])
        ok(listing == 0, "--list exits 0 and reports the spool")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_partial_trailing_sample_is_tolerated():
    """A writer killed mid-sample must not make the whole spool unreadable."""
    if not HAVE_NUMPY:
        return _skip("truncated spool")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)
        srv._spool_write(_chunk(4, 10))
        orphan = srv.recording.spool_path
        srv.recording.spool_file.close()
        srv.recording.is_recording = False

        with open(orphan, "ab") as f:                 # half a sample of garbage
            f.write(b"\x01\x02\x03\x04\x05")

        data = rr.load_spool(orphan, rr.read_sidecar(orphan))
        ok(data.shape == (4, 10),
           f"the trailing fragment is discarded, 10 samples kept ({data.shape})")

    with _Workspace() as output:
        arun(lambda: scenario(output))


# ── save failures ────────────────────────────────────────────────────────────

def test_failed_save_keeps_the_spool_and_resets_state():
    """A save failure must cost neither the data nor the next recording."""
    if not HAVE_NUMPY:
        return _skip("failed save")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)
        srv._spool_write(_chunk(4, 10))
        spool = srv.recording.spool_path

        def explode(*a, **k):
            raise OSError("No space left on device")

        srv._finalize_recording = explode
        srv.sent.clear()
        await srv.stop_recording(None)
        del srv._finalize_recording           # the next save must work again

        ok(os.path.exists(spool), "the spool is KEPT when the save fails")
        ok(os.path.exists(rr.sidecar_path(spool)), "so is its sidecar")
        ok(not _pkls(output), "no half-written .pkl was left behind")

        errors = [m for m in srv.sent if m.get("type") == "error"]
        ok(len(errors) == 1, f"one error was broadcast ({len(errors)})")
        ok(spool in errors[0]["message"],
           "the error names the spool path so the operator can find it")
        ok("recover_recording" in errors[0]["message"],
           "the error names the recovery command")
        statuses = [m for m in srv.sent if m.get("type") == "recording_status"]
        ok(statuses and statuses[-1]["recording"] is False,
           "the UI is still told recording stopped, so the flag unlatches")

        ok(srv.recording.spool_path is None and srv.recording.metadata is None
           and srv.recording.spool_samples == 0,
           "state is reset even though the save failed")

        # The next recording must work, and must not touch the retained spool.
        await srv.start_recording({"subjectId": "next"})
        srv._spool_write(_chunk(4, 10))
        await srv.stop_recording(None)
        ok(len(_pkls(output)) == 1, "the next recording saves normally")
        ok(os.path.exists(spool),
           "the retained spool from the failed save is still untouched")

        recovered = rr.recover(spool)
        ok(os.path.exists(recovered), "and it still recovers to a real .pkl")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_filename_collision_does_not_overwrite():
    """Two recordings in the same second must produce two files."""
    if not HAVE_NUMPY:
        return _skip("filename collision")

    async def scenario(output):
        srv = _server(output)
        for _ in range(3):
            await srv.start_recording(META)          # same second, same tokens
            srv._spool_write(_chunk(4, 5))
            await srv.stop_recording(None)

        files = _pkls(output)
        ok(len(files) == 3, f"three distinct recordings on disk ({files})")
        ok(len(set(files)) == 3, "no name was reused")
        ok(not _spools(output), "and every spool was cleaned up")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_spool_write_failure_is_counted_and_reported():
    """A gap in the data must never be passed off as a complete recording."""
    if not HAVE_NUMPY:
        return _skip("spool write failure")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)
        srv._spool_write(_chunk(4, 10))

        class Brokenwriter:
            def write(self, _):
                raise OSError("disk went away")

        good_writer = srv.recording.spool_file
        srv.recording.spool_file = Brokenwriter()
        srv._spool_write(_chunk(4, 10))               # lost
        srv.recording.spool_file = good_writer
        srv._spool_write(_chunk(4, 10))

        ok(srv.recording.spool_write_errors == 1, "the failure was counted")
        ok(srv.recording.spool_dropped_samples == 10, "the lost samples were counted")

        await srv.stop_recording(None)
        with open(os.path.join(output, _pkls(output)[0]), "rb") as f:
            saved = pickle.load(f)
        ok(saved.get("spool_write_errors", {}).get("count") == 1,
           "the saved file records that it has a gap")
        ok(saved["data"].shape[1] == 20,
           f"only the samples that reached disk are present "
           f"({saved['data'].shape[1]})")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_no_data_recorded_cleans_up_and_resets():
    """An empty recording must leave nothing behind."""
    if not HAVE_NUMPY:
        return _skip("empty recording")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)               # no rolling buffer, no chunks
        srv.sent.clear()
        await srv.stop_recording(None)

        ok(not _spools(output), "the empty spool is removed")
        ok(not _pkls(output), "no .pkl is written for an empty recording")
        ok(any("No data" in (m.get("message") or "") for m in srv.sent),
           "the operator is told nothing was recorded")
        ok(srv.recording.spool_path is None, "state is reset")

    with _Workspace() as output:
        arun(lambda: scenario(output))


# ── the _mv_paused latch (audit B2) ──────────────────────────────────────────

def test_mv_record_stop_clears_the_pause_latch():
    """B2: stop never cleared it, so later recordings were silently empty."""
    async def scenario(output):
        srv = _server(output)
        await srv.handle_message('{"command": "mv_record_start"}', Sock())
        await srv.handle_message('{"command": "mv_record_pause"}', Sock())
        ok(srv._mv_paused is True, "pause latches during the tutorial phase")

        await srv.handle_message('{"command": "mv_record_stop"}', Sock())
        ok(srv._mv_paused is False,
           "mv_record_stop clears the latch (the frontend never sends resume)")
        ok(srv._mv_capturing is False, "and ends the capture")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_start_recording_clears_a_latched_pause():
    """Defence in depth: a fresh recording never starts silently paused."""
    if not HAVE_NUMPY:
        return _skip("start_recording pause clear")

    async def scenario(output):
        srv = _server(output)
        srv._mv_paused = True                         # an aborted session's leftover

        await srv.start_recording(META)
        ok(srv._mv_paused is False,
           "start_recording clears the stale latch instead of recording nothing")

        srv._spool_write(_chunk(4, 10))
        await srv.stop_recording(None)
        with open(os.path.join(output, _pkls(output)[0]), "rb") as f:
            saved = pickle.load(f)
        ok(saved["data"].shape[1] == 10,
           "the recording actually contains samples")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_mv_record_start_refuses_a_second_capture():
    """A duplicate start used to silently discard the block in progress."""
    async def scenario(output):
        srv = _server(output)
        await srv.handle_message('{"command": "mv_record_start"}', Sock())
        srv._mv_raw = ["chunk-a", "chunk-b"]          # a capture worth keeping

        sock = Sock("second")
        await srv.handle_message('{"command": "mv_record_start"}', sock)

        ok(len(sock.sent) == 1 and "already in progress" in sock.sent[0],
           "the second start is refused, and the requester is told why")
        ok(srv._mv_raw == ["chunk-a", "chunk-b"],
           "the in-progress capture is untouched")
        ok(srv._mv_capturing is True, "and still running")

    with _Workspace() as output:
        arun(lambda: scenario(output))


# ── finalising an unowned recording ──────────────────────────────────────────

class _EmptySocket(Sock):
    """A socket that connects and immediately ends its message stream."""

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def test_last_client_disconnect_finalizes_the_recording():
    """Nobody left to stop it means it would otherwise grow forever."""
    if not HAVE_NUMPY:
        return _skip("last-client finalize")

    async def scenario(output):
        srv = _server(output)
        await srv.start_recording(META)
        srv._spool_write(_chunk(4, 10))

        await srv.handle_client(_EmptySocket("only"))   # connects, then leaves

        ok(srv.recording.is_recording is False,
           "the recording is finalised when the last client goes")
        ok(len(_pkls(output)) == 1, "and it is on disk as a normal .pkl")
        ok(not _spools(output), "with its spool cleaned up")

    with _Workspace() as output:
        arun(lambda: scenario(output))


def test_sim_server_teardown_saves_an_active_recording():
    """End of playback lands in run()'s finally; the block must survive."""
    if not HAVE_NUMPY:
        return _skip("sim teardown save")

    async def scenario(output):
        srv = SimulatedWebSocketServer(npz_path="unused.npz", output_folder=output)
        srv.sample_rate = 100.0
        srv.n_channels = 4
        srv.sent = []

        async def record_broadcast(message):
            srv.sent.append(message)

        srv.broadcast = record_broadcast

        await srv.start_recording(META)
        srv._spool_write(_chunk(4, 10))
        ok(srv.recording.is_recording is True, "the sim server is recording")

        await srv.teardown()

        ok(srv.recording.is_recording is False, "teardown finalised it")
        ok(len(_pkls(output)) == 1,
           "the sim server saves an active recording at end of playback")
        ok(not _spools(output), "with its spool cleaned up")
        with open(os.path.join(output, _pkls(output)[0]), "rb") as f:
            saved = pickle.load(f)
        ok(saved["stream_type"] == "simulated" and saved["data"].shape == (4, 10),
           "the saved file is a normal simulated recording")

    with _Workspace() as output:
        arun(lambda: scenario(output))


TESTS = [
    test_spool_is_written_incrementally_and_nothing_is_kept_in_ram,
    test_saved_file_matches_the_pre_spool_format,
    test_crash_leaves_a_spool_that_recovers_to_the_same_recording,
    test_recover_cli_entry_point,
    test_partial_trailing_sample_is_tolerated,
    test_failed_save_keeps_the_spool_and_resets_state,
    test_filename_collision_does_not_overwrite,
    test_spool_write_failure_is_counted_and_reported,
    test_no_data_recorded_cleans_up_and_resets,
    test_mv_record_stop_clears_the_pause_latch,
    test_start_recording_clears_a_latched_pause,
    test_mv_record_start_refuses_a_second_capture,
    test_last_client_disconnect_finalizes_the_recording,
    test_sim_server_teardown_saves_an_active_recording,
]


if __name__ == "__main__":
    sys.exit(run_suite("recording integrity: spool, recovery, save failures",
                       TESTS))
