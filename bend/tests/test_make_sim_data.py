"""Tests for the synthetic simulation-data generator.

Layer under test: `server.make_sim_data`, which produces the ``.npz`` that
`start_sim.bat` feeds to the simulated backend. Simulation mode shipped broken
for its whole life - the launcher passed `sim/datafile1_filtered.npz` and no
such file has ever existed in the repository (architecture audit P3) - so the
contract that matters here is narrow and mechanical: whatever this writes,
`devices/simulated.py` must be able to load and stream.

These checks therefore pin the FORMAT (key, shape, dtype) and prove a real
`SimulatedDevice` consumes the result, not the signal's realism. The data is
deliberately not physiological; see the module docstring.

Needs numpy, so it runs on the `patientgui` conda environment and skips with a
message on a bare Python, per the convention in `mvdecoder/tests/test_parity.py`.

Run:  py -3 bend/tests/test_make_sim_data.py
      (or `pytest bend/tests/test_make_sim_data.py`)
"""

import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from _stim_stubs import ensure_src_on_path, ok, run_suite       # noqa: E402

ensure_src_on_path()

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

if HAVE_NUMPY:
    from server import make_sim_data as gen
else:                                                            # bare Python
    gen = None


def _skip(what):
    print(f"  [skip] {what} (numpy is not installed in this interpreter)")


class _Workspace:
    def __enter__(self):
        self.path = tempfile.mkdtemp(prefix="sim_data_test_")
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
        return False


def test_generated_npz_has_the_format_the_device_requires():
    """key 'data', shape (channels, samples), float32 - nothing else matters."""
    if not HAVE_NUMPY:
        return _skip("npz format")

    with _Workspace() as folder:
        out = os.path.join(folder, "sim.npz")
        code = gen.main([out, "--channels", "8", "--seconds", "2",
                         "--srate", "500", "--seed", "1"])
        ok(code == 0, "the CLI exits 0")
        ok(os.path.exists(out), "the .npz is written")

        loaded = np.load(out)
        ok("data" in loaded, f"it carries the 'data' key ({list(loaded.keys())})")
        data = loaded["data"]
        ok(data.shape == (8, 1000),
           f"shape is (channels, samples) = {data.shape}")
        ok(data.dtype == np.float32, f"dtype is float32 ({data.dtype})")
        ok(bool(np.isfinite(data).all()),
           "every sample is finite - no NaN or inf to poison the filters")
        ok(not os.path.exists(out + ".partial.npz"),
           "the atomic temp file is cleaned up")


def test_simulated_device_streams_the_generated_file():
    """The real consumer must accept it, not just numpy."""
    if not HAVE_NUMPY:
        return _skip("SimulatedDevice round-trip")

    try:
        from server.devices.simulated import SimulatedDevice
    except Exception as exc:                                     # noqa: BLE001
        return _skip(f"SimulatedDevice import failed: {exc}")

    import time

    with _Workspace() as folder:
        out = os.path.join(folder, "sim.npz")
        gen.main([out, "--channels", "6", "--seconds", "0.5", "--srate", "500"])

        device = SimulatedDevice(npz_path=out, srate=500.0)
        ok(len(device.elec_ids) == 6,
           f"the device reports 6 channels ({len(device.elec_ids)})")

        # The device paces playback against the wall clock, so the first fetch
        # only starts the clock and legitimately returns nothing. Poll the way
        # the server's stream loop does.
        first = None
        total = 0
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            chunk, _ts = device.fetch()
            if chunk is not None and chunk.size:
                if first is None:
                    first = chunk
                total += chunk.shape[0]
            if device.finished:
                break
            time.sleep(0.01)

        ok(first is not None, "the device streams the generated data")
        ok(first.shape[1] == 6,
           f"chunks are [samples, channels] as the server expects ({first.shape})")
        ok(first.dtype == np.float32, f"and stay float32 ({first.dtype})")
        ok(device.finished, "playback reaches the end of the file")
        ok(total == 250, f"every generated sample is streamed exactly once ({total})")


def test_defaults_match_what_the_launcher_promises():
    """start_sim.bat tells the operator 64 ch / 60 s / 2000 Hz."""
    if not HAVE_NUMPY:
        return _skip("defaults")

    ok(gen.DEFAULT_CHANNELS == 64, f"64 channels ({gen.DEFAULT_CHANNELS})")
    ok(gen.DEFAULT_SECONDS == 60.0, f"60 seconds ({gen.DEFAULT_SECONDS:g})")
    ok(gen.DEFAULT_SRATE == 2000.0, f"2000 Hz ({gen.DEFAULT_SRATE:g})")
    megabytes = 64 * 60 * 2000 * 4 / 1e6
    ok(megabytes < 40, f"the default file stays around 30 MB ({megabytes:.0f} MB)")


def test_signal_has_rest_and_burst_periods():
    """Not a realism check - just that the envelope actually modulates.

    A flat array would still satisfy the format contract while making the
    frontend plot and the filters useless for eyeballing the pipeline.
    """
    if not HAVE_NUMPY:
        return _skip("burst envelope")

    data = gen.make_emg_like(n_channels=4, seconds=6.0, srate=500.0, seed=3)
    quiet = float(np.abs(data[:, :400]).mean())        # first 0.8 s: baseline
    loud = float(np.abs(data[:, 600:1400]).mean())     # inside the first burst
    ok(loud > quiet * 3,
       f"bursts are clearly above the resting baseline ({quiet:.4f} -> {loud:.4f})")
    ok(quiet > 0, "and the baseline is not digital silence")


def test_reproducible_and_refuses_to_clobber():
    """A given seed must reproduce, and a stale file must not be overwritten."""
    if not HAVE_NUMPY:
        return _skip("seeding and overwrite")

    first = gen.make_emg_like(n_channels=3, seconds=1.0, srate=500.0, seed=7)
    again = gen.make_emg_like(n_channels=3, seconds=1.0, srate=500.0, seed=7)
    other = gen.make_emg_like(n_channels=3, seconds=1.0, srate=500.0, seed=8)
    ok(np.array_equal(first, again), "the same seed reproduces the same data")
    ok(not np.array_equal(first, other), "a different seed does not")

    with _Workspace() as folder:
        out = os.path.join(folder, "sim.npz")
        ok(gen.main([out, "--channels", "2", "--seconds", "1",
                     "--srate", "500"]) == 0, "the first write succeeds")
        ok(gen.main([out, "--channels", "2", "--seconds", "1",
                     "--srate", "500"]) == 1,
           "a second write refuses rather than clobbering existing data")
        ok(gen.main([out, "--channels", "2", "--seconds", "1",
                     "--srate", "500", "--force"]) == 0,
           "--force overwrites deliberately")


TESTS = [
    test_generated_npz_has_the_format_the_device_requires,
    test_simulated_device_streams_the_generated_file,
    test_defaults_match_what_the_launcher_promises,
    test_signal_has_rest_and_burst_periods,
    test_reproducible_and_refuses_to_clobber,
]


if __name__ == "__main__":
    sys.exit(run_suite("simulation data generator", TESTS))
