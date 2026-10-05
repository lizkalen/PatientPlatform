"""
Synthetic simulation data generator
===================================

Writes an ``.npz`` the simulated backend can play back::

    python -m server.make_sim_data <out.npz> [--channels N] [--seconds S]
                                             [--srate HZ] [--seed K] [--force]

Why this exists: `start_sim.bat` has always passed ``sim/datafile1_filtered.npz``
and no such file has ever been in the repository, so simulation mode failed out
of the box on every clean machine (architecture audit P3). Rather than commit a
30 MB binary, the launcher now offers to generate one.

Output format - what `devices/simulated.py` requires:

    key      "data"
    shape    (n_channels, n_samples)
    dtype    float32

THIS IS NOT PHYSIOLOGICAL DATA. It is band-limited Gaussian noise with a few
raised burst envelopes: enough for the streaming path, the filters, the
recording spool and the frontend plot to have something plausible-looking to
chew on, and nothing more. In particular it has:

  * no motor-unit structure whatsoever, so decomposition will find nothing
    meaningful in it (it will run, which is the point - it must not crash);
  * no stimulation-trigger channel, so the movement-training flow, which reads
    a trigger on channel 192, has nothing to segment on. Generate more channels
    than 192 if you need that path to execute, but the trigger will be noise.
  * amplitudes in arbitrary units chosen to look like millivolt-scale surface
    EMG on the plot. Do not read anything into them.

Never use it to validate an algorithm, a threshold or a model. It exists so a
new machine can prove the plumbing works without hardware.
"""

import argparse
import os
import sys

import numpy as np

DEFAULT_CHANNELS = 64
DEFAULT_SECONDS = 60.0
DEFAULT_SRATE = 2000.0

# Surface-EMG-ish passband. Matches the server's default bandpass (20-500 Hz),
# so the filtered stream is not simply empty.
BAND_LOW_HZ = 20.0
BAND_HIGH_HZ = 450.0

# Arbitrary units, chosen to sit in a plausible millivolt-ish plot range.
REST_AMPLITUDE = 0.01
BURST_AMPLITUDE = 0.25

# Channels are generated in blocks so peak memory stays a fraction of the file:
# the FFT works in float64/complex128, i.e. ~4x the final float32 size.
CHANNEL_BLOCK = 16


def _band_limited_noise(n_channels, n_samples, srate, rng):
    """Gaussian noise with everything outside the EMG passband removed.

    FFT masking rather than an IIR filter: it keeps this module dependent on
    numpy alone, so it can be run (and tested) without scipy.
    """
    out = np.empty((n_channels, n_samples), dtype=np.float32)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / srate)
    mask = ((freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)).astype(np.float64)

    for start in range(0, n_channels, CHANNEL_BLOCK):
        stop = min(start + CHANNEL_BLOCK, n_channels)
        noise = rng.standard_normal((stop - start, n_samples))
        spectrum = np.fft.rfft(noise, axis=1) * mask
        band = np.fft.irfft(spectrum, n=n_samples, axis=1)
        # Normalise per channel so the burst envelope below sets the scale.
        rms = np.sqrt(np.mean(band ** 2, axis=1, keepdims=True))
        rms[rms == 0] = 1.0
        out[start:stop] = (band / rms).astype(np.float32)
    return out


def _burst_envelope(n_samples, srate, rng):
    """A rest/contraction envelope: quiet baseline with smoothly ramped bursts.

    Roughly 2 s of contraction every 5 s, with 200 ms cosine ramps so the
    onsets are not step discontinuities the bandpass would ring on.
    """
    envelope = np.full(n_samples, REST_AMPLITUDE, dtype=np.float64)
    period = int(5.0 * srate)
    burst = int(2.0 * srate)
    ramp = max(1, int(0.2 * srate))

    for start in range(int(1.0 * srate), n_samples, period):
        stop = min(start + burst, n_samples)
        if stop - start < 2 * ramp:
            break
        # Slight per-burst variation so every contraction is not identical.
        peak = BURST_AMPLITUDE * float(rng.uniform(0.7, 1.3))
        shape = np.full(stop - start, peak)
        ramp_up = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp)))
        shape[:ramp] = REST_AMPLITUDE + (peak - REST_AMPLITUDE) * ramp_up
        shape[-ramp:] = REST_AMPLITUDE + (peak - REST_AMPLITUDE) * ramp_up[::-1]
        envelope[start:stop] = shape
    return envelope


def make_emg_like(n_channels=DEFAULT_CHANNELS, seconds=DEFAULT_SECONDS,
                  srate=DEFAULT_SRATE, seed=0):
    """Build the synthetic ``(n_channels, n_samples)`` float32 array."""
    n_samples = int(round(seconds * srate))
    if n_channels < 1 or n_samples < 2:
        raise ValueError("need at least 1 channel and 2 samples")

    rng = np.random.default_rng(seed)
    data = _band_limited_noise(n_channels, n_samples, srate, rng)
    envelope = _burst_envelope(n_samples, srate, rng).astype(np.float32)

    # Per-channel gain so the array is not 64 copies of one amplitude, and a
    # small independent offset so channels are not perfectly correlated.
    gains = rng.uniform(0.6, 1.4, size=(n_channels, 1)).astype(np.float32)
    data *= envelope[None, :] * gains
    return data


def write_npz(path, data, srate, force=False):
    """Write the .npz atomically, so an interrupted run leaves no half file.

    That matters here: the launcher only generates when the file is MISSING, so
    a truncated leftover would be treated as valid forever after.
    """
    if os.path.exists(path) and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    tmp = os.path.abspath(path) + ".partial.npz"
    # `srate` is extra: SimulatedDevice reads only "data" and is told the rate
    # on the command line, but a file that carries its own rate is friendlier.
    np.savez(tmp, data=data, srate=np.float64(srate))
    os.replace(tmp, path)
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m server.make_sim_data",
        description="Generate a synthetic EMG-like .npz for simulation mode. "
                    "The data is NOT physiological - see the module docstring.")
    parser.add_argument("output", help="path of the .npz to write")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                        help=f"channel count (default {DEFAULT_CHANNELS})")
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                        help=f"duration in seconds (default {DEFAULT_SECONDS:g})")
    parser.add_argument("--srate", type=float, default=DEFAULT_SRATE,
                        help=f"sample rate in Hz (default {DEFAULT_SRATE:g})")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed, so a given file is reproducible")
    parser.add_argument("--force", action="store_true",
                        help="overwrite the output if it already exists")
    args = parser.parse_args(argv)

    if args.channels < 1 or args.seconds <= 0 or args.srate <= 0:
        parser.error("--channels, --seconds and --srate must all be positive")

    n_samples = int(round(args.seconds * args.srate))
    megabytes = args.channels * n_samples * 4 / 1e6
    print(f"[make_sim_data] generating {args.channels} channels x {n_samples} "
          f"samples @ {args.srate:g} Hz ({megabytes:.0f} MB)")
    print("[make_sim_data] NOTE: synthetic band-limited noise with burst "
          "envelopes - NOT physiological data")

    try:
        data = make_emg_like(args.channels, args.seconds, args.srate, args.seed)
        written = write_npz(args.output, data, args.srate, force=args.force)
    except Exception as exc:                                       # noqa: BLE001
        print(f"[make_sim_data] FAILED: {exc}", file=sys.stderr)
        return 1

    print(f"[make_sim_data] wrote {os.path.abspath(written)} "
          f"(key 'data', shape {data.shape}, dtype {data.dtype})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
