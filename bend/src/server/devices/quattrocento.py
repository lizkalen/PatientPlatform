"""
Quattrocento (OTBioLab+) device
===============================

Exposes the OTBioLab+ TCP stream through the same interface as ``RippleDevice``
so it can drop into ``RippleWebSocketServer`` via a device factory:

    srate · elec_ids · time() · fetch() -> (data[samples, channels], ts) · __del__

Unlike Ripple (pull-based, driven by a 30 kHz hardware clock), OTBioLab+ pushes
a raw int16 TCP stream. A background thread drains the socket into a queue and
``fetch()`` returns whatever samples arrived since the previous call.

Protocol (see also online/otb_realtime_plot.py):
  * connect to host:port, send b'startTX' (b'stopTX' to end)
  * server replies with the 8-byte banner 'OTBioLab'
  * then int16, little-endian, channel-interleaved per sample; the channel count
    is fixed by the loaded OTBioLab+ configuration (measured from the byte rate)
  * OTBioLab+ streams the montage's EMG channels first, then control/aux
    channels. EMG counts -> mV via 0.00050863 (QUATTROCENTO).

NOTE: the socket only supports startTX/stopTX. The montage and sample rate are
chosen in the OTBioLab+ GUI (with acquisition running); they cannot be set over
this socket. Pick which channels to expose with ``keep_channels``.
"""
import socket
import threading
import time
import typing
import xml.etree.ElementTree as ET
from collections import deque

import numpy as np
import numpy.typing as npt

CLOCK_RATE = 30_000  # synthesized clock ticks/s (matches Ripple for LSL time sync)
QUATTRO_MV_PER_BIT = 0.00050863  # EMG counts -> mV (matches OpenOTBplus reader)
DEFAULT_PORT = 31000
DEFAULT_FSAMP = 2048.0


def _read_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("OTBioLab+ stream closed")
        buf.extend(chunk)
    return bytes(buf)


def detect_nch(host, port, fsamp, probe_seconds=1.5, settle_seconds=0.7, timeout=5.0):
    """Infer the stream's channel count from its byte rate.

    A frame is ``nch`` int16 samples per time step, so at a fixed fsamp:
        nch = bytes_per_sec / (fsamp * 2)
    Opens a short-lived connection, drains an initial settle window, then
    measures ~probe_seconds and disconnects. The settle drain matters because
    OTBioLab+ dumps any buffered backlog right after startTX, which would
    otherwise inflate the measured rate (and the channel count).
    Returns (nch_int, nch_float, bytes_per_sec).
    """
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.sendall(b"startTX")
        _read_exact(sock, 8)  # 'OTBioLab' banner
        sock.settimeout(timeout)
        # Drain the post-startTX backlog so we measure the real-time rate.
        t_settle = time.time()
        while time.time() - t_settle < settle_seconds:
            if not sock.recv(65536):
                raise ConnectionError("stream closed during channel detection")
        total, t0 = 0, time.time()
        while time.time() - t0 < probe_seconds:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("stream closed during channel detection")
            total += len(chunk)
        elapsed = time.time() - t0
    finally:
        try:
            sock.sendall(b"stopTX")
        except OSError:
            pass
        sock.close()
    rate = total / elapsed
    nch_f = rate / (fsamp * 2.0)
    return int(round(nch_f)), nch_f, rate


def parse_otb_config(path):
    """Parse an OTBioLab+ device configuration file (the extensionless XML).

    OTBioLab+ streams exactly the channels defined in the loaded configuration,
    in document order (adapters, then their channels). Control adapters
    (ID='AdapterControl' — ramp, buffer, accelerometers) come after the EMG
    adapters. So counting <Channel> elements gives the stream's channel count,
    and their order matches the interleaving on the wire.

    Returns a dict:
        nch          total channels in the stream (int)
        fsamp        SampleFrequency from the file (float)
        device_name, model
        emg_channels stream indices of EMG channels (non-control adapters)
        labels       per-channel label strings (prefix + description)
    """
    root = ET.parse(path).getroot()  # <Device ...>
    fsamp = float(root.get("SampleFrequency", 2048))
    channels_el = root.find("Channels")
    if channels_el is None:
        raise ValueError(f"No <Channels> element found in config: {path}")

    labels, emg_channels, idx = [], [], 0
    for adapter in channels_el.findall("Adapter"):
        is_control = adapter.get("ID", "") == "AdapterControl"
        for ch in adapter.findall("Channel"):
            label = f"{ch.get('Prefix', '') or ''}{ch.get('Description', '') or ''}".strip()
            labels.append(label)
            if not is_control:
                emg_channels.append(idx)
            idx += 1

    if idx == 0:
        raise ValueError(f"No <Channel> elements found in config: {path}")
    return {
        "nch": idx,
        "fsamp": fsamp,
        "device_name": root.get("Name"),
        "model": root.get("Model"),
        "emg_channels": emg_channels,
        "labels": labels,
    }


class QuattrocentoDevice:
    """OTBioLab+ TCP stream as a Ripple-compatible device."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = DEFAULT_PORT,
        fsamp: float = DEFAULT_FSAMP,
        nch: typing.Optional[int] = None,
        keep_channels: typing.Optional[typing.Sequence[int]] = None,
        scale_to_mv: bool = True,
        fread: float = 16.0,
        clock_rate: float = CLOCK_RATE,
        connect_timeout: float = 5.0,
        auto_detect: bool = False,
        drain_startup_seconds: float = 0.5,
    ):
        self._host, self._port = host, port
        self._srate = float(fsamp)
        self._scale = QUATTRO_MV_PER_BIT if scale_to_mv else 1.0
        self._clock_rate = clock_rate
        self._stream_type = "quattrocento"
        self._connect_timeout = connect_timeout
        self._drain_startup_seconds = drain_startup_seconds

        # Channel count is fixed by the loaded OTBioLab+ config. Prefer passing it
        # explicitly (e.g. parsed from the config file via parse_otb_config).
        # Byte-rate auto-detection is opt-in because OTBioLab+ streams in bursts,
        # which makes short-window rate measurements unreliable.
        if nch is None:
            if not auto_detect:
                raise ValueError(
                    "nch is required. Pass it explicitly (e.g. from an OTBioLab+ "
                    "config file via parse_otb_config), or set auto_detect=True."
                )
            nch, nch_f, rate = detect_nch(host, port, fsamp)
            if abs(nch_f - nch) > 0.15:
                raise RuntimeError(
                    f"Detected {nch_f:.2f} channels at {fsamp:g} Hz (not near an "
                    f"integer) - is fsamp correct for the loaded configuration?"
                )
            print(f"[Quattrocento] auto-detected {nch} channels "
                  f"({rate:,.0f} B/s @ {fsamp:g} Hz)")
        self._nch = int(nch)

        # Which channels to expose downstream (default: all).
        if keep_channels is None:
            keep_channels = list(range(self._nch))
        bad = [c for c in keep_channels if not (0 <= c < self._nch)]
        if bad:
            raise ValueError(f"keep_channels out of range 0..{self._nch - 1}: {bad}")
        self._keep = list(keep_channels)
        self._elec_ids = list(range(len(self._keep)))

        self._samples_per_read = max(1, int(round(fsamp / fread)))

        # Streaming state (filled by the reader thread).
        self._blocks: deque = deque()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        self._sock: typing.Optional[socket.socket] = None
        self._error: typing.Optional[Exception] = None
        self._start_time: typing.Optional[float] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- RippleDevice contract ------------------------------------------------
    @property
    def stream_type(self) -> str:
        return self._stream_type

    @property
    def elec_ids(self) -> list:
        return self._elec_ids

    @property
    def srate(self) -> float:
        return self._srate

    def time(self) -> int:
        """Synthesized device clock in ticks (wall-clock based, like the sim)."""
        if self._start_time is None:
            return 0
        return int((time.perf_counter() - self._start_time) * self._clock_rate)

    def fetch(self) -> typing.Tuple[typing.Optional[npt.NDArray[np.float32]], int]:
        """Return samples accumulated since the last call, shape (samples, channels).

        Returns (None, ts) when no new data is available. Raises if the reader
        thread hit a fatal error (so the server's reconnect logic kicks in).
        """
        if self._error is not None:
            raise self._error
        with self._lock:
            if not self._blocks:
                return None, self.time()
            blocks = list(self._blocks)
            self._blocks.clear()
        data = np.vstack(blocks)                 # (samples, nch_all)
        data = data[:, self._keep] * self._scale  # (samples, n_keep), mV
        return np.ascontiguousarray(data.astype(np.float32)), self.time()

    def __del__(self):
        self.stop()

    # -- lifecycle ------------------------------------------------------------
    def stop(self):
        # Guard with getattr: __init__ may raise (e.g. detect_nch fails when
        # acquisition is off) before _running exists, and __del__ still calls this.
        running = getattr(self, "_running", None)
        if running is not None:
            running.clear()

    def _run(self):
        try:
            self._sock = socket.create_connection(
                (self._host, self._port), timeout=self._connect_timeout
            )
            self._sock.settimeout(self._connect_timeout)
            self._sock.sendall(b"startTX")
            _read_exact(self._sock, 8)  # 'OTBioLab' banner
            self._start_time = time.perf_counter()

            self._sock.settimeout(1.0)  # so stop() stays responsive
            row_bytes = self._nch * 2  # one time-sample across all channels

            # Discard the backlog OTBioLab+ dumps right after startTX so the
            # first fetch reflects near-real-time data (avoids a huge initial
            # chunk). Keep only a partial-row tail to preserve channel alignment.
            buf = bytearray()
            if self._drain_startup_seconds > 0:
                drained = bytearray()
                t_drain = time.time()
                while time.time() - t_drain < self._drain_startup_seconds:
                    try:
                        chunk = self._sock.recv(65536)
                    except socket.timeout:
                        break
                    if not chunk:
                        raise ConnectionError("OTBioLab+ stream closed")
                    drained.extend(chunk)
                tail = len(drained) % row_bytes
                if tail:
                    buf = bytearray(drained[-tail:])

            frame_bytes = self._samples_per_read * self._nch * 2
            while self._running.is_set():
                try:
                    chunk = self._sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    raise ConnectionError("OTBioLab+ stream closed")
                buf.extend(chunk)
                while len(buf) >= frame_bytes:
                    frame = bytes(buf[:frame_bytes])
                    del buf[:frame_bytes]
                    block = np.frombuffer(frame, dtype="<i2").reshape(
                        self._samples_per_read, self._nch
                    ).astype(np.float64)
                    with self._lock:
                        self._blocks.append(block)
        except Exception as e:  # surfaced to the server via fetch()
            self._error = e
        finally:
            if self._sock is not None:
                try:
                    self._sock.sendall(b"stopTX")
                except OSError:
                    pass
                self._sock.close()
