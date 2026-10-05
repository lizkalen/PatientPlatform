"""
Real-time EMG viewer for the OTBioLab+ TCP stream (QUATTROCENTO).

OTBioLab+ exposes a TCP socket (default 127.0.0.1:31000) that streams the
channels of the currently loaded acquisition configuration. Unlike the
LSL/Ripple path in ``streaming.py``, this talks directly to the OTBioLab+
socket and renders with pyqtgraph for low-latency scrolling.

Protocol (from the bundled MATLAB reference
'ReadFromOTBiolabLightQuattrocento.m', confirmed empirically):
  * connect to host:port
  * send b'startTX' to begin, b'stopTX' to end
  * server replies with the 8-byte banner 'OTBioLab'
  * then streams int16, little-endian, channel-interleaved per sample:
        [c0_t0, c1_t0, ... c(N-1)_t0, c0_t1, ...]
  * N (channels) is fixed by the loaded OTBioLab+ configuration, NOT always
    the full Quattrocento frame. It is measured from the byte rate at startup:
        N = bytes_per_sec / (fsamp * 2)

Reference QUATTROCENTO config used during development (N = 200):
  0..127   IN 1-8 splitter EMG
  128..191 AD64F EMG (MULTIPLE IN 1)
  192      Ramp   (sawtooth, link sanity check)
  193      Buffer
  194..199 6x accelerometer AUX
A different OTBioLab+ configuration (channel count) is detected automatically;
a different socket is reachable via --host/--port.

EMG counts -> mV via the QUATTROCENTO factor 0.00050863 (matches the offline
OpenOTBplus reader). Ramp/aux channels are left in raw counts.

Run:
  python -m src.online.otb_realtime_plot --channels 0-7
  python otb_realtime_plot.py --ramp            # verify the link
  python otb_realtime_plot.py --channels 128-191 --offset 0.5

Deps: numpy, pyqtgraph (+ any Qt binding: PyQt6/PyQt5/PySide).
"""
import argparse
import socket
import sys
import threading
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore  # binding-agnostic (PyQt6/PyQt5/PySide)

QUATTRO_MV_PER_BIT = 0.00050863  # EMG counts -> mV (matches OpenOTBplus reader)
DEFAULT_PORT = 31000
DEFAULT_FSAMP = 2048.0


def parse_channels(spec):
    """'0-7', '0,64,128', '0-15,192' -> sorted unique list of ints."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def _read_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("stream closed")
        buf.extend(chunk)
    return bytes(buf)


def detect_nch(host, port, fsamp, probe_seconds=1.0, timeout=5.0):
    """Infer the stream's channel count from its byte rate.

    A frame is ``nch`` int16 samples per time step, so at a fixed fsamp:
        nch = bytes_per_sec / (fsamp * 2)
    Opens a short-lived connection, measures for ~probe_seconds, disconnects.

    Returns (nch_int, nch_float, bytes_per_sec). If nch_float is not close to
    an integer, fsamp is probably wrong for the current configuration.
    """
    sock = socket.create_connection((host, port), timeout=timeout)
    try:
        sock.sendall(b"startTX")
        _read_exact(sock, 8)  # 'OTBioLab' banner
        sock.settimeout(timeout)
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


class OTBStreamReader(threading.Thread):
    """Reads int16 frames from the OTBioLab+ socket into a shared ring buffer."""

    def __init__(self, host, port, nch, fsamp, samples_per_read, window_s):
        super().__init__(daemon=True)
        self.host, self.port = host, port
        self.nch = nch
        self.samples_per_read = samples_per_read
        self.buflen = int(window_s * fsamp)
        self.ring = np.zeros((self.buflen, nch), dtype=np.float64)
        self.widx = 0
        self.filled = 0
        self.lock = threading.Lock()
        self.running = threading.Event()
        self.running.set()
        self.sock = None
        self.error = None

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n and self.running.is_set():
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("stream closed by OTBioLab+")
            buf.extend(chunk)
        return bytes(buf)

    def run(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=5)
            self.sock.sendall(b"startTX")
            self._recv_exact(8)  # 'OTBioLab' banner
            frame_bytes = self.samples_per_read * self.nch * 2
            while self.running.is_set():
                raw = self._recv_exact(frame_bytes)
                block = np.frombuffer(raw, dtype="<i2").reshape(
                    self.samples_per_read, self.nch
                ).astype(np.float64)
                self._write(block)
        except Exception as e:  # surface to the GUI thread
            self.error = e
        finally:
            self._shutdown()

    def _write(self, block):
        n = block.shape[0]
        with self.lock:
            end = self.widx + n
            if end <= self.buflen:
                self.ring[self.widx:end] = block
            else:
                first = self.buflen - self.widx
                self.ring[self.widx:] = block[:first]
                self.ring[:end - self.buflen] = block[first:]
            self.widx = end % self.buflen
            self.filled = min(self.filled + n, self.buflen)

    def snapshot(self):
        """Time-ordered copy of the ring: shape (filled, nch), oldest first."""
        with self.lock:
            if self.filled < self.buflen:
                return self.ring[:self.filled].copy()
            return np.roll(self.ring, -self.widx, axis=0).copy()

    def _shutdown(self):
        if self.sock:
            try:
                self.sock.sendall(b"stopTX")
            except OSError:
                pass
            self.sock.close()

    def stop(self):
        self.running.clear()


class RealtimePlot(QtWidgets.QMainWindow):
    def __init__(self, reader, channels, fsamp, offset_mv, scale_emg, ramp_idx):
        super().__init__()
        self.reader = reader
        self.channels = channels
        self.fsamp = fsamp
        self.offset = offset_mv
        self.scale_emg = scale_emg
        self.ramp_idx = ramp_idx
        self.setWindowTitle("OTBioLab+ real-time EMG")
        self.resize(1100, 700)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "time", units="s")
        self.plot.setLabel("left", "channel (stacked)")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.setCentralWidget(self.plot)

        self.curves = []
        for i, ch in enumerate(channels):
            color = pg.intColor(i, hues=max(len(channels), 1))
            self.curves.append(self.plot.plot(pen=pg.mkPen(color=color, width=1)))
            txt = pg.TextItem(f"ch {ch}", color=color, anchor=(0, 0.5))
            txt.setPos(0, -i * self.offset)
            self.plot.addItem(txt)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(33)  # ~30 fps

    def update_plot(self):
        if self.reader.error is not None:
            self.timer.stop()
            QtWidgets.QMessageBox.critical(self, "Stream error", str(self.reader.error))
            self.close()
            return
        data = self.reader.snapshot()
        if data.shape[0] < 2:
            return
        t = np.arange(data.shape[0]) / self.fsamp
        for i, ch in enumerate(self.channels):
            y = data[:, ch]
            if ch != self.ramp_idx and self.scale_emg:
                y = y * QUATTRO_MV_PER_BIT
            self.curves[i].setData(t, y - i * self.offset)
        self.plot.setXRange(t[0], t[-1], padding=0)

    def closeEvent(self, ev):
        self.timer.stop()
        self.reader.stop()
        self.reader.join(timeout=2)
        super().closeEvent(ev)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--nch", default="auto",
                    help="channels in the stream frame; 'auto' measures it from "
                         "the byte rate (= bytes/sec / (fsamp*2)), or pass an int")
    ap.add_argument("--fsamp", type=float, default=DEFAULT_FSAMP)
    ap.add_argument("--fread", type=float, default=16.0,
                    help="reads/sec (frame size = fsamp/fread samples)")
    ap.add_argument("--window", type=float, default=3.0, help="seconds shown")
    ap.add_argument("--channels", default="0-7",
                    help="e.g. '0-7', '0,64,128', '0-15,192'")
    ap.add_argument("--ramp", action="store_true",
                    help="plot only the ramp channel (link sanity check)")
    ap.add_argument("--ramp-index", type=int, default=192)
    ap.add_argument("--offset", type=float, default=1.0,
                    help="vertical stack offset (mV, or counts with --no-scale)")
    ap.add_argument("--no-scale", action="store_true",
                    help="plot raw int16 counts instead of mV")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if str(args.nch).lower() == "auto":
        print(f"[otb] detecting channel count on {args.host}:{args.port} ...")
        try:
            nch, nch_f, rate = detect_nch(args.host, args.port, args.fsamp)
        except OSError as e:
            sys.exit(f"could not connect for channel detection: {e}")
        warn = "" if abs(nch_f - nch) < 0.15 else "  <-- not near an integer; is --fsamp correct?"
        print(f"[otb] {rate:,.0f} B/s -> {nch_f:.2f} ch @ {args.fsamp:g} Hz -> nch={nch}{warn}")
    else:
        nch = int(args.nch)

    channels = [args.ramp_index] if args.ramp else parse_channels(args.channels)
    bad = [c for c in channels if not (0 <= c < nch)]
    if bad:
        sys.exit(f"channels out of range 0..{nch - 1}: {bad}")

    samples_per_read = max(1, int(round(args.fsamp / args.fread)))
    reader = OTBStreamReader(args.host, args.port, nch, args.fsamp,
                             samples_per_read, args.window)
    reader.start()

    app = QtWidgets.QApplication(sys.argv)
    win = RealtimePlot(reader, channels, args.fsamp, args.offset,
                       scale_emg=not args.no_scale, ramp_idx=args.ramp_index)
    win.show()
    ret = app.exec() if hasattr(app, "exec") else app.exec_()
    reader.stop()
    sys.exit(ret)


if __name__ == "__main__":
    main()
