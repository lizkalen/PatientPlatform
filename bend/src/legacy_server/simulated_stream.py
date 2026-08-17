import time
import threading
import math

import pylsl

from server.devices.simulated import SimulatedDevice


CLOCK_RATE = 30_000  # Match Ripple's clock rate for time conversion


class SimulatedStream:
    """
    Streams data from a SimulatedDevice over LSL.
    Mirrors RippleStream but tailored for simulated playback.
    """

    def __init__(
        self,
        device: SimulatedDevice,
        chunk_dur: float = 0.010,
    ):
        if chunk_dur < 0.002:
            raise ValueError("Chunk duration must be at least 0.002s")
        self._device = device

        info = pylsl.StreamInfo(
            f"Simulated_{self._device.stream_type}",
            "EPhys",
            len(self._device.elec_ids),
            self._device.srate,
            "float32",
            f"SimulatedEMG{self._device.stream_type}",
        )
        info.desc().append_child_value("manufacturer", "Simulated")
        chns = info.desc().append_child("channels")
        for label in self._device.elec_ids:
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label))
        self._outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=360)
        self._expected_chunk_size = int(math.ceil(chunk_dur * self._device.srate))

        self._time_offset = 0.0
        self._time_gain = 1 / CLOCK_RATE
        self._thread = threading.Thread(target=self._time_sync_thread)
        self._thread.daemon = True
        self._thread.start()

    def _sim2lsl(self, sim_time: int) -> float:
        """Convert simulated device time to LSL time."""
        return sim_time * self._time_gain + self._time_offset

    def _time_sync_thread(self):
        """Keep simulated clock synchronized with LSL clock."""
        while True:
            lsl_now = pylsl.local_clock()
            sim_now = self._device.time()
            alpha = 0.05
            new_offset = lsl_now - sim_now * self._time_gain
            self._time_offset = alpha * new_offset + (1 - alpha) * self._time_offset
            time.sleep(5.0)

    def shutdown(self):
        del self._outlet

    def start(self, loop: bool = False):
        """Start streaming. Blocks until data is exhausted (or forever if looping)."""
        samples_pushed = 0
        samples_previous = 0
        start_time = time.time()
        loop_count = 1

        print(f"\tSamples streamed: \t\t\t\t", end="\r")

        while True:
            # Check if device has finished
            if self._device.finished:
                if loop:
                    print(f"\n\tLoop {loop_count} complete. Restarting...")
                    loop_count += 1
                    self._device.reset()
                else:
                    print(f"\n\tStreaming complete. Total samples: {samples_pushed:,}")
                    break

            n_missing = self._expected_chunk_size
            data, ts = self._device.fetch()

            if data is not None and data.size > 0:
                # Convert to list to avoid pylsl numpy memory layout issues
                self._outlet.push_chunk(data.tolist(), self._sim2lsl(ts))
                n_missing = self._expected_chunk_size - data.shape[0]
                samples_pushed += data.shape[0]

            # Print progress every second worth of samples
            if samples_pushed - samples_previous > self._device.srate:
                elapsed = time.time() - start_time
                print(f"\tSamples streamed: {samples_pushed:12,} ({elapsed:.1f}s)", end="\r")
                samples_previous = samples_pushed

            if n_missing > 0:
                sleep_time = n_missing / self._device.srate
                time.sleep(sleep_time)
