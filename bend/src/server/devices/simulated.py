import time
import typing
import pickle as pkl

import numpy as np
import numpy.typing as npt


class SimulatedDevice:
    """
    Simulates a Ripple device by playing back EMG data from an .npz file.
    Mimics the RippleDevice interface so it can be used with RippleStream.
    """


    def load_emg_data(data_path):
        """Load EMG data from a streaming results or checkpoint file.

        Args:
            data_path: Path to .pkl file containing EMG data

        Returns:
            tuple: (emg_data, srate) where emg_data is (n_channels, n_samples)
        """
        with open(data_path, "rb") as f:
            data = pkl.load(f)

        # Handle different file formats
        if isinstance(data, tuple):
            # Checkpoint format: (results, metadata) or similar
            results, metadata = data
            if "emg" in results:
                return results["emg"], metadata.get("srate", 2048)
            elif "data" in results:
                return results["data"], metadata.get("srate", 2048)
        elif isinstance(data, dict):
            # Streaming results format
            if "emg" in data:
                return data["emg"], data.get("srate", 2048)
            elif "data" in data:
                return data["data"], data.get("srate", 2048)

        raise ValueError(f"Could not find EMG data in {data_path}")

    def __init__(
        self,
        npz_path: str,
        srate: float = 2000.0,
        fetch_delay: float = 0.004,
    ):
        """
        Args:
            npz_path: Path to .npz file containing EMG data with key "data"
            srate: Sample rate of the data in Hz
            fetch_delay: Delay before fetching to avoid pulling "future" data
        """
        self._srate = srate
        self._fetch_delay = fetch_delay
        self._stream_type = "simulated"

        # # Load the data
        npz_data = np.load(npz_path)
        # # Data is stored as (channels, samples), keep it that way for now
        self._data = npz_data["data"].astype(np.float32)
        

        # # Load EMG data from .pkl file
        # self._data, file_srate = SimulatedDevice.load_emg_data(npz_path)
        # print(f"Loaded EMG data from {npz_path}")

        # self._data = self._data.astype(np.float32)

        
        # if file_srate != srate:
        #     print(
        #         f"Warning: EMG data sample rate ({file_srate}Hz) does not match specified srate ({srate}Hz)."
        #     )
        self._n_channels, self._n_samples = self._data.shape
        

        # Create fake electrode IDs (0 to n_channels-1)
        self._elec_ids = list(range(self._n_channels))

        # Playback state
        self._sample_index = 0  # Current position in the data
        self._start_time: float | None = None  # Wall-clock time when streaming started
        self._clock_ticks = 0  # Simulated device clock (like Ripple's 30kHz clock)
        self._clock_rate = 30_000  # Match Ripple's clock rate for compatibility

        self._finished = False

        print(f"Loaded {npz_path}")
        print(f"  Channels: {self._n_channels}, Samples: {self._n_samples}")
        print(f"  Duration: {self._n_samples / self._srate:.2f}s at {self._srate}Hz")

    @property
    def stream_type(self) -> str:
        return self._stream_type

    @property
    def elec_ids(self) -> list[int]:
        return self._elec_ids

    @property
    def srate(self) -> float:
        return self._srate

    @property
    def finished(self) -> bool:
        return self._finished

    def time(self) -> int:
        """Return simulated device time in clock ticks."""
        if self._start_time is None:
            return 0
        elapsed = time.perf_counter() - self._start_time
        return int(elapsed * self._clock_rate)

    def fetch(self) -> typing.Tuple[npt.NDArray[np.float32] | None, int]:
        """
        Fetch the next chunk of data based on elapsed real time.
        Returns (data, timestamp) matching RippleDevice interface.
        Data shape is (samples, channels).
        """
        if self._finished:
            return None, self._clock_ticks

        # Initialize start time on first fetch
        if self._start_time is None:
            self._start_time = time.perf_counter()

        # Calculate how many samples should have been delivered by now
        elapsed = time.perf_counter() - self._start_time
        target_sample = int((elapsed - self._fetch_delay) * self._srate)
        target_sample = max(0, min(target_sample, self._n_samples))

        # Calculate how many new samples to return
        samples_to_fetch = target_sample - self._sample_index

        if samples_to_fetch <= 0:
            return None, self._clock_ticks

        # Check if we've reached the end
        end_index = min(self._sample_index + samples_to_fetch, self._n_samples)
        actual_samples = end_index - self._sample_index

        if actual_samples <= 0:
            self._finished = True
            return None, self._clock_ticks

        # Extract chunk: _data is (channels, samples), we need (samples, channels)
        chunk = self._data[:, self._sample_index:end_index].T
        chunk = np.ascontiguousarray(chunk)

        # Update state
        self._sample_index = end_index
        self._clock_ticks = int(self._sample_index * self._clock_rate / self._srate)

        # Check if we've consumed all data
        if self._sample_index >= self._n_samples:
            self._finished = True

        return chunk, self._clock_ticks

    def reset(self):
        """Reset playback state to beginning of data."""
        self._sample_index = 0
        self._start_time = None
        self._clock_ticks = 0
        self._finished = False

    def __del__(self):
        pass  # No hardware to clean up
