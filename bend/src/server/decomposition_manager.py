"""
Decomposition Manager
=====================

Encapsulates real-time MU decomposition and classification state.
Loads a pretrained model, processes EMG chunks into source signals,
spike times, SIL scores, and firing rates. Optionally classifies
movement from a pair of MUs.

Used by RippleWebSocketServer to add decomposition to the live stream.
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

try:
    import serial as _serial
except ImportError:
    _serial = None

import sys
from pathlib import Path

from dsp.loading import load_pretrained_model
from dsp.processing import process_chunk, calculate_firing_rates


@dataclass
class ClassificationConfig:
    """Configuration for 2-MU movement classification."""
    active: bool = False
    mu1_idx: int = 0
    mu2_idx: int = 1
    firing_rate_threshold: float = 10.0
    window_sec: float = 1.0
    prev_is_active: bool = False  # tracks rest/movement transitions
    # Optional serial port for exo hand control
    bt_serial: object = None   # serial.Serial instance, or None
    bt_send_id: int = 0        # incrementing message ID for the serial protocol


class DecompositionManager:
    """
    Manages real-time MU decomposition state and processing.

    Lifecycle:
        1. Instantiate (no model loaded yet)
        2. load_model(path, sample_rate, n_channels) -> activates decomposition
        3. process(samples) -> returns DecompositionResult per chunk
        4. unload_model() -> deactivates

    The server only needs to call process() in the stream loop and
    broadcast the returned result dict.
    """

    def __init__(self):
        # Model state
        self.active: bool = False
        self.model_path: Optional[str] = None

        # Model parameters (set on load)
        self.mu_filters: Optional[np.ndarray] = None
        self.Z: Optional[np.ndarray] = None
        self.n_mus: int = 0
        self.centroids: Optional[Dict] = None
        self.norm_factors: Optional[Dict] = None
        self.channels_to_remove: List[int] = []
        self.channels_to_keep: List[int] = []

        # Processing state (maintained across chunks)
        self.prev_tail: Optional[np.ndarray] = None
        self.spike_count_history: List[List] = []
        self.total_samples: int = 0
        self.sample_rate: float = 0.0

        # Classification
        self.classification = ClassificationConfig()

    # -------------------------------------------------------------------------
    # MODEL LIFECYCLE
    # -------------------------------------------------------------------------

    def load_model(self, model_path: str, sample_rate: float, n_channels: int) -> dict:
        """
        Load a pretrained decomposition model.

        Args:
            model_path: Path to the .pkl model file
            sample_rate: Current device sample rate
            n_channels: Total number of channels from the device

        Returns:
            dict with model info for broadcasting to clients

        Raises:
            FileNotFoundError, KeyError, ValueError on invalid model
        """
        mu_filters, Z, n_mus, centroids, norm_factors, channels_to_remove = \
            load_pretrained_model(model_path, sample_rate)

        # Store model parameters
        self.mu_filters = mu_filters
        self.Z = Z
        self.n_mus = n_mus
        self.centroids = centroids
        self.norm_factors = norm_factors
        self.channels_to_remove = channels_to_remove
        self.model_path = model_path
        self.sample_rate = sample_rate

        # Compute channels to keep (exclude calibration-removed channels)
        channels_to_remove_set = set(channels_to_remove) if channels_to_remove else set()
        self.channels_to_keep = [
            ch for ch in range(n_channels) if ch not in channels_to_remove_set
        ]

        # Initialize processing state
        self.prev_tail = None
        self.spike_count_history = [[] for _ in range(n_mus)]
        self.total_samples = 0

        # Reset classification
        self.classification = ClassificationConfig()

        self.active = True

        info = {
            "n_mus": self.n_mus,
            "channels_to_remove": self.channels_to_remove,
            "channels_to_keep": self.channels_to_keep,
            "has_centroids": centroids is not None,
            "has_norm_factors": norm_factors is not None,
            "model_path": model_path,
        }
        print(f"[DecompManager] Model loaded: {n_mus} MUs, "
              f"removing channels {channels_to_remove}, "
              f"keeping {len(self.channels_to_keep)}/{n_channels} channels")
        return info

    def unload_model(self):
        """Unload the current model and reset all state."""
        self.active = False
        self.model_path = None
        self.mu_filters = None
        self.Z = None
        self.n_mus = 0
        self.centroids = None
        self.norm_factors = None
        self.channels_to_remove = []
        self.channels_to_keep = []
        self.prev_tail = None
        self.spike_count_history = []
        self.total_samples = 0
        self.classification = ClassificationConfig()
        print("[DecompManager] Model unloaded")

    # -------------------------------------------------------------------------
    # CLASSIFICATION
    # -------------------------------------------------------------------------

    def start_classification(
        self,
        mu1_idx: int = 0,
        mu2_idx: int = 1,
        threshold: float = 10.0,
        window_sec: float = 1.0,
        bt_port: Optional[str] = None,
    ) -> dict:
        """
        Activate 2-MU movement classification.

        Classification labels:
            0 = rest (neither MU above threshold)
            1 = movement 1 (both MU1 and MU2 firing above threshold)
            2 = movement 2 (only MU2 firing above threshold)

        Args:
            mu1_idx: Index of motor unit 1
            mu2_idx: Index of motor unit 2
            threshold: Firing rate threshold in Hz
            window_sec: Sliding window duration in seconds

        Returns:
            dict with classification config for broadcasting

        Raises:
            ValueError if decomposition is not active or MU indices invalid
        """
        if not self.active:
            raise ValueError("Cannot start classification: no model loaded")
        if mu1_idx >= self.n_mus or mu2_idx >= self.n_mus:
            raise ValueError(
                f"MU indices ({mu1_idx}, {mu2_idx}) out of range for {self.n_mus} MUs"
            )

        # Open Bluetooth/serial port for exo hand control if requested
        print(f"[DecompManager] start_classification: bt_port={repr(bt_port)}, "
              f"_serial available={_serial is not None}")
        bt_serial = None
        if bt_port:
            if _serial is None:
                print("[DecompManager] PySerial not installed; ignoring bt_port")
            else:
                print(f"[DecompManager] Opening serial port {bt_port}...")
                try:
                    bt_serial = _serial.Serial(bt_port, timeout=2)
                    print(f"[DecompManager] Serial port {bt_port} opened: isOpen={bt_serial.isOpen()}")
                    # Send a heartbeat to verify the link
                    hb = json.dumps({"command": "heart_beat", "send": 0, "args": []})
                    bt_serial.write(hb.encode("utf-8"))
                    print(f"[DecompManager] Heartbeat sent to {bt_port}")
                    print(f"[DecompManager] Exo serial connected on {bt_port}")
                except Exception as exc:
                    print(f"[DecompManager] Failed to open {bt_port}: {exc}")
                    bt_serial = None
        else:
            print("[DecompManager] No bt_port provided — serial control disabled")

        self.classification = ClassificationConfig(
            active=True,
            mu1_idx=mu1_idx,
            mu2_idx=mu2_idx,
            firing_rate_threshold=threshold,
            window_sec=window_sec,
            prev_is_active=False,
            bt_serial=bt_serial,
            bt_send_id=1,
        )

        info = {
            "mu1_idx": mu1_idx,
            "mu2_idx": mu2_idx,
            "threshold": threshold,
            "window_sec": window_sec,
            "bt_port": bt_port or "",
        }
        print(f"[DecompManager] Classification started: MU{mu1_idx} & MU{mu2_idx}, "
              f"threshold={threshold} Hz, window={window_sec}s"
              + (f", exo port={bt_port}" if bt_port else ""))
        return info

    def stop_classification(self):
        """Deactivate classification."""
        if self.classification.bt_serial is not None:
            try:
                self.classification.bt_serial.close()
                print("[DecompManager] Exo serial port closed")
            except Exception as exc:
                print(f"[DecompManager] Error closing serial port: {exc}")
        self.classification = ClassificationConfig()
        print("[DecompManager] Classification stopped")

    # -------------------------------------------------------------------------
    # CHUNK PROCESSING
    # -------------------------------------------------------------------------

    def process(self, samples: np.ndarray) -> Optional[dict]:
        """
        Process a chunk of filtered EMG data through the decomposition pipeline.

        This mirrors the logic in streaming.py's update() closure:
        1. Select channels_to_keep from the full sample matrix
        2. Run process_chunk (extension, whitening, separation, spike detection)
        3. Trim overlap samples, rebase spike indices
        4. Calculate rolling firing rates
        5. Optionally classify movement

        Args:
            samples: Full EMG data from device, shape (n_channels, n_samples).
                     Already filtered by the server.

        Returns:
            dict with decomposition results, or None if decomposition is inactive
        """
        if not self.active or self.mu_filters is None:
            return None

        # Select only the channels used during calibration
        decomp_samples = samples[self.channels_to_keep, :]
        n_new = decomp_samples.shape[1]

        if n_new == 0:
            return None

        # Run decomposition (extension + whitening + separation + spike detection)
        sources, spikes, sil_scores, new_tail = process_chunk(
            decomp_samples,
            self.mu_filters,
            self.Z,
            self.n_mus,
            self.sample_rate,
            self.centroids,
            self.prev_tail,
            self.norm_factors,
            verbose=False,
        )
        self.prev_tail = new_tail

        # Trim the prepended overlap samples and rebase spike indices
        # process_chunk prepends prev_tail (R-1 samples) for continuity;
        # we need to remove those from the output
        src_len = sources.shape[1]
        trim_left = src_len - n_new
        if trim_left > 0:
            sources = sources[:, trim_left:]
            rebased_spikes = {}
            for mu_idx in range(self.n_mus):
                mu_spikes = np.asarray(spikes[mu_idx], dtype=int)
                if mu_spikes.size == 0:
                    rebased_spikes[mu_idx] = mu_spikes
                    continue
                # Keep only spikes in the new portion and adjust indices
                mu_spikes = mu_spikes[mu_spikes >= trim_left] - trim_left
                rebased_spikes[mu_idx] = mu_spikes
            spikes = rebased_spikes

        # Calculate firing rates over rolling window of 5 chunks
        chunk_duration = n_new / self.sample_rate
        spikes_dict = {i: spikes[i] for i in range(self.n_mus)}
        firing_rates, self.spike_count_history = calculate_firing_rates(
            spikes_dict, self.n_mus, chunk_duration, self.spike_count_history
        )

        # Build result dict
        result = {
            "sources": sources,               # (n_mus, n_new) ndarray
            "spikes": spikes_dict,             # {mu_idx: array of sample indices}
            "sil_scores": sil_scores,          # (n_mus,) ndarray
            "firing_rates": firing_rates,      # (n_mus,) ndarray
            "n_samples": n_new,
            "total_samples": self.total_samples,
        }

        self.total_samples += n_new

        # Classification (if active)
        if self.classification.active:
            result["classification"] = self._classify(firing_rates)

        return result

    def _classify(self, firing_rates: np.ndarray) -> dict:
        """
        Classify movement based on firing rates of two selected MUs.

        Labels:
            0 = rest (neither MU above threshold)
            1 = movement 1 (both MU1 and MU2 firing above threshold)
            2 = movement 2 (only MU2 firing above threshold)

        Returns:
            dict with classification label and MU firing rates
        """
        c = self.classification
        fr1 = float(firing_rates[c.mu1_idx])
        fr2 = float(firing_rates[c.mu2_idx])
        threshold = c.firing_rate_threshold

        mu1_active = fr1 >= threshold
        mu2_active = fr2 >= threshold

        if mu1_active and mu2_active:
            label = 1  # movement 1: both active
        elif mu2_active:
            label = 2  # movement 2: only MU2
        else:
            label = 0  # rest

        # Track state transitions
        is_active = label > 0
        transition = None
        if is_active and not c.prev_is_active:
            transition = "rest_to_active"
        elif not is_active and c.prev_is_active:
            transition = "active_to_rest"
        c.prev_is_active = is_active

        # Send trigger_open_close on any REST ↔ MOVE transition
        if transition is not None:
            print(f"[DecompManager] Transition detected: {transition} "
                  f"(bt_serial={'open' if c.bt_serial is not None else 'None'})")
            if c.bt_serial is not None:
                try:
                    msg = json.dumps({"command": "trigger_open_close", "send": c.bt_send_id, "args": []})
                    c.bt_serial.write(msg.encode("utf-8"))
                    c.bt_send_id += 1
                    print(f"[DecompManager] [BT] TX -> {msg}")
                except Exception as exc:
                    print(f"[DecompManager] [BT] Serial write error: {exc}")

        return {
            "label": label,
            "mu1_firing_rate": fr1,
            "mu2_firing_rate": fr2,
            "transition": transition,
        }

    # -------------------------------------------------------------------------
    # SERIALIZATION HELPERS
    # -------------------------------------------------------------------------

    def result_to_json(self, result: dict) -> dict:
        """
        Convert a process() result dict to JSON-serializable form.

        Converts numpy arrays to lists for WebSocket transmission.
        """
        if result is None:
            return None

        json_result = {
            "type": "decomposition_chunk",
            "sources": result["sources"].tolist(),
            "spikes": {
                str(k): v.tolist() if hasattr(v, 'tolist') else list(v)
                for k, v in result["spikes"].items()
            },
            "sil_scores": result["sil_scores"].tolist(),
            "firing_rates": result["firing_rates"].tolist(),
            "n_samples": result["n_samples"],
            "total_samples": result["total_samples"],
        }

        if "classification" in result:
            json_result["classification"] = result["classification"]

        return json_result

    def get_status(self) -> dict:
        """Return current decomposition/classification status for broadcasting."""
        status = {
            "decomposition_active": self.active,
            "n_mus": self.n_mus,
            "model_path": self.model_path,
            "channels_to_remove": self.channels_to_remove,
            "total_samples_processed": self.total_samples,
            "classification_active": self.classification.active,
        }
        if self.classification.active:
            status["classification_config"] = {
                "mu1_idx": self.classification.mu1_idx,
                "mu2_idx": self.classification.mu2_idx,
                "threshold": self.classification.firing_rate_threshold,
                "window_sec": self.classification.window_sec,
            }
        return status
