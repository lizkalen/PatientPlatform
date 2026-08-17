"""
Simulated WebSocket Server
==========================

A server that plays back EMG data from .npz files and streams it over WebSocket,
mimicking the RippleWebSocketServer for testing without hardware.

Architecture:
    .npz file --> SimulatedDevice --> This Server --> Frontend (WebSocket)
                                           |
                                           +--> (optional) LSL outlet
                                           |
                                           v
                                      Saved .pkl files

Usage:
    uv run lsl-simulate-server data.npz --port 8765 --loop
"""

import asyncio
import json
import numpy as np
import pickle as pkl
import os
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Set
import threading

import pylsl

try:
    import websockets
    from websockets.server import serve
except ImportError:
    raise ImportError("Please install websockets: pip install websockets")

from server.devices.simulated import SimulatedDevice

import sys
from pathlib import Path

from dsp.processing import design_filters, init_filter_states, apply_filters
from server.decomposition_manager import DecompositionManager
from server.movement_classifier_manager import MovementClassifierManager
from server.movement_training import MovementTrainer
from server.stimulation_client import StimulationClient
from server.stim_authority import StimAuthority, DEFAULT_MAX_TRAIN_SECONDS


CLOCK_RATE = 30_000  # Match Ripple's clock rate for time conversion


@dataclass
class DecompositionRecordingState:
    """Accumulates decomposition results during a recording session."""
    # Per-chunk firing rates: list of (n_mus,) arrays
    firing_rates: list = field(default_factory=list)
    # Per-chunk spike indices: list of {mu_idx: array} dicts
    spikes: list = field(default_factory=list)
    # Per-chunk SIL scores: list of (n_mus,) arrays
    sil_scores: list = field(default_factory=list)
    # Per-chunk source signals: list of (n_mus, n_samples) arrays
    sources: list = field(default_factory=list)
    # Classification events: list of {label, mu1_fr, mu2_fr, transition, timestamp, sample_offset}
    classification_events: list = field(default_factory=list)
    # Decomp config snapshot (captured at recording start)
    config: Optional[dict] = None
    # Running sample counter within this recording (for aligning spikes to EMG)
    sample_offset: int = 0


@dataclass
class RecordingState:
    """Tracks the current recording state."""
    is_recording: bool = False
    pre_trigger_data: Optional[np.ndarray] = None
    recorded_chunks: list = field(default_factory=list)
    start_time: Optional[datetime] = None
    metadata: Optional[dict] = None
    decomp: DecompositionRecordingState = field(default_factory=DecompositionRecordingState)


class SimulatedWebSocketServer:
    """
    WebSocket server that streams simulated EMG data from .npz files.

    Mirrors RippleWebSocketServer functionality but uses SimulatedDevice
    for playback instead of real hardware.

    Attributes:
        npz_path: Path to .npz file containing EMG data
        host: Server host address
        port: Server port
        loop: Whether to loop the data continuously
        pre_trigger_seconds: Seconds of data to keep before recording trigger
        output_folder: Where to save recordings
        enable_filtering: Apply bandpass and notch filters
        enable_lsl: Also create an LSL outlet for other consumers
    """

    def __init__(
        self,
        npz_path: str,
        host: str = "localhost",
        port: int = 8765,
        srate: float = 2000.0,
        loop: bool = False,
        pre_trigger_seconds: float = 10.0,
        output_folder: str = "./recordings",
        enable_filtering: bool = True,
        enable_lsl: bool = True,
        chunk_interval_ms: int = 50,
        filter_lowcut: float = 20.0,
        filter_highcut: float = 500.0,
        filter_notch: float = 50.0,
        stim_controller_url: Optional[str] = None,
        stim_max_seconds: float = DEFAULT_MAX_TRAIN_SECONDS,
    ):
        self.npz_path = npz_path
        self.host = host
        self.port = port
        self.srate = srate
        self.loop = loop
        self.pre_trigger_seconds = pre_trigger_seconds
        self.output_folder = output_folder
        self.enable_filtering = enable_filtering
        self.enable_lsl = enable_lsl
        self.chunk_interval_ms = chunk_interval_ms
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_notch = filter_notch

        # Will be set when device connects
        self.device: Optional[SimulatedDevice] = None
        self.sample_rate: Optional[float] = None
        self.n_channels: Optional[int] = None

        # Optional LSL outlet
        self.lsl_outlet: Optional[pylsl.StreamOutlet] = None
        self._time_offset = 0.0
        self._time_gain = 1 / CLOCK_RATE
        self._time_sync_thread: Optional[threading.Thread] = None
        self._running = False

        # Rolling buffer for pre-trigger data
        self.rolling_buffer: Optional[deque] = None

        # Recording state
        self.recording = RecordingState()

        # Connected WebSocket clients
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

        # Filter state
        self.filter_state = None
        self.bp_sos = None
        self.notch_sos = None

        # Decomposition manager (model loaded at runtime via frontend command)
        self.decomp = DecompositionManager()

        # Movement-discrimination manager (consumes the RAW chunk incl. trigger)
        self.movement = MovementClassifierManager()

        # Movement-model training state (raw capture buffer + trainer holding the
        # base model between train and calibrate).
        self.mv_trainer: MovementTrainer = None
        self._mv_raw = []
        self._mv_capturing = False
        self._mv_last_raw = None
        self._trig_monitor_ch = None      # channel index to live-plot, or None

        # Stimulation authority — the SAME class the live server uses, so stim
        # policy is not forked along with the rest of this file. Simulation mode
        # still drives the real stimulator controller if one is reachable, which
        # is exactly why it needs the same validation, ownership and deadline.
        # The controller URL is server configuration, not a message field
        # (audit S7).
        self.stim = StimAuthority(
            client=StimulationClient(
                base_url=stim_controller_url) if stim_controller_url else StimulationClient(),
            broadcast=self.broadcast,
            max_train_seconds=stim_max_seconds,
        )

        # Teardown latch (see teardown()).
        self._torn_down = False

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def _sim2lsl(self, sim_time: int) -> float:
        """Convert simulated device time to LSL time."""
        return sim_time * self._time_gain + self._time_offset

    def _time_sync_loop(self):
        """Background thread for time synchronization."""
        while self._running and self.device is not None:
            try:
                lsl_now = pylsl.local_clock()
                sim_now = self.device.time()
                alpha = 0.05
                new_offset = lsl_now - sim_now * self._time_gain
                self._time_offset = alpha * new_offset + (1 - alpha) * self._time_offset
            except Exception:
                pass
            import time
            time.sleep(5.0)

    def connect_to_device(self) -> bool:
        """Load the simulated device from .npz file."""
        print(f"Loading simulated EMG data from: {self.npz_path}")

        try:
            self.device = SimulatedDevice(npz_path=self.npz_path, srate=self.srate)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return False

        self.sample_rate = self.device.srate
        self.n_channels = len(self.device.elec_ids)

        print(f"Simulated device ready")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Loop mode: {'enabled' if self.loop else 'disabled'}")

        # Initialize rolling buffer
        buffer_samples = int(self.pre_trigger_seconds * self.sample_rate)
        self.rolling_buffer = deque(maxlen=buffer_samples)
        print(f"  Pre-trigger buffer: {self.pre_trigger_seconds}s ({buffer_samples} samples)")

        # Initialize filters
        if self.enable_filtering:
            self.bp_sos, self.notch_sos = design_filters(
                self.sample_rate,
                lowcut=self.filter_lowcut,
                highcut=self.filter_highcut,
                notch_freq=self.filter_notch
            )
            bp_zi, notch_zi = init_filter_states(
                self.bp_sos, self.notch_sos, self.n_channels
            )
            self.filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}
            print(f"  Filtering: {self.filter_lowcut}-{self.filter_highcut} Hz bandpass, {self.filter_notch} Hz notch")

        # Create LSL outlet if enabled
        if self.enable_lsl:
            info = pylsl.StreamInfo(
                "Simulated_EMG",
                "EPhys",
                self.n_channels,
                self.sample_rate,
                "float32",
                "SimulatedEMG",
            )
            info.desc().append_child_value("manufacturer", "Simulated")
            chns = info.desc().append_child("channels")
            for label in self.device.elec_ids:
                ch = chns.append_child("channel")
                ch.append_child_value("label", str(label))
            self.lsl_outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=360)
            print(f"  LSL outlet: enabled (source_id=SimulatedEMG)")

            # Seed the clock offset synchronously so the very first chunks carry
            # a plausible LSL timestamp. Without this, _time_offset stays 0.0
            # until the sync thread's first tick (up to 5s later), and liblsl
            # logs "warning: invalid timestamp" for the early push_chunk calls.
            self._time_offset = pylsl.local_clock() - self.device.time() * self._time_gain

            # Start time sync thread
            self._running = True
            self._time_sync_thread = threading.Thread(target=self._time_sync_loop, daemon=True)
            self._time_sync_thread.start()

        return True

    async def broadcast(self, message: dict):
        """Send a message to all connected WebSocket clients."""
        if not self.clients:
            return

        msg_str = json.dumps(message)
        disconnected = set()
        for client in list(self.clients):
            try:
                await client.send(msg_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        self.clients -= disconnected

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        self.clients.add(websocket)
        client_id = id(websocket)
        print(f"Client {client_id} connected. Total clients: {len(self.clients)}")

        decomp_status = self.decomp.get_status()
        # stimulation_active / recording let a reconnecting client rehydrate the
        # two pieces of state it cannot observe (audit S5/B1); mirrors
        # RippleWebSocketServer._connected_payload.
        await websocket.send(json.dumps({
            "type": "connected",
            "sample_rate": self.sample_rate,
            "n_channels": self.n_channels,
            "pre_trigger_seconds": self.pre_trigger_seconds,
            "stream_type": "simulated",
            "filtering_enabled": self.enable_filtering,
            "lsl_enabled": self.enable_lsl,
            "simulated": True,
            "loop_enabled": self.loop,
            "stimulation_active": self.stim.active,
            "recording": self.recording.is_recording,
            **decomp_status,
            **self.movement.get_status(),
        }))

        try:
            async for message in websocket:
                await self.handle_message(message, websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")
            # A departing socket must not leave its train running (audit S2).
            try:
                await self.stim.on_client_disconnect(websocket, len(self.clients))
            except Exception as e:
                print(f"[SimServer] stim stop-on-disconnect failed: {e}")

    async def handle_message(self, message: str, websocket: websockets.WebSocketServerProtocol):
        """Process incoming messages from clients."""
        # Bound before the try so the catch-all handler below can name the
        # command even when the failure happened before/at json.loads.
        command = None
        try:
            data = json.loads(message)
            command = data.get("command")

            if command == "start_recording":
                metadata = data.get("metadata")
                await self.start_recording(metadata)

            elif command == "stop_recording":
                timeline = data.get("timeline")
                await self.stop_recording(timeline)

            elif command == "load_model":
                model_path = data.get("model_path")
                if not model_path:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "load_model requires 'model_path'"
                    }))
                else:
                    try:
                        model_info = self.decomp.load_model(
                            model_path, self.sample_rate, self.n_channels
                        )
                        await self.broadcast({
                            "type": "decomposition_status",
                            "active": True,
                            **model_info,
                        })
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Failed to load model: {e}"
                        }))

            elif command == "unload_model":
                self.decomp.unload_model()
                await self.broadcast({
                    "type": "decomposition_status",
                    "active": False,
                    "n_mus": 0,
                })

            elif command == "load_movement_model":
                model_path = data.get("model_path")
                if not model_path:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": "load_movement_model requires 'model_path'"
                    }))
                else:
                    try:
                        info = self.movement.load_model(
                            model_path, self.sample_rate, self.n_channels
                        )
                        await self.broadcast({
                            "type": "movement_status",
                            "active": True,
                            **info,
                        })
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": f"Failed to load movement model: {e}"
                        }))

            elif command == "unload_movement_model":
                self.movement.unload_model()
                await self.broadcast({
                    "type": "movement_status",
                    "active": False,
                })

            # ── movement-model training flow (record raw -> train -> calibrate) ──
            elif command == "mv_record_start":
                self._mv_raw = []
                self._mv_capturing = True
                await self.broadcast({"type": "mv_train_status", "state": "recording"})

            elif command == "mv_record_stop":
                self._mv_capturing = False
                self._mv_last_raw = np.hstack(self._mv_raw) if self._mv_raw else None
                n = int(self._mv_last_raw.shape[1]) if self._mv_last_raw is not None else 0
                self._mv_raw = []
                await self.broadcast({"type": "mv_train_status", "state": "recorded",
                                      "n_samples": n})

            elif command == "mv_train":
                await self._mv_train(data, websocket)

            elif command == "mv_calibrate":
                await self._mv_calibrate(data, websocket)

            elif command == "monitor_trigger":
                ch = data.get("channel")
                self._trig_monitor_ch = int(ch) if ch is not None else None
                await self.broadcast({"type": "trigger_monitor_status",
                                      "channel": self._trig_monitor_ch})

            elif command == "start_classification":
                try:
                    info = self.decomp.start_classification(
                        mu1_idx=data.get("mu1_idx", 0),
                        mu2_idx=data.get("mu2_idx", 1),
                        threshold=data.get("threshold", 10.0),
                        window_sec=data.get("window_sec", 1.0),
                        bt_port=data.get("bt_port") or None,
                    )
                    await self.broadcast({
                        "type": "classification_status",
                        "active": True,
                        **info,
                    })
                except ValueError as e:
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))

            elif command == "stop_classification":
                self.decomp.stop_classification()
                await self.broadcast({
                    "type": "classification_status",
                    "active": False,
                })

            elif command == "stimulate_start":
                # The authority validates, binds the train to this socket, arms
                # the deadline and publishes stimulation_status itself.
                await self.stim.handle_start(data, websocket)

            elif command == "stimulate_stop":
                await self.stim.handle_stop(websocket)

            elif command == "get_status":
                decomp_status = self.decomp.get_status()
                await websocket.send(json.dumps({
                    "type": "status",
                    "connected_to_device": self.device is not None,
                    "recording": self.recording.is_recording,
                    "stimulation_active": self.stim.active,
                    "buffer_seconds": self.pre_trigger_seconds,
                    "n_clients": len(self.clients),
                    "stream_type": "simulated",
                    "lsl_enabled": self.enable_lsl,
                    "simulated": True,
                    "loop_enabled": self.loop,
                    **decomp_status,
                }))

            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Unknown command: {command}"
                }))

        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "type": "error",
                "message": "Invalid JSON message"
            }))
        except Exception as e:
            # Audit D3: this used to catch JSONDecodeError only, so any other
            # exception closed the socket with a 1011. With stop-on-disconnect
            # wired in, that close would now also stop a legitimate train and
            # take recording control down with it. Report and stay up.
            import traceback
            print(f"[SimServer] Unhandled error while processing command "
                  f"{command!r}: {e}")
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Command '{command}' failed: {e}"
                }))
            except Exception:
                pass    # the socket is already gone; nothing left to report to

    @staticmethod
    def _build_movement_timeline(classification_events, sample_rate):
        """Build a timeline of REST / MOVE periods from classification events.

        Each period dict:  {state, label, start_sample, end_sample,
                             start_sec, end_sec, duration_sec}

        label mapping:  0 -> REST, 1 -> MOVE1 (hold), 2 -> MOVE2 (hold)
        """
        LABEL_STATE = {0: "REST", 1: "MOVE1", 2: "MOVE2"}
        if not classification_events:
            return []

        timeline = []
        prev_label = classification_events[0].get("label", 0)
        prev_sample = classification_events[0].get("sample_offset", 0)

        for evt in classification_events[1:]:
            cur_label = evt.get("label", 0)
            cur_sample = evt.get("sample_offset", 0)
            if cur_label != prev_label:
                start_sec = prev_sample / sample_rate
                end_sec = cur_sample / sample_rate
                timeline.append({
                    "state": LABEL_STATE.get(prev_label, f"LABEL{prev_label}"),
                    "label": prev_label,
                    "start_sample": prev_sample,
                    "end_sample": cur_sample,
                    "start_sec": round(start_sec, 4),
                    "end_sec": round(end_sec, 4),
                    "duration_sec": round(end_sec - start_sec, 4),
                })
                prev_label = cur_label
                prev_sample = cur_sample

        # Close the last period with the final event
        last_sample = classification_events[-1].get("sample_offset", prev_sample)
        start_sec = prev_sample / sample_rate
        end_sec = last_sample / sample_rate
        timeline.append({
            "state": LABEL_STATE.get(prev_label, f"LABEL{prev_label}"),
            "label": prev_label,
            "start_sample": prev_sample,
            "end_sample": last_sample,
            "start_sec": round(start_sec, 4),
            "end_sec": round(end_sec, 4),
            "duration_sec": round(end_sec - start_sec, 4),
        })
        return timeline

    async def _mv_train(self, data, websocket):
        """Train the base movement model from the last captured raw recording."""
        if self._mv_last_raw is None:
            await websocket.send(json.dumps({
                "type": "mv_train_status", "state": "error",
                "message": "no training recording captured (mv_record_start/stop first)"}))
            return
        await self.broadcast({"type": "mv_train_status", "state": "training"})
        try:
            gl = data.get("grids")
            grids = ({f"Grid{i+1}": (int(a), int(b)) for i, (a, b) in enumerate(gl)}
                     if gl else None)
            self.mv_trainer = MovementTrainer(
                fs=self.sample_rate,
                win_ms=data.get("win_ms", 256), step_ms=data.get("step_ms", 64),
                smooth=data.get("smooth", 0.3), trig_ch=data.get("trig_ch", 192),
                clf=data.get("clf", "qda"), norm=data.get("norm", "coral"),
                pca=data.get("pca", 30), mean_shrink=data.get("mean_shrink", "js"),
                grids=grids)
            summary = await asyncio.to_thread(
                self.mv_trainer.train_base, self._mv_last_raw,
                data.get("class_sequence", []), data.get("class_order", []))
            await self.broadcast({"type": "mv_train_status", "state": "trained", **summary})
        except Exception as e:
            import traceback
            traceback.print_exc()
            await self.broadcast({"type": "mv_train_status", "state": "error",
                                  "message": str(e)})

    async def _mv_calibrate(self, data, websocket):
        """Finalize from the last captured (1-rep-each) calibration recording:
        CORAL-align, fit, freeze the artifact, and auto-load it live."""
        if self.mv_trainer is None or self.mv_trainer.base is None:
            await websocket.send(json.dumps({
                "type": "mv_train_status", "state": "error",
                "message": "train the base model before calibrating"}))
            return
        if self._mv_last_raw is None:
            await websocket.send(json.dumps({
                "type": "mv_train_status", "state": "error",
                "message": "no calibration recording captured"}))
            return
        await self.broadcast({"type": "mv_train_status", "state": "calibrating"})
        try:
            save_path = data.get("save_path") or os.path.join(
                self.output_folder, "movement_model.pkl")
            info = await asyncio.to_thread(
                self.mv_trainer.finalize, self._mv_last_raw,
                data.get("class_sequence", []), save_path)
            mv_info = self.movement.load_model(save_path, self.sample_rate, self.n_channels)
            await self.broadcast({"type": "mv_train_status", "state": "calibrated", **info})
            await self.broadcast({"type": "movement_status", "active": True, **mv_info})
        except Exception as e:
            import traceback
            traceback.print_exc()
            await self.broadcast({"type": "mv_train_status", "state": "error",
                                  "message": str(e)})

    async def start_recording(self, metadata: Optional[dict] = None):
        """Start recording, capturing the pre-trigger buffer."""
        if self.recording.is_recording:
            await self.broadcast({
                "type": "error",
                "message": "Recording already in progress"
            })
            return

        if self.rolling_buffer:
            buffer_list = list(self.rolling_buffer)
            if buffer_list:
                self.recording.pre_trigger_data = np.array(buffer_list).T
            else:
                self.recording.pre_trigger_data = None

        self.recording.is_recording = True
        self.recording.recorded_chunks = []
        self.recording.start_time = datetime.now()
        self.recording.metadata = metadata

        # Initialize decomposition recording with current config snapshot
        decomp_config = None
        if self.decomp.active:
            decomp_config = {
                "model_path": self.decomp.model_path,
                "n_mus": self.decomp.n_mus,
                "channels_to_keep": self.decomp.channels_to_keep,
                "channels_to_remove": self.decomp.channels_to_remove,
                "sample_rate": self.decomp.sample_rate,
            }
            if self.decomp.classification.active:
                decomp_config["classification"] = {
                    "mu1_idx": self.decomp.classification.mu1_idx,
                    "mu2_idx": self.decomp.classification.mu2_idx,
                    "firing_rate_threshold": self.decomp.classification.firing_rate_threshold,
                    "window_sec": self.decomp.classification.window_sec,
                }
        self.recording.decomp = DecompositionRecordingState(config=decomp_config)

        if metadata:
            print(f"Recording started with metadata: subject={metadata.get('subjectId')}, session={metadata.get('sessionId')}")

        pre_duration = 0
        if self.recording.pre_trigger_data is not None:
            pre_duration = self.recording.pre_trigger_data.shape[1] / self.sample_rate

        print(f"Recording started (pre-trigger: {pre_duration:.1f}s)"
              f"{' + decomposition' if self.decomp.active else ''}")

        await self.broadcast({
            "type": "recording_status",
            "recording": True,
            "message": f"Recording started with {pre_duration:.1f}s pre-trigger data"
        })

    async def stop_recording(self, timeline: Optional[dict] = None):
        """Stop recording and save the data.

        Args:
            timeline: Optional session timeline with phase events from frontend
        """
        if not self.recording.is_recording:
            await self.broadcast({
                "type": "error",
                "message": "No recording in progress"
            })
            return

        self.recording.is_recording = False

        all_chunks = []
        if self.recording.pre_trigger_data is not None:
            all_chunks.append(self.recording.pre_trigger_data)

        if self.recording.recorded_chunks:
            recorded_data = np.hstack(self.recording.recorded_chunks)
            all_chunks.append(recorded_data)

        if not all_chunks:
            await self.broadcast({
                "type": "error",
                "message": "No data was recorded"
            })
            return

        full_recording = np.hstack(all_chunks)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        has_classif = bool(self.recording.decomp.classification_events)
        suffix = "_classif" if has_classif else ""
        filename = f"emg_recording_{timestamp}{suffix}.pkl"
        filepath = os.path.join(self.output_folder, filename)

        recording_data = {
            "data": full_recording,
            "srate": self.sample_rate,
            "n_channels": self.n_channels,
            "filtered": self.enable_filtering,
            "pre_trigger_seconds": self.pre_trigger_seconds,
            "timestamp": timestamp,
            "stream_type": "simulated",
            "trial_metadata": self.recording.metadata,
            "session_timeline": timeline,
        }

        # Add decomposition data if any was recorded
        dr = self.recording.decomp
        if dr.firing_rates:
            # Build movement timeline: list of {state, label, start_sec, end_sec, duration_sec}
            movement_timeline = self._build_movement_timeline(dr.classification_events, self.sample_rate)

            recording_data["decomposition"] = {
                "config": dr.config,
                "firing_rates": np.array(dr.firing_rates),       # (n_chunks, n_mus)
                "sil_scores": np.array(dr.sil_scores),           # (n_chunks, n_mus)
                "sources": dr.sources,                            # list of (n_mus, n_samples) arrays
                "spikes": dr.spikes,                              # list of {mu_idx: array} dicts
                "classification_events": dr.classification_events, # list of event dicts
                "movement_timeline": movement_timeline,           # REST/HOLD periods
                "n_chunks": len(dr.firing_rates),
            }
            n_class_events = len(dr.classification_events)
            print(f"  Decomposition: {len(dr.firing_rates)} chunks, "
                  f"{n_class_events} classification events, "
                  f"{len(movement_timeline)} movement periods")
            for period in movement_timeline:
                print(f"    {period['state']:>5s} (label {period['label']}) "
                      f"{period['start_sec']:.2f}s - {period['end_sec']:.2f}s "
                      f"({period['duration_sec']:.2f}s)")

        abs_filepath = os.path.abspath(filepath)
        with open(filepath, "wb") as f:
            pkl.dump(recording_data, f)

        duration = full_recording.shape[1] / self.sample_rate
        print(f"\n  Recording saved to: {abs_filepath}")
        print(f"  Duration: {duration:.1f}s, Channels: {self.n_channels}")

        await self.broadcast({
            "type": "recording_status",
            "recording": False,
            "message": "Recording stopped and saved"
        })

        await self.broadcast({
            "type": "recording_saved",
            "filepath": filepath,
            "duration_seconds": duration,
            "n_channels": self.n_channels,
            "n_samples": full_recording.shape[1]
        })

        self.recording.pre_trigger_data = None
        self.recording.recorded_chunks = []
        self.recording.start_time = None
        self.recording.metadata = None
        self.recording.decomp = DecompositionRecordingState()

    async def stream_data(self):
        """Continuously read from simulated device and broadcast to clients."""
        if self.device is None:
            print("Cannot stream: device not loaded")
            return

        print(f"Starting data stream (chunk interval: {self.chunk_interval_ms}ms)")
        loop_count = 1

        while True:
            # Check if device has finished playing back
            if self.device.finished:
                if self.loop:
                    print(f"\nLoop {loop_count} complete. Restarting...")
                    loop_count += 1
                    self.device.reset()
                    # Reset filter state for clean restart
                    if self.enable_filtering:
                        bp_zi, notch_zi = init_filter_states(
                            self.bp_sos, self.notch_sos, self.n_channels
                        )
                        self.filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}
                else:
                    print("\nSimulated data playback complete.")
                    await self.broadcast({
                        "type": "stream_ended",
                        "message": "Simulated data playback complete"
                    })
                    break

            # Fetch data from simulated device
            data, ts = await asyncio.to_thread(self.device.fetch)

            if data is not None and data.size > 0:
                # data from device is [samples, channels], we need [channels, samples]
                samples = data.T
                n_new = samples.shape[1]

                # RAW chunk (pre-filter, trigger intact) for the movement engine.
                raw_samples = samples

                # Accumulate raw chunks while capturing a movement-training recording.
                if self._mv_capturing:
                    self._mv_raw.append(raw_samples.copy())

                # Live trigger monitor: broadcast the selected channel (downsampled).
                tmc = self._trig_monitor_ch
                if tmc is not None and 0 <= tmc < raw_samples.shape[0]:
                    row = raw_samples[tmc]
                    step = max(1, row.shape[0] // 120)
                    await self.broadcast({"type": "trigger_chunk", "channel": tmc,
                                          "data": row[::step].astype(float).tolist(),
                                          "n_samples": int(row.shape[0])})

                # Push to LSL outlet (raw, unfiltered data)
                if self.lsl_outlet is not None:
                    self.lsl_outlet.push_chunk(data.tolist(), self._sim2lsl(ts))

                # Apply filters if enabled
                if self.enable_filtering and self.filter_state:
                    samples, self.filter_state['bp_zi'], self.filter_state['notch_zi'] = apply_filters(
                        samples, self.bp_sos, self.notch_sos,
                        self.filter_state['bp_zi'], self.filter_state['notch_zi']
                    )

                # Update rolling buffer
                for i in range(n_new):
                    self.rolling_buffer.append(samples[:, i])

                # If recording, store in recorded chunks
                if self.recording.is_recording:
                    self.recording.recorded_chunks.append(samples.copy())

                # Broadcast EMG to WebSocket clients
                timestamp = self._sim2lsl(ts) if ts else None
                await self.broadcast({
                    "type": "emg_chunk",
                    "data": samples.tolist(),
                    "timestamp": timestamp,
                    "n_samples": n_new
                })

                # Run decomposition if a model is loaded
                if self.decomp.active:
                    try:
                        decomp_result = await asyncio.to_thread(
                            self.decomp.process, samples
                        )
                        if decomp_result is not None:
                            msg = self.decomp.result_to_json(decomp_result)
                            msg["timestamp"] = timestamp
                            await self.broadcast(msg)

                            # Accumulate decomp results if recording
                            if self.recording.is_recording:
                                dr = self.recording.decomp
                                dr.firing_rates.append(decomp_result["firing_rates"].copy())
                                dr.sil_scores.append(decomp_result["sil_scores"].copy())
                                dr.sources.append(decomp_result["sources"].copy())
                                # Store spikes with absolute sample offset
                                chunk_spikes = {}
                                for mu_idx, sp in decomp_result["spikes"].items():
                                    arr = np.asarray(sp, dtype=int)
                                    chunk_spikes[mu_idx] = arr + dr.sample_offset if arr.size else arr
                                dr.spikes.append(chunk_spikes)
                                # Record classification events with timestamps
                                if "classification" in decomp_result:
                                    cl = decomp_result["classification"]
                                    dr.classification_events.append({
                                        "label": cl["label"],
                                        "mu1_firing_rate": cl["mu1_firing_rate"],
                                        "mu2_firing_rate": cl["mu2_firing_rate"],
                                        "transition": cl["transition"],
                                        "timestamp": timestamp,
                                        "sample_offset": dr.sample_offset,
                                    })
                                dr.sample_offset += decomp_result["n_samples"]
                    except Exception as e:
                        print(f"[DecompSim] Decomposition error: {e}")

                # Movement classification on the RAW chunk (0, 1 or several window
                # decisions per chunk; the latest is broadcast to the frontend).
                if self.movement.active:
                    try:
                        mv_result = await asyncio.to_thread(
                            self.movement.process, raw_samples
                        )
                        mv_msg = self.movement.result_to_json(mv_result)
                        if mv_msg is not None:
                            mv_msg["timestamp"] = timestamp
                            await self.broadcast(mv_msg)
                    except Exception as e:
                        print(f"[MovementSim] Movement classification error: {e}")

            await asyncio.sleep(self.chunk_interval_ms / 1000.0)

    def shutdown(self):
        """Synchronous resource release: stop the loops, drop the device and LSL.

        Kept under this name for any caller that cannot await, and reused as the
        last step of `teardown()`. Not a safe shutdown on its own: a synchronous
        function structurally cannot await the stimulator stop or the recording
        flush (audit S2/S8.4).
        """
        self._running = False
        if self.device is not None:
            del self.device
            self.device = None
        if self.lsl_outlet is not None:
            del self.lsl_outlet
            self.lsl_outlet = None

    async def teardown(self):
        """Async teardown, awaited from run()'s finally. Idempotent.

        Mirrors RippleWebSocketServer.teardown() minus the steps this fork has
        no equivalent for (no online-run writer, no movement-recording save
        path). Order: stimulation off first, then the RAM-only recording, then
        the exo COM port, then the device.
        """
        if self._torn_down:
            return
        self._torn_down = True
        print("\n[Teardown] shutting down...")
        self._running = False

        try:
            await self.stim.shutdown()
        except Exception as e:
            print(f"[Teardown] stimulation stop FAILED: {e} — check the stimulator")

        await self._flush_recording_for_teardown()

        try:
            self.decomp.stop_classification()       # closes the exo COM port
        except Exception as e:
            print(f"[Teardown] stop_classification failed: {e}")

        try:
            await asyncio.to_thread(self.shutdown)
        except Exception as e:
            print(f"[Teardown] device/LSL cleanup failed: {e}")

        print("[Teardown] done.")

    async def _flush_recording_for_teardown(self):
        """Save whatever a running recording captured, instead of dropping it.

        Also covers the sim-only path where playback simply ends (audit P5: the
        sim server used to drop an active recording wordlessly), because that
        exits stream_data() and lands in run()'s finally.
        """
        if not self.recording.is_recording:
            return
        print("[Teardown] a recording is still running — flushing it to disk")
        try:
            await self.stop_recording(None)
            return
        except Exception as e:
            import traceback
            print(f"[Teardown] normal recording save FAILED ({e}); "
                  f"falling back to an emergency dump")
            traceback.print_exc()
        try:
            await asyncio.to_thread(self._emergency_dump_recording)
        except Exception as e:
            print(f"[Teardown] EMERGENCY DUMP FAILED: {e} — the recording is lost")

    def _emergency_dump_recording(self):
        """Last-resort write of an in-progress recording (blocking; use to_thread).

        Chunks are pickled as a LIST, not hstacked: hstack is a plausible cause
        of the failure that got us here and doubles peak RSS. Reassemble on load
        with np.hstack([pre_trigger_data] + recorded_chunks).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.output_folder, f"emg_recording_EMERGENCY_{ts}.pkl")
        payload = {
            "kind": "emergency_teardown_dump",
            "note": "backend shut down mid-recording; chunks are NOT concatenated: "
                    "np.hstack([pre_trigger_data] + recorded_chunks)",
            "pre_trigger_data": self.recording.pre_trigger_data,
            "recorded_chunks": self.recording.recorded_chunks,
            "srate": self.sample_rate,
            "n_channels": self.n_channels,
            "filtered": self.enable_filtering,
            "pre_trigger_seconds": self.pre_trigger_seconds,
            "timestamp": ts,
            "stream_type": "simulated",
            "trial_metadata": self.recording.metadata,
        }
        try:
            with open(path, "wb") as f:
                pkl.dump(payload, f)
        except Exception as e:
            print(f"[Teardown] emergency dump failed ({e}); retrying without metadata")
            payload["trial_metadata"] = repr(self.recording.metadata)[:2000]
            with open(path, "wb") as f:
                pkl.dump(payload, f)
        print(f"[Teardown] emergency dump written: {os.path.abspath(path)}")

    async def run(self):
        """Start the server and data streaming."""
        if not self.connect_to_device():
            print("Failed to load simulated device. Exiting.")
            return

        print(f"\nStarting WebSocket server on ws://{self.host}:{self.port}")
        print("Waiting for frontend connections...")

        try:
            # ping_interval/ping_timeout are explicit: the websockets defaults
            # (20/20) mean a hard-crashed tab is not even noticed for up to 40s,
            # and stop-on-disconnect is only as good as the disconnect detection
            # (audit S2). 5/5 bounds that at ~10s with a train active.
            async with serve(self.handle_client, self.host, self.port,
                             ping_interval=5, ping_timeout=5):
                try:
                    await self.stream_data()
                finally:
                    # Inside the serve() context, so the final stimulation and
                    # recording statuses can still reach connected clients.
                    await self.teardown()
        finally:
            # Safety net for a failure before or during bind. teardown() is
            # latched, so this is a no-op whenever the inner call already ran.
            await self.teardown()
