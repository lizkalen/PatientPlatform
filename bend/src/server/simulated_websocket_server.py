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
from server.recover_recording import (
    SPOOL_KIND, SPOOL_LAYOUT, SPOOL_SUFFIX, build_recording_payload, load_spool,
    safe_name, segment_paths, sidecar_path, unique_path, write_pickle_atomic,
    write_sidecar,
)


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
    """Tracks the current recording state.

    Samples are NOT held here; they go straight to an on-disk spool as they
    arrive (audit P5). Mirrors RippleWebSocketServer.RecordingState — see
    `server/recover_recording.py` for the shared spool format.
    """
    is_recording: bool = False
    start_time: Optional[datetime] = None
    metadata: Optional[dict] = None
    decomp: DecompositionRecordingState = field(default_factory=DecompositionRecordingState)
    # Monotonic id of the recording session currently occupying this state.
    # Bumped every time a recording opens; stop_recording captures it and only
    # resets state it still owns. Without it, a slow stop's delayed cleanup
    # wiped the state of a recording that started while it was finalising.
    generation: int = 0
    # Spool state
    spool_path: Optional[str] = None
    spool_file: Optional[object] = None      # open binary writer, 1 MB buffered
    spool_samples: int = 0                   # samples written, pre-trigger included
    spool_dtype: Optional[str] = None        # pinned at the first write, widened never narrowed
    spool_segments: list = field(default_factory=list)   # [{file, dtype}], see _spool_write
    pre_trigger_samples: int = 0
    spool_write_errors: int = 0
    spool_dropped_samples: int = 0
    spool_last_error: Optional[str] = None


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

        # Live references to fire-and-forget recording_warning broadcasts, so
        # the tasks are not garbage-collected mid-flight.
        self._recording_warning_tasks = set()

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
            # A recording nobody is connected to can never be stopped by anyone
            # (audit P5). The samples are already spooled, so finalising here
            # only costs the reassembly.
            if not self.clients:
                if self.recording.is_recording:
                    print("[SimServer] last client disconnected while recording — "
                          "finalising the recording")
                    try:
                        await self.stop_recording(None)
                    except Exception as e:
                        print(f"[SimServer] finalise on last disconnect failed: {e}")
                # This fork has no _save_mv_recording and no online writer, so
                # there is nothing to persist - only capture state to clear, so
                # a reconnecting client is not refused by the "already in
                # progress" guard. (The live server saves the block; playback
                # captures are reproducible by replaying the same .npz.)
                if self._mv_raw or self._mv_capturing:
                    print("[SimServer] last client disconnected mid-movement-capture "
                          "— discarding it (playback data is reproducible)")
                    self._mv_capturing = False
                    self._mv_raw = []

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
                if self._mv_capturing:
                    # Audit P5: a duplicate start used to silently discard the
                    # block already being captured. Refuse instead.
                    await websocket.send(json.dumps({
                        "type": "mv_train_status", "state": "error",
                        "message": ("a movement capture is already in progress "
                                    f"({len(self._mv_raw)} chunks); stop it with "
                                    "mv_record_stop before starting another")}))
                else:
                    self._mv_raw = []
                    self._mv_capturing = True
                    await self.broadcast({"type": "mv_train_status",
                                          "state": "recording"})

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
        except websockets.exceptions.ConnectionClosed:
            # Not a command failure: the socket dropped mid-command. Must be
            # re-raised so handle_client's dedicated handler sees it, rather
            # than logged as an alarming traceback and answered down a socket
            # that is already gone.
            raise
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

    # ── recording spool (audit P5) ───────────────────────────────────────────
    # Mirrors RippleWebSocketServer's spool helpers. Kept as a parallel copy
    # rather than shared, because collapsing this fork is its own work package;
    # the FORMAT is shared (server/recover_recording.py), so the recovery tool
    # and the file layout cannot drift even while these methods do.

    def _recording_base_name(self, metadata: Optional[dict]) -> str:
        """`emg_recording_<subject>_<session>_<start-ts>`."""
        meta = metadata or {}
        subject = safe_name(meta.get("subjectId") or meta.get("subject"), "subject")
        session = safe_name(meta.get("sessionId") or meta.get("session"), "session")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"emg_recording_{subject}_{session}_{ts}"

    def _spool_sidecar_info(self) -> dict:
        """Everything needed to interpret the spool, plus the recording metadata."""
        rec = self.recording
        info = {
            "kind": SPOOL_KIND,
            "spool_file": os.path.basename(rec.spool_path or ""),
            "n_channels": self.n_channels,
            "dtype": rec.spool_dtype,
            "segments": [dict(s) for s in rec.spool_segments],
            "layout": SPOOL_LAYOUT,
            "srate": self.sample_rate,
            "filtered": self.enable_filtering,
            "pre_trigger_seconds": self.pre_trigger_seconds,
            "pre_trigger_samples": rec.pre_trigger_samples,
            "n_samples": rec.spool_samples,
            "stream_type": "simulated",
            "trial_metadata": rec.metadata,
            "started": rec.start_time.strftime("%Y%m%d_%H%M%S") if rec.start_time else None,
        }
        if rec.spool_write_errors:
            info["spool_write_errors"] = {
                "count": rec.spool_write_errors,
                "dropped_samples": rec.spool_dropped_samples,
                "last_error": rec.spool_last_error,
            }
        return info

    def _open_spool(self, metadata: Optional[dict]) -> bool:
        """Open the spool and write its sidecar. False if the disk said no.

        The sidecar is written HERE, not at close, so a spool orphaned by a
        crash is always interpretable (audit P5 / recover_recording.py).
        """
        rec = self.recording
        path = os.path.join(self.output_folder,
                            self._recording_base_name(metadata) + SPOOL_SUFFIX)
        try:
            rec.spool_path = unique_path(path)
            rec.spool_file = open(rec.spool_path, "wb", buffering=1 << 20)
            rec.spool_samples = 0
            rec.spool_dtype = None
            rec.spool_segments = [{"file": os.path.basename(rec.spool_path),
                                   "dtype": None}]
            rec.generation += 1          # this state now belongs to a new recording
            write_sidecar(rec.spool_path, self._spool_sidecar_info())
            print(f"  Spool: {os.path.basename(rec.spool_path)} (incremental, "
                  f"crash-safe)")
            return True
        except Exception as e:
            # Refusing is the honest answer. The alternative - falling back to
            # the old in-RAM list - re-arms exactly the failure this replaces,
            # and would do it silently at the moment the disk is already sick.
            print(f"[Recording] could not open the recording spool: {e}")
            try:
                if rec.spool_file is not None:
                    rec.spool_file.close()
            except Exception:
                pass
            rec.spool_file = None
            rec.spool_path = None
            return False

    def _spool_write(self, samples) -> bool:
        """Append one channels-major chunk to the spool. Never raises.

        A write failure must not kill the acquisition loop, so it is counted and
        reported rather than propagated - and it is never hidden: the first one
        broadcasts a `recording_warning` (see _notify_recording_warning),
        rewrites the sidecar so the gap survives a crash, and the running total
        ends up in the saved file.

        The sample dtype is pinned at the first chunk. If a later chunk needs a
        WIDER one, the stream continues into a new segment rather than being
        cast down: the old in-RAM path ended in an hstack, which PROMOTED, so
        narrowing here would silently truncate the recording.
        """
        rec = self.recording
        writer = rec.spool_file
        if writer is None:
            return False
        try:
            if rec.spool_dtype is None:
                # The dtype is only knowable once a chunk exists (filtering
                # promotes to float64, raw streams do not). Pin it and rewrite
                # the sidecar so an orphan from here on is interpretable.
                rec.spool_dtype = str(samples.dtype)
                if rec.spool_segments:
                    rec.spool_segments[-1]["dtype"] = rec.spool_dtype
                write_sidecar(rec.spool_path, self._spool_sidecar_info())
            elif str(samples.dtype) != rec.spool_dtype:
                promoted = np.promote_types(np.dtype(rec.spool_dtype), samples.dtype)
                if promoted != np.dtype(rec.spool_dtype):
                    self._start_spool_segment(promoted)
                    writer = rec.spool_file
            writer.write(np.ascontiguousarray(samples.T, dtype=rec.spool_dtype).tobytes())
            # Flush (not fsync) per chunk: a flushed write survives process
            # death, which is the failure the spool exists for. See the live
            # server's _spool_write for the full reasoning.
            writer.flush()
            rec.spool_samples += int(samples.shape[1])
            return True
        except Exception as e:
            rec.spool_write_errors += 1
            rec.spool_dropped_samples += int(getattr(samples, "shape", (0, 0))[1] or 0)
            rec.spool_last_error = str(e)
            if rec.spool_write_errors == 1:
                print(f"[Recording] *** SPOOL WRITE FAILED: {e} — the recording "
                      f"now has a gap. Further failures are counted and reported "
                      f"with the saved file. ***")
                # Persist the disclosure immediately: the sidecar is otherwise
                # only rewritten at close, so a crash after an error would
                # recover a file that looks complete.
                try:
                    write_sidecar(rec.spool_path, self._spool_sidecar_info())
                except Exception:
                    pass
                self._notify_recording_warning(
                    f"Recording data is being LOST: a spool write failed ({e}). "
                    f"The recording now has a gap.")
            return False

    def _start_spool_segment(self, dtype):
        """Continue the recording into a new segment carrying a wider dtype.

        Called only when a chunk arrives that cannot be stored in the pinned
        dtype without loss. The alternative - rewriting the gigabytes already
        spooled - is slow and, worse, not crash-safe: dying part-way through
        leaves a file no sidecar can describe. Appending a segment is O(1), and
        `load_spool` concatenates segments with numpy promotion, which is
        exactly what the old hstack path did.

        Ordering matters: the segment is registered in the sidecar BEFORE its
        first byte exists, so a crash in that gap recovers every complete
        segment and merely notes the missing one.
        """
        rec = self.recording
        print(f"[Recording] sample dtype widened {rec.spool_dtype} -> {dtype}; "
              f"continuing into a new spool segment rather than narrowing")
        try:
            rec.spool_file.flush()
            rec.spool_file.close()
        except Exception as e:
            print(f"[Recording] could not close the previous spool segment: {e}")
        path = f"{rec.spool_path}.{len(rec.spool_segments)}"
        rec.spool_segments.append({"file": os.path.basename(path),
                                   "dtype": str(dtype)})
        rec.spool_dtype = str(dtype)
        write_sidecar(rec.spool_path, self._spool_sidecar_info())
        rec.spool_file = open(path, "wb", buffering=1 << 20)

    def _notify_recording_warning(self, message: str):
        """Broadcast a `recording_warning` from synchronous code.

        Message shape:
            {type: "recording_warning", recording: bool, message: str}

        A separate message type rather than a field on `recording_status`:
        that one carries the on/off flag the frontend latches on, and a warning
        must not read as a state transition. Clients that do not know the type
        drop it, which is the safe failure mode for a warning.

        Fire-and-forget, because the caller is the acquisition loop and a
        warning must never block or break it. A reference is held so the task
        cannot be garbage-collected before it runs.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return                      # no running loop (sync context)
        task = loop.create_task(self.broadcast({
            "type": "recording_warning",
            "recording": self.recording.is_recording,
            "message": message,
        }))
        self._recording_warning_tasks.add(task)
        task.add_done_callback(self._recording_warning_tasks.discard)

    def _close_spool(self):
        """Flush and close the spool. Returns (path, sidecar info) or (None, None)."""
        rec = self.recording
        writer, path = rec.spool_file, rec.spool_path
        rec.spool_file = None
        if writer is None:
            return None, None
        try:
            writer.flush()
            os.fsync(writer.fileno())
        except Exception as e:
            print(f"[Recording] could not flush the spool: {e}")
        try:
            writer.close()
        except Exception:
            pass
        info = self._spool_sidecar_info()
        try:
            write_sidecar(path, info)
        except Exception as e:
            print(f"[Recording] could not update the spool sidecar: {e}")
        return path, info

    @staticmethod
    def _discard_spool(path: Optional[str], info: Optional[dict] = None):
        """Remove a spool, every segment of it, and its sidecar.

        Only ever called once the .pkl is durably on disk. `info` is the
        sidecar dict, needed because a recording whose dtype widened mid-run
        spans several files (see _start_spool_segment).
        """
        if not path:
            return
        targets = [p for p, _ in segment_paths(path, info or {})]
        if path not in targets:
            targets.append(path)
        targets.append(sidecar_path(path))
        for target in targets:
            try:
                os.remove(target)
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[Recording] could not remove {os.path.basename(target)}: {e}")

    @staticmethod
    def _finalize_recording(spool_path, info, out_path, timeline, decomposition,
                            timestamp):
        """Reassemble the spool and write the final .pkl. Runs in a worker thread.

        Static and self-free on purpose: the caller has already detached the
        spool from `self.recording`, so a `start_recording` arriving while this
        multi-GB pickle is in flight cannot interfere with it. Doing this work
        off the event loop is also the fix for audit P6 - the synchronous
        pkl.dump used to stall the loop, and with it every queued command,
        including `stimulate_stop`.
        """
        data = load_spool(spool_path, info)
        payload = build_recording_payload(data, info, timestamp=timestamp,
                                          timeline=timeline,
                                          decomposition=decomposition)
        written = write_pickle_atomic(payload, unique_path(out_path))
        return written, int(data.shape[1])

    def _reset_recording_state(self):
        """Return to idle. Called from a `finally` on EVERY stop path.

        Audit P5: this used to be the tail of `stop_recording`, so a `pkl.dump`
        failure skipped it and left `is_recording` False but the rest of the
        state populated - and the next `start_recording` then wiped the data.

        `generation` is deliberately NOT reset: it is monotonic, and it is what
        lets a slow stop tell its own state apart from a later recording's.
        """
        rec = self.recording
        rec.start_time = None
        rec.metadata = None
        rec.decomp = DecompositionRecordingState()
        rec.spool_path = None
        rec.spool_file = None
        rec.spool_samples = 0
        rec.spool_dtype = None
        rec.spool_segments = []
        rec.pre_trigger_samples = 0
        rec.spool_write_errors = 0
        rec.spool_dropped_samples = 0
        rec.spool_last_error = None

    async def start_recording(self, metadata: Optional[dict] = None):
        """Start recording, capturing the pre-trigger buffer."""
        if self.recording.is_recording:
            await self.broadcast({
                "type": "error",
                "message": "Recording already in progress"
            })
            return

        # Pre-trigger snapshot, unchanged: taken before anything else.
        pre_trigger = None
        if self.rolling_buffer:
            buffer_list = list(self.rolling_buffer)
            if buffer_list:
                pre_trigger = np.array(buffer_list).T

        self.recording.start_time = datetime.now()
        self.recording.metadata = metadata
        self.recording.spool_write_errors = 0
        self.recording.spool_dropped_samples = 0
        self.recording.spool_last_error = None
        self.recording.pre_trigger_samples = 0

        if not self._open_spool(metadata):
            self._reset_recording_state()
            await self.broadcast({
                "type": "error",
                "message": ("Cannot start recording: the recording spool could not "
                            "be opened (check the output folder and free disk "
                            "space). Nothing was recorded."),
            })
            return

        # The pre-trigger goes in first, as part of the same stream.
        if pre_trigger is not None:
            self._spool_write(pre_trigger)
            self.recording.pre_trigger_samples = int(pre_trigger.shape[1])
        pre_trigger = None                    # released: the spool is the only copy

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
        if self.recording.pre_trigger_samples and self.sample_rate:
            pre_duration = self.recording.pre_trigger_samples / self.sample_rate

        # Set last: the stream loop spools on this flag, and the pre-trigger must
        # already be in the file before any live chunk lands behind it.
        self.recording.is_recording = True

        print(f"Recording started (pre-trigger: {pre_duration:.1f}s)"
              f"{' + decomposition' if self.decomp.active else ''}")

        await self.broadcast({
            "type": "recording_status",
            "recording": True,
            "message": f"Recording started with {pre_duration:.1f}s pre-trigger data"
        })

    def _decomposition_payload(self):
        """The `decomposition` block for the saved file, or None if none ran."""
        dr = self.recording.decomp
        if not dr.firing_rates:
            return None
        movement_timeline = self._build_movement_timeline(
            dr.classification_events, self.sample_rate)
        payload = {
            "config": dr.config,
            "firing_rates": np.array(dr.firing_rates),       # (n_chunks, n_mus)
            "sil_scores": np.array(dr.sil_scores),           # (n_chunks, n_mus)
            "sources": dr.sources,                            # list of (n_mus, n_samples) arrays
            "spikes": dr.spikes,                              # list of {mu_idx: array} dicts
            "classification_events": dr.classification_events, # list of event dicts
            "movement_timeline": movement_timeline,           # REST/HOLD periods
            "n_chunks": len(dr.firing_rates),
        }
        print(f"  Decomposition: {len(dr.firing_rates)} chunks, "
              f"{len(dr.classification_events)} classification events, "
              f"{len(movement_timeline)} movement periods")
        for period in movement_timeline:
            print(f"    {period['state']:>5s} (label {period['label']}) "
                  f"{period['start_sec']:.2f}s - {period['end_sec']:.2f}s "
                  f"({period['duration_sec']:.2f}s)")
        return payload

    async def stop_recording(self, timeline: Optional[dict] = None):
        """Stop recording, reassemble the spool and save the .pkl.

        Mirrors RippleWebSocketServer.stop_recording: reassembly happens in a
        worker thread (audit P6), and a failed save keeps the spool instead of
        destroying it (audit P5).

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
        # Everything below may await for seconds while the spool is reassembled,
        # and a new recording can legitimately start in that window. Capture the
        # generation now and only touch state that still belongs to us.
        my_generation = self.recording.generation
        spool_path, info = self._close_spool()
        write_errors = (info or {}).get("spool_write_errors")

        try:
            if not spool_path or not info or not info.get("n_samples"):
                # Still a state transition: without recording_status the
                # frontend's isRecording flag would stay latched forever.
                await self.broadcast({
                    "type": "recording_status",
                    "recording": False,
                    "message": "Recording stopped: no data was recorded"
                })
                await self.broadcast({
                    "type": "error",
                    "message": "No data was recorded"
                })
                self._discard_spool(spool_path, info)   # empty: nothing to preserve
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "_classif" if self.recording.decomp.classification_events else ""
            base = os.path.basename(spool_path)
            if base.endswith(SPOOL_SUFFIX):
                base = base[:-len(SPOOL_SUFFIX)]
            out_path = os.path.join(self.output_folder, f"{base}{suffix}.pkl")
            decomposition = self._decomposition_payload()

            try:
                filepath, n_samples = await asyncio.to_thread(
                    self._finalize_recording, spool_path, info, out_path,
                    timeline, decomposition, timestamp)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[Recording] *** SAVE FAILED: {e} — the spool is KEPT at "
                      f"{spool_path} ***")
                await self.broadcast({
                    "type": "recording_status",
                    "recording": False,
                    "message": "Recording stopped, but SAVING FAILED — the raw "
                               "data is safe in the spool file",
                })
                await self.broadcast({
                    "type": "error",
                    "message": (f"Failed to save the recording: {e}. The raw data "
                                f"is intact at {spool_path} — recover it with: "
                                f"python -m server.recover_recording "
                                f"\"{spool_path}\""),
                })
                return

            duration = n_samples / self.sample_rate if self.sample_rate else 0.0
            print(f"\n  Recording saved to: {os.path.abspath(filepath)}")
            print(f"  Duration: {duration:.1f}s, Channels: {self.n_channels}")

            if write_errors:
                print(f"  WARNING: {write_errors.get('count')} spool write "
                      f"error(s), ~{write_errors.get('dropped_samples')} samples "
                      f"missing — recorded in the saved file")

            # Durably written: only now may the spool go.
            self._discard_spool(spool_path, info)

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
                "n_samples": n_samples
            })
        finally:
            # Every exit path, including the failed save: the next
            # start_recording must find a clean slate (audit P5). But ONLY if
            # this state is still ours - a recording that started while the
            # reassembly above was in flight owns it now, and wiping its spool
            # handle mid-session silently dropped every subsequent sample.
            if self.recording.generation == my_generation:
                self._reset_recording_state()
            else:
                print(f"[Recording] not resetting state: recording generation "
                      f"{self.recording.generation} started while generation "
                      f"{my_generation} was being saved")

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

                # If recording, append straight to the on-disk spool (audit P5).
                if self.recording.is_recording:
                    self._spool_write(samples)

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
        """Finalise a running recording instead of dropping it.

        Also covers the sim-only path where playback simply ends (audit P5: the
        sim server used to drop an active recording wordlessly), because that
        exits stream_data() and lands in run()'s finally. The samples are
        already spooled, so this only reassembles them; if that fails,
        stop_recording keeps the spool and prints the recovery command.
        """
        if not self.recording.is_recording:
            return
        spool_path = self.recording.spool_path
        print("[Teardown] a recording is still running — finalising it")
        try:
            await self.stop_recording(None)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Teardown] finalising the recording FAILED ({e}). The raw data "
                  f"is intact at {spool_path} — recover it with: "
                  f"python -m server.recover_recording \"{spool_path}\"")

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
