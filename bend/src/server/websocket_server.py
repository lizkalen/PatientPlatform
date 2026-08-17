"""
Combined Ripple + WebSocket Server
==================================

A unified server that connects directly to Ripple Trellis hardware and streams
EMG data over WebSocket to a frontend, with optional LSL outlet for other consumers.

Architecture:
    Ripple Hardware --> This Server --> Frontend (WebSocket)
                            |
                            +--> (optional) LSL outlet
                            |
                            v
                       Saved .pkl files

Usage:
    uv run lsl-ripple-server --port 8765 --pre-trigger 10

This combines the functionality of:
    - lsl-ripple (Ripple device connection + LSL streaming)
    - emg_websocket_server.py (WebSocket server + recording)
"""

import asyncio
import json
import numpy as np
import pickle as pkl
import os
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional, Set, Union
import threading

import pylsl

try:
    import websockets
    from websockets.server import serve
except ImportError:
    raise ImportError("Please install websockets: pip install websockets")

# RippleDevice is imported lazily in _open_device() so this server can also run
# non-Ripple devices (e.g. Quattrocento via a device factory) on machines
# without the Ripple SDK (xipppy). CLOCK_RATE is the default synthesized clock
# rate used for LSL timestamp conversion.
CLOCK_RATE = 30_000

from dsp.processing import design_filters, init_filter_states, apply_filters
from server.decomposition_manager import DecompositionManager
from server.movement_classifier_manager import MovementClassifierManager
from server.movement_training import MovementTrainer, trigger_bouts, estimate_trig_mid
from server.stimulation_client import StimulationClient


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


class RippleWebSocketServer:
    """
    Combined server that reads directly from Ripple hardware and broadcasts
    over WebSocket, with optional LSL outlet.

    Attributes:
        host: Server host address
        port: Server port
        pre_trigger_seconds: Seconds of data to keep before recording trigger
        output_folder: Where to save recordings
        enable_filtering: Apply bandpass and notch filters
        enable_lsl: Also create an LSL outlet for other consumers
        stream_type: Ripple stream type (hi-res, raw, lfp)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        pre_trigger_seconds: float = 10.0,
        output_folder: str = "./recordings",
        enable_filtering: bool = True,
        enable_lsl: bool = True,
        broadcast_emg: bool = True,
        stream_type: str = "hi-res",
        chunk_interval_ms: int = 50,
        filter_lowcut: float = 20.0,
        filter_highcut: float = 500.0,
        filter_notch: float = 50.0,
        reconnect_interval_s: float = 2.0,
        device_factory: Optional[Callable[[], object]] = None,
        device_kind: str = "ripple",
        clock_rate: float = CLOCK_RATE,
        lsl_source_id: str = "RippleTrellis",
        lsl_name: Optional[str] = None,
        config_loader: Optional[Callable[[str], tuple]] = None,
    ):
        self.host = host
        self.port = port
        self.pre_trigger_seconds = pre_trigger_seconds
        self.output_folder = output_folder
        self.enable_filtering = enable_filtering
        self.enable_lsl = enable_lsl
        # Live EMG broadcast to the frontend. The per-chunk tolist()+json.dumps is
        # the dominant event-loop cost and, on high-channel devices (Quattrocento),
        # can stall the loop for seconds. Disabling it drops only the live plot;
        # recording, LSL, filtering and decomposition are unaffected.
        self.broadcast_emg = broadcast_emg
        self.stream_type = stream_type
        self.chunk_interval_ms = chunk_interval_ms
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_notch = filter_notch
        self.reconnect_interval_s = reconnect_interval_s

        # Device abstraction: when device_factory is None the server builds a
        # RippleDevice (default, backwards compatible); otherwise it calls the
        # factory to build any RippleDevice-compatible device (e.g. Quattrocento).
        self._device_factory = device_factory
        # Optional callback (path -> (device_factory, info dict)) that rebuilds the
        # device from a configuration file at runtime. Enables the frontend's
        # "Load OTB Config" button (set_otb_config command). None -> not supported.
        self._config_loader = config_loader
        self._device_kind = device_kind
        self._clock_rate = clock_rate
        self._lsl_source_id = lsl_source_id
        self._lsl_name = lsl_name

        # Will be set when device connects
        self.device: Optional["RippleDevice"] = None
        self.sample_rate: Optional[float] = None
        self.n_channels: Optional[int] = None

        # Whether the server's main loops should keep running. Set in run(),
        # cleared in shutdown(); used by the stream/reconnect and time-sync loops.
        self._running = False
        # Channel count the processing pipeline (buffer/filters/LSL) is sized
        # for, so we can detect a changed device on reconnect and rebuild.
        self._pipeline_n_channels: Optional[int] = None

        # Optional LSL outlet
        self.lsl_outlet: Optional[pylsl.StreamOutlet] = None
        self._time_offset = 0.0
        self._time_gain = 1 / self._clock_rate
        self._time_sync_thread: Optional[threading.Thread] = None

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

        # Movement-discrimination manager (independent of decomposition; loaded at
        # runtime via load_movement_model). Consumes the RAW chunk (does its own
        # notch/blank/bandpass) and needs the trigger channel present.
        self.movement = MovementClassifierManager()

        # Movement-model training state: a RAW-chunk capture buffer (the engine
        # trains on raw, so we accumulate raw_samples, not the filtered ones) and
        # the trainer that holds the base model between train and calibrate.
        self.mv_trainer: MovementTrainer = None
        self._mv_raw = []
        self._mv_capturing = False
        self._mv_paused = False           # capture paused (tutorial phase) — excluded from data
        self._mv_last_raw = None
        self._mv_meta = None              # metadata (label + movements) for the current block
        # Online-run capture: the guided ONLINE phase is saved separately (raw + decisions)
        # so every run is persisted with an informative name.
        self._mv_online_capturing = False
        self._mv_online_dec = []
        self._mv_online_meta = None
        self._mv_online_file = None       # open binary writer (raw streamed incrementally)
        self._mv_online_path = None
        self._mv_online_nsamp = 0
        self._trig_monitor_ch = None      # channel index to live-plot, or None

        # Stimulation client (forwards start/stop to the external stimulator
        # controller; configured per-command from the frontend).
        self.stim = StimulationClient()

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def _rpl2lsl(self, ripple_time: int) -> float:
        """Convert Ripple timestamp to LSL time."""
        return ripple_time * self._time_gain + self._time_offset

    def _time_sync_loop(self):
        """Background thread for time synchronization."""
        import time
        while self._running:
            if self.device is not None:
                try:
                    lsl_now = pylsl.local_clock()
                    rpl_now = self.device.time()
                    alpha = 0.05
                    new_offset = lsl_now - rpl_now * self._time_gain
                    self._time_offset = alpha * new_offset + (1 - alpha) * self._time_offset
                except Exception:
                    pass
            time.sleep(5.0)

    def connect_to_device(self) -> bool:
        """Connect to the acquisition device and configure the pipeline."""
        print(f"Connecting to {self._device_kind} ({self.stream_type} stream)...")
        if not self._open_device():
            return False
        self._configure_for_device()
        return True

    def _open_device(self) -> bool:
        """(Re)create the RippleDevice connection.

        Returns True on success. On failure self.device is left as None and the
        caller is expected to retry, so this never raises for a missing/absent
        processor.
        """
        try:
            if self._device_factory is not None:
                self.device = self._device_factory()
            else:
                from server.devices.ripple import RippleDevice  # lazy: requires the Ripple SDK
                self.device = RippleDevice(targ_stream_type=self.stream_type)
        except Exception as e:
            print(f"Failed to connect to {self._device_kind}: {e}")
            self.device = None
            return False

        self.sample_rate = self.device.srate
        self.n_channels = len(self.device.elec_ids)
        return True

    def _configure_for_device(self):
        """Set up (or refresh) the processing pipeline for the open device.

        Idempotent across reconnects: the rolling buffer and filters are only
        rebuilt when the channel count changes, and the LSL outlet / time-sync
        thread are created once. The clock offset is re-seeded on every call so
        the first chunks after a (re)connect carry plausible timestamps.
        """
        channels_changed = self._pipeline_n_channels != self.n_channels

        print(f"Connected to Ripple Trellis")
        print(f"  Stream type: {self.stream_type}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Channels: {self.n_channels}")

        # Initialize rolling buffer (rebuild if size changed)
        buffer_samples = int(self.pre_trigger_seconds * self.sample_rate)
        if self.rolling_buffer is None or self.rolling_buffer.maxlen != buffer_samples:
            self.rolling_buffer = deque(maxlen=buffer_samples)
            print(f"  Pre-trigger buffer: {self.pre_trigger_seconds}s ({buffer_samples} samples)")

        # Initialize filters (rebuild if not yet built or channel count changed)
        if self.enable_filtering and (self.filter_state is None or channels_changed):
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

        # Create LSL outlet if enabled (recreate if channel count changed)
        if self.enable_lsl:
            if self.lsl_outlet is None or channels_changed:
                if self.lsl_outlet is not None:
                    del self.lsl_outlet
                    self.lsl_outlet = None
                self._create_lsl_outlet()

            # Seed the clock offset synchronously so the very first chunks carry
            # a plausible LSL timestamp. Without this, _time_offset stays stale
            # until the sync thread's next tick (up to 5s later), and liblsl
            # logs "warning: invalid timestamp" for the early push_chunk calls.
            try:
                self._time_offset = pylsl.local_clock() - self.device.time() * self._time_gain
            except Exception:
                pass

            # Start time sync thread once (it survives reconnects via _running)
            if self._time_sync_thread is None or not self._time_sync_thread.is_alive():
                self._time_sync_thread = threading.Thread(target=self._time_sync_loop, daemon=True)
                self._time_sync_thread.start()

        self._pipeline_n_channels = self.n_channels

    def _create_lsl_outlet(self):
        """Create the LSL outlet for the currently connected device."""
        lsl_name = self._lsl_name or f"Ripple_{self.stream_type}"
        info = pylsl.StreamInfo(
            lsl_name,
            "EPhys",
            self.n_channels,
            self.sample_rate,
            "float32",
            self._lsl_source_id,
        )
        info.desc().append_child_value("manufacturer", self._device_kind.capitalize())
        chns = info.desc().append_child("channels")
        for label in self.device.elec_ids:
            ch = chns.append_child("channel")
            ch.append_child_value("label", str(label))
        self.lsl_outlet = pylsl.StreamOutlet(info, chunk_size=0, max_buffered=360)
        print(f"  LSL outlet: enabled (source_id={self._lsl_source_id})")

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

    def _connected_payload(self) -> dict:
        """The 'connected' handshake payload.

        Sent per-client on connect, and re-broadcast after a (re)connect so the
        frontend refreshes channel count / sample rate when the device changes
        (e.g. after a runtime config swap via set_otb_config).
        """
        return {
            "type": "connected",
            "sample_rate": self.sample_rate,
            "n_channels": self.n_channels,
            "pre_trigger_seconds": self.pre_trigger_seconds,
            "stream_type": self.stream_type,
            "filtering_enabled": self.enable_filtering,
            "lsl_enabled": self.enable_lsl,
            "live_emg_enabled": self.broadcast_emg,
            **self.decomp.get_status(),
            **self.movement.get_status(),
        }

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        self.clients.add(websocket)
        client_id = id(websocket)
        print(f"Client {client_id} connected. Total clients: {len(self.clients)}")

        await websocket.send(json.dumps(self._connected_payload()))

        try:
            async for message in websocket:
                await self.handle_message(message, websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"Client {client_id} disconnected. Total clients: {len(self.clients)}")

    async def handle_message(self, message: str, websocket: websockets.WebSocketServerProtocol):
        """Process incoming messages from clients."""
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
                self._mv_paused = False
                self._mv_meta = data          # label + class_order/sequence + montage
                await self.broadcast({"type": "mv_train_status", "state": "recording"})

            elif command == "mv_record_pause":       # tutorial phase — exclude from data
                self._mv_paused = True

            elif command == "mv_record_resume":
                self._mv_paused = False

            elif command == "mv_record_stop":
                self._mv_capturing = False
                self._mv_last_raw = np.hstack(self._mv_raw) if self._mv_raw else None
                n = int(self._mv_last_raw.shape[1]) if self._mv_last_raw is not None else 0
                self._mv_raw = []
                await self.broadcast({"type": "mv_train_status", "state": "recorded",
                                      "n_samples": n})
                # Persist the RAW block OFF the event loop — pickling ~100+MB to disk in the
                # loop was freezing streaming/stim. to_thread releases the GIL during I/O.
                if self._mv_last_raw is not None:
                    asyncio.create_task(
                        asyncio.to_thread(self._save_mv_recording, self._mv_last_raw, self._mv_meta))

            elif command == "mv_online_start":
                self._mv_online_dec = []
                self._mv_online_meta = data
                self._open_online_writer(data)          # stream raw to disk, not RAM
                self._mv_online_capturing = self._mv_online_file is not None
                await self.broadcast({"type": "mv_train_status", "state": "online_recording"})

            elif command == "mv_online_stop":
                self._mv_online_capturing = False
                dec, meta = self._mv_online_dec, self._mv_online_meta
                self._mv_online_dec = []
                self._close_online_writer(dec, meta)     # close file + tiny metadata sidecar
                await self.broadcast({"type": "mv_train_status", "state": "online_saved"})

            elif command == "mv_train":
                await self._mv_train(data, websocket)

            elif command == "mv_train_clean":
                await self._mv_train_clean(data, websocket)

            elif command == "mv_calibrate":
                await self._mv_calibrate(data, websocket)

            elif command == "mv_set_stim_binary":
                # Runtime toggle: STIM head reports full 4-class vs a move/rest gate.
                status = self.movement.set_stim_binary(bool(data.get("on", False)))
                await self.broadcast({"type": "movement_status",
                                      "active": self.movement.active, **status})

            elif command == "monitor_trigger":
                ch = data.get("channel")
                self._trig_monitor_ch = int(ch) if ch is not None else None
                await self.broadcast({"type": "trigger_monitor_status",
                                      "channel": self._trig_monitor_ch})

            elif command == "start_classification":
                try:
                    _bt_port_raw = data.get("bt_port")
                    print(f"[Server] start_classification received: mu1={data.get('mu1_idx')} "
                          f"mu2={data.get('mu2_idx')} threshold={data.get('threshold')} "
                          f"window={data.get('window_sec')} bt_port={repr(_bt_port_raw)}")
                    info = self.decomp.start_classification(
                        mu1_idx=data.get("mu1_idx", 0),
                        mu2_idx=data.get("mu2_idx", 1),
                        threshold=data.get("threshold", 10.0),
                        window_sec=data.get("window_sec", 1.0),
                        bt_port=_bt_port_raw or None,
                    )
                    await self.broadcast({
                        "type": "classification_status",
                        "active": True,
                        **info,
                    })
                except ValueError as e:
                    print(f"[Server] start_classification ValueError: {e}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": str(e)
                    }))
                except Exception as e:
                    import traceback
                    print(f"[Server] start_classification unexpected error: {e}")
                    traceback.print_exc()
                    await websocket.send(json.dumps({
                        "type": "error",
                        "message": f"Unexpected error: {e}"
                    }))

            elif command == "stop_classification":
                self.decomp.stop_classification()
                await self.broadcast({
                    "type": "classification_status",
                    "active": False,
                })

            elif command == "stimulate_start":
                result = await self.stim.start_train(
                    channels=data.get("channels") or [],
                    stimulator_type=data.get("stimulator_type"),
                    port=data.get("port"),
                    controller_url=data.get("controller_url"),
                    metadata=data.get("metadata"),
                )
                await self.broadcast({
                    "type": "stimulation_status",
                    "active": result.get("status") == "success",
                    **result,
                })

            elif command == "stimulate_stop":
                result = await self.stim.stop(
                    controller_url=data.get("controller_url"),
                )
                await self.broadcast({
                    "type": "stimulation_status",
                    "active": False,
                    **result,
                })

            elif command == "set_otb_config":
                await self.apply_device_config(data.get("path"), websocket)

            elif command == "get_status":
                decomp_status = self.decomp.get_status()
                await websocket.send(json.dumps({
                    "type": "status",
                    "connected_to_device": self.device is not None,
                    "recording": self.recording.is_recording,
                    "buffer_seconds": self.pre_trigger_seconds,
                    "n_clients": len(self.clients),
                    "stream_type": self.stream_type,
                    "lsl_enabled": self.enable_lsl,
                    **decomp_status,
                    **self.movement.get_status(),
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

    async def apply_device_config(self, path: Optional[str],
                                  websocket: websockets.WebSocketServerProtocol):
        """Reload the acquisition device from a config file at runtime.

        Wired to the frontend's "Load OTB Config" button. Only available when the
        server was constructed with a config_loader (Quattrocento). Rebuilds the
        device factory from the given config file and drops the current device so
        the stream loop reconnects with it, rebuilding the pipeline for the new
        channel count / sample rate. `path` is resolved on the server machine.
        """
        async def _err(message: str):
            await websocket.send(json.dumps({"type": "error", "message": message}))

        if self._config_loader is None:
            await _err("This server does not support runtime configuration.")
            return
        if not path:
            await _err("set_otb_config requires a 'path'.")
            return
        if self.recording.is_recording:
            await _err("Stop the current recording before changing the configuration.")
            return

        try:
            device_factory, info = await asyncio.to_thread(self._config_loader, path)
        except Exception as e:
            await _err(f"Failed to load config '{path}': {e}")
            return

        # Swap in the new factory and drop the current device; the stream loop
        # reconnects with it and rebuilds buffers/filters/LSL for the new channels.
        self._device_factory = device_factory
        await asyncio.to_thread(self._close_device)
        print(f"[Server] New OTB config applied: {info['nch']} ch @ {info['fsamp']:g} Hz "
              f"(exposing {info['n_exposed']}); reconnecting...")
        await self.broadcast({
            "type": "config_status",
            "message": (f"Loaded {info.get('device_name') or 'config'} "
                        f"({info['n_exposed']} ch @ {info['fsamp']:g} Hz) — reconnecting..."),
            **info,
        })

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

    def _spawn_mv_job(self, make_coro):
        """Run a movement train/clean/finalize job in the BACKGROUND, chained after the
        previous one so they execute in submission order (train -> clean -> finalize)
        even while the NEXT recording is being captured concurrently. This is what lets
        us train the stim model while the patient performs the no-stim block, and vice
        versa. `make_coro` is a zero-arg coroutine function (the actual job)."""
        prev = getattr(self, "_mv_prev_task", None)
        if not hasattr(self, "_mv_tasks"):
            self._mv_tasks = set()

        async def runner():
            if prev is not None:
                try:
                    await prev                    # strict ordering, regardless of scheduling
                except Exception:
                    pass
            await make_coro()

        task = asyncio.create_task(runner())
        self._mv_prev_task = task
        self._mv_tasks.add(task)
        task.add_done_callback(self._mv_tasks.discard)
        return task

    async def _mv_train(self, data, websocket):
        """Kick off STIM base-model training in the BACKGROUND so the operator can start
        the next (no-stim) recording immediately while this trains. The captured raw is
        handed to the job and the server reference dropped, so it is freed as soon as
        features are extracted (only the small feature matrix is retained)."""
        if self._mv_last_raw is None:
            await websocket.send(json.dumps({
                "type": "mv_train_status", "state": "error",
                "message": "no training recording captured (mv_record_start/stop first)"}))
            return
        raw = self._mv_last_raw
        self._mv_last_raw = None                    # hand off; free the server's ref
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
        trainer = self.mv_trainer
        cseq, corder = data.get("class_sequence", []), data.get("class_order", [])

        async def job():
            await self.broadcast({"type": "mv_train_status", "state": "training"})
            try:
                summary = await asyncio.to_thread(trainer.train_base, raw, cseq, corder)
                await self.broadcast({"type": "mv_train_status", "state": "trained", **summary})
            except Exception as e:
                import traceback
                traceback.print_exc()
                await self.broadcast({"type": "mv_train_status", "state": "error", "message": str(e)})
        self._spawn_mv_job(job)

    async def _mv_train_clean(self, data, websocket):
        """Train the CLEAN (nostim) specialist in the BACKGROUND, chained after the stim
        base. The patient gets no stim, but the stimulator still drives the trigger (ch3),
        so bouts are trigger-segmented like the stim block — only blanking is off. Frozen
        into the artifact as model_clean at calibrate time."""
        if self._mv_last_raw is None:
            await websocket.send(json.dumps({
                "type": "mv_train_status", "state": "error",
                "message": "no clean recording captured (mv_record_start/stop first)"}))
            return
        raw = self._mv_last_raw
        self._mv_last_raw = None
        cseq, corder = data.get("class_sequence", []), data.get("class_order", [])

        async def job():
            if self.mv_trainer is None or self.mv_trainer.base is None:
                await self.broadcast({"type": "mv_train_status", "state": "error",
                                      "message": "clean pass needs a trained stim base (mv_train first)"})
                return
            await self.broadcast({"type": "mv_train_status", "state": "training_clean"})
            try:
                summary = await asyncio.to_thread(self.mv_trainer.train_clean, raw, cseq, corder)
                await self.broadcast({"type": "mv_train_status", "state": "trained_clean", **summary})
            except Exception as e:
                import traceback
                traceback.print_exc()
                await self.broadcast({"type": "mv_train_status", "state": "error", "message": str(e)})
        self._spawn_mv_job(job)

    async def _mv_calibrate(self, data, websocket):
        """Finalize in the BACKGROUND: chained after the stim base AND any clean pass, so
        it only runs once both are done. CORAL-aligns, fits, freezes BOTH specialists into
        one artifact (model + model_clean), and auto-loads it live."""
        if self._mv_last_raw is None:
            await websocket.send(json.dumps({
                "type": "mv_train_status", "state": "error",
                "message": "no calibration recording captured"}))
            return
        raw = self._mv_last_raw
        self._mv_last_raw = None
        cseq = data.get("class_sequence", [])
        save_path = data.get("save_path") or os.path.join(self.output_folder, "movement_model.pkl")

        async def job():
            if self.mv_trainer is None or self.mv_trainer.base is None:
                await self.broadcast({"type": "mv_train_status", "state": "error",
                                      "message": "train the base model before calibrating"})
                return
            await self.broadcast({"type": "mv_train_status", "state": "calibrating"})
            try:
                info = await asyncio.to_thread(self.mv_trainer.finalize, raw, cseq, save_path)
                mv_info = self.movement.load_model(save_path, self.sample_rate, self.n_channels)
                await self.broadcast({"type": "mv_train_status", "state": "calibrated", **info})
                await self.broadcast({"type": "movement_status", "active": True, **mv_info})
            except Exception as e:
                import traceback
                traceback.print_exc()
                await self.broadcast({"type": "mv_train_status", "state": "error", "message": str(e)})
        self._spawn_mv_job(job)

    @staticmethod
    def _safe_name(s, default="x"):
        """Filesystem-safe token for building informative recording filenames."""
        s = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(s or default))
        return s.strip("-. ") or default

    def _save_mv_recording(self, raw, meta):
        """Persist a RAW movement-training block (trigger intact) to disk so it can be
        re-trained later, PLUS a best-effort per-movement segment file for each movement.
        The live main recording only saves FILTERED data, so this is the sole re-trainable
        copy. Returns the list of saved paths (or None). Never raises — data safety first."""
        if raw is None:
            return None
        try:
            meta = meta or {}
            label = self._safe_name(meta.get("label", "block"))
            session = self._safe_name(meta.get("session") or meta.get("sessionId") or "session")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            class_order = list(meta.get("class_order", []))
            class_seq = list(meta.get("class_sequence", []))
            trig_ch = int(meta.get("trig_ch", 192))
            base = f"mv_{session}_{label}_{ts}"
            common = dict(srate=self.sample_rate, n_channels=self.n_channels, filtered=False,
                          timestamp=ts, label=label, class_order=class_order,
                          class_sequence=class_seq, trig_ch=trig_ch,
                          grids=meta.get("grids"), metadata=meta)
            block_path = os.path.join(self.output_folder, base + ".pkl")
            with open(block_path, "wb") as f:
                pkl.dump(dict(data=raw, kind="movement_train", **common), f)
            saved = [block_path]
            # best-effort per-movement segments (trigger-segmented, grouped by class label)
            try:
                if class_order and 0 <= trig_ch < raw.shape[0]:
                    spans = trigger_bouts(raw[trig_ch], self.sample_rate,
                                          estimate_trig_mid(raw[trig_ch]))
                    per = {}
                    for i, (s, e) in enumerate(spans):
                        if i >= len(class_seq):
                            break
                        ci = int(class_seq[i])
                        lbl = class_order[ci] if 0 <= ci < len(class_order) else str(ci)
                        per.setdefault(lbl, []).append(raw[:, s:e])
                    for lbl, segs in per.items():
                        mp = os.path.join(self.output_folder,
                                          f"{base}_{self._safe_name(lbl)}.pkl")
                        with open(mp, "wb") as f:
                            pkl.dump(dict(data=np.hstack(segs), kind="movement_single",
                                          movement=lbl, n_bouts=len(segs), **common), f)
                        saved.append(mp)
            except Exception as e:
                print(f"[MovementSave] per-movement segmentation skipped: {e}")
            print(f"[MovementSave] {label}: saved {os.path.basename(block_path)} "
                  f"(+{len(saved)-1} per-movement) -> {self.output_folder}")
            return saved
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MovementSave] FAILED to save movement block: {e}")
            return None

    def _open_online_writer(self, meta):
        """Open an incremental binary writer for the online run, so the raw stream goes
        straight to disk (samples-major float32) instead of growing in RAM. The chunks
        are written in the loop; a small metadata sidecar is written at stop. Never raises."""
        if self._mv_online_file is not None:            # a run was left open — close it
            try:
                self._mv_online_file.close()
            except Exception:
                pass
            self._mv_online_file = None
        try:
            meta = meta or {}
            session = self._safe_name(meta.get("session") or meta.get("sessionId") or "session")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._mv_online_path = os.path.join(self.output_folder, f"mv_{session}_online_{ts}.raw")
            self._mv_online_file = open(self._mv_online_path, "wb", buffering=1 << 20)  # 1 MB buffer
            self._mv_online_nsamp = 0
            print(f"[MovementSave] online run -> streaming to "
                  f"{os.path.basename(self._mv_online_path)} (incremental)")
            return True
        except Exception as e:
            print(f"[MovementSave] could not open online writer: {e}")
            self._mv_online_file = self._mv_online_path = None
            return False

    def _close_online_writer(self, decisions, meta):
        """Close the incremental raw file and write the (small) metadata sidecar with the
        shape/dtype needed to load it: np.fromfile(raw, float32).reshape(-1, n_ch).T."""
        f, path, nsamp = self._mv_online_file, self._mv_online_path, self._mv_online_nsamp
        self._mv_online_file = self._mv_online_path = None
        self._mv_online_nsamp = 0
        if f is None:
            return None
        try:
            f.close()
            meta = meta or {}
            side = os.path.splitext(path)[0] + ".pkl"
            with open(side, "wb") as sf:
                pkl.dump(dict(kind="movement_online", raw_file=os.path.basename(path),
                              shape=[int(self.n_channels or 0), int(nsamp)], dtype="float32",
                              layout="samples-major: np.fromfile(raw, float32).reshape(-1, n_channels).T",
                              srate=self.sample_rate, filtered=False,
                              class_order=list(meta.get("class_order", [])),
                              model_path=self.movement.model_path, decisions=decisions,
                              metadata=meta), sf)
            dur = nsamp / self.sample_rate if self.sample_rate else 0
            print(f"[MovementSave] online run saved: {os.path.basename(path)} + sidecar "
                  f"({dur:.1f}s, {len(decisions)} decisions)")
            return path
        except Exception as e:
            print(f"[MovementSave] failed to finalize online run: {e}")
            return None

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
            "stream_type": self.stream_type,
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

    def _close_device(self):
        """Tear down the current device connection (best-effort)."""
        dev = self.device
        self.device = None
        if dev is not None:
            try:
                del dev  # triggers RippleDevice.__del__ (safe, never raises)
            except Exception:
                pass

    async def _on_device_lost(self, error: Exception):
        """Handle a streaming failure: drop the device and notify clients."""
        print(f"[Server] Lost connection to Ripple device: {error}")
        print(f"[Server] Will retry every {self.reconnect_interval_s}s...")
        # __del__ sleeps ~1s, so close off the event loop thread.
        await asyncio.to_thread(self._close_device)
        await self.broadcast({
            "type": "device_status",
            "connected": False,
            "message": "Lost connection to Ripple device, attempting to reconnect...",
        })
        await asyncio.sleep(self.reconnect_interval_s)

    async def _try_reconnect(self) -> bool:
        """Attempt to (re)open the device. Returns True once connected."""
        if not await asyncio.to_thread(self._open_device):
            return False
        self._configure_for_device()
        print("[Server] Reconnected to Ripple device")
        await self.broadcast({
            "type": "device_status",
            "connected": True,
            "message": "Reconnected to Ripple device",
        })
        # Refresh channel count / sample rate on the frontend — the device may
        # have changed (e.g. a runtime config swap swapped the device factory).
        await self.broadcast(self._connected_payload())
        return True

    async def stream_data(self):
        """Continuously read from Ripple device and broadcast to clients.

        Resilient to the device going away: if the Ripple stream is interrupted
        or was never present, the server stays up and keeps retrying the
        connection instead of crashing.
        """
        print(f"Starting data stream (chunk interval: {self.chunk_interval_ms}ms)")

        while self._running:
            # (Re)establish the device connection if needed.
            if self.device is None:
                if not await self._try_reconnect():
                    await asyncio.sleep(self.reconnect_interval_s)
                continue

            # Fetch data from Ripple device in thread to avoid blocking
            try:
                data, ts = await asyncio.to_thread(self.device.fetch)
            except Exception as e:
                await self._on_device_lost(e)
                continue

            if data is not None and data.size > 0:
                # data from device is [samples, channels], we need [channels, samples]
                samples = data.T
                n_new = samples.shape[1]

                # Keep a reference to the RAW (unfiltered) chunk for the movement
                # engine, which does its own notch/blank/bandpass and needs the
                # trigger channel intact. apply_filters returns a new array (it does
                # not mutate `samples`), so this view stays raw after filtering.
                raw_samples = samples

                # Accumulate raw chunks while capturing a movement-training recording,
                # or while capturing an online run (both need the RAW, trigger intact).
                if self._mv_capturing and not self._mv_paused:
                    self._mv_raw.append(raw_samples.copy())
                if self._mv_online_capturing and self._mv_online_file is not None:
                    try:                    # stream the raw chunk to disk (samples-major)
                        self._mv_online_file.write(
                            np.ascontiguousarray(raw_samples.T, dtype=np.float32).tobytes())
                        self._mv_online_nsamp += raw_samples.shape[1]
                    except Exception as e:
                        print(f"[MovementSave] online write error (stopping capture): {e}")
                        self._mv_online_capturing = False

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
                    self.lsl_outlet.push_chunk(data, self._rpl2lsl(ts))

                # Apply filters if enabled
                if self.enable_filtering and self.filter_state:
                    samples, self.filter_state['bp_zi'], self.filter_state['notch_zi'] = apply_filters(
                        samples, self.bp_sos, self.notch_sos,
                        self.filter_state['bp_zi'], self.filter_state['notch_zi']
                    )

                # Update rolling buffer
                for i in range(n_new):
                    self.rolling_buffer.append(samples[:, i])

                # If recording, store in recorded chunks (paused during a tutorial phase
                # so it's excluded from the saved recording too; _mv_paused is False in
                # any non-movement-session recording, so this is a no-op there).
                if self.recording.is_recording and not self._mv_paused:
                    self.recording.recorded_chunks.append(samples.copy())

                # Broadcast EMG to WebSocket clients (skippable: serializing the
                # chunk is the dominant per-chunk cost; decomposition below still
                # needs `timestamp`, so compute it either way).
                timestamp = self._rpl2lsl(ts) if ts else None
                if self.broadcast_emg:
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
                        print(f"[DecompServer] Decomposition error: {e}")

                # Run movement classification if a model is loaded. Feed the RAW
                # chunk (pre-filter, trigger included); the engine emits 0, 1 or
                # several window decisions per chunk and returns the latest.
                if self.movement.active:
                    try:
                        mv_result = await asyncio.to_thread(
                            self.movement.process, raw_samples
                        )
                        mv_msg = self.movement.result_to_json(mv_result)
                        if mv_msg is not None:
                            mv_msg["timestamp"] = timestamp
                            await self.broadcast(mv_msg)
                            if self._mv_online_capturing:   # log the decision stream
                                self._mv_online_dec.append(
                                    {"pred": mv_result.get("pred"),
                                     "label": mv_result.get("label"),
                                     "total_samples": mv_result.get("total_samples"),
                                     "timestamp": timestamp})
                        # Periodic ROUTING + TRIGGER diagnostic (~1s): is the silent
                        # trigger sitting above trig_mid (routing clean/initiator) or
                        # crossing it (spurious pulses -> misrouting to the stim model)?
                        self._mv_diag_ctr = getattr(self, "_mv_diag_ctr", 0) + 1
                        eng = self.movement.engine
                        if eng is not None and self._mv_diag_ctr % 20 == 0:
                            tc = int(eng.trig_ch)
                            if 0 <= tc < raw_samples.shape[0]:
                                tr = raw_samples[tc]
                                lab = (mv_result or self.movement.last_decision or {}).get("label", "-")
                                stim = (mv_result or self.movement.last_decision or {}).get("stim")
                                print(f"[MovementRoute] trig cur={tr[-1]:.0f} min={tr.min():.0f} "
                                      f"max={tr.max():.0f} vs mid={eng.trig_mid:.0f} | "
                                      f"routed clean={self.movement.route_clean} "
                                      f"stim={self.movement.route_stim} | last={lab} "
                                      f"({'STIM' if stim else 'CLEAN'} model)")
                    except Exception as e:
                        print(f"[MovementServer] Movement classification error: {e}")

            await asyncio.sleep(self.chunk_interval_ms / 1000.0)

    def shutdown(self):
        """Clean up resources."""
        self._running = False
        if self.device is not None:
            del self.device
            self.device = None
        if self.lsl_outlet is not None:
            del self.lsl_outlet
            self.lsl_outlet = None

    async def run(self):
        """Start the server and data streaming."""
        self._running = True

        # Try the initial connection, but don't exit if the device is missing —
        # the stream loop keeps retrying so the server survives a late/absent
        # or interrupted Ripple stream.
        if not self.connect_to_device():
            print(f"Ripple device not available yet; "
                  f"retrying every {self.reconnect_interval_s}s in the background.")

        print(f"\nStarting WebSocket server on ws://{self.host}:{self.port}")
        print("Waiting for frontend connections...")

        try:
            async with serve(self.handle_client, self.host, self.port):
                await self.stream_data()
        finally:
            self.shutdown()
