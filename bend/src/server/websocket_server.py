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
from server.stim_authority import StimAuthority, DEFAULT_MAX_TRAIN_SECONDS
from server.recover_recording import (
    SPOOL_KIND, SPOOL_LAYOUT, SPOOL_SUFFIX, build_recording_payload, load_spool,
    safe_name, segment_paths, sidecar_path, unique_path, write_pickle_atomic,
    write_sidecar,
)


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

    Samples are NOT held here. They go straight to an on-disk spool as they
    arrive (audit P5: the old `recorded_chunks` list grew ~7 GB/hour at
    Quattrocento rates and was lost in its entirety if the process died before
    `stop_recording` pickled it). See `server/recover_recording.py` for the
    spool format and the recovery tool.
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
    # Non-zero only when a chunk failed to reach disk; surfaced in the saved file
    # so a recording with a gap is never silently passed off as complete.
    spool_write_errors: int = 0
    spool_dropped_samples: int = 0
    spool_last_error: Optional[str] = None


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
        stim_controller_url: Optional[str] = None,
        stim_max_seconds: float = DEFAULT_MAX_TRAIN_SECONDS,
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

        # Stimulation authority: validates parameters, owns the active train,
        # arms the dead-man deadline and is the only writer of stim state. The
        # controller URL is SERVER configuration, not a message field — a client
        # that sends one is ignored (audit S7). self.broadcast is a bound method
        # and is safe to hand over here.
        self.stim = StimAuthority(
            client=StimulationClient(
                base_url=stim_controller_url) if stim_controller_url else StimulationClient(),
            broadcast=self.broadcast,
            max_train_seconds=stim_max_seconds,
        )

        # Live references to fire-and-forget recording_warning broadcasts, so
        # the tasks are not garbage-collected mid-flight.
        self._recording_warning_tasks = set()

        # Teardown latch: run()'s finally and any direct shutdown() call must not
        # tear the same resources down twice.
        self._torn_down = False

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

        stimulation_active / recording let a reconnecting client rehydrate the
        two pieces of state it cannot observe (audit S5/B1). Without them the
        frontend's emergency stop is a no-op after a blip: its local `stimming`
        flag initialises false, so it sends nothing while a train still runs.
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
            "stimulation_active": self.stim.active,
            "recording": self.recording.is_recording,
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
            # A departing socket must not leave its train running (audit S2).
            # The authority stops it when this socket owned it, and also when it
            # was the last one — with nobody connected, no `stimulate_stop` can
            # ever arrive. Never let this abort the handler's cleanup.
            try:
                await self.stim.on_client_disconnect(websocket, len(self.clients))
            except Exception as e:
                print(f"[Server] stim stop-on-disconnect failed: {e}")
            # A recording nobody is connected to can never be stopped by anyone,
            # and used to just keep growing (audit P5). The samples are already
            # spooled, so finalising here only costs the reassembly.
            if not self.clients and self.recording.is_recording:
                print("[Server] last client disconnected while recording — "
                      "finalising the recording")
                try:
                    await self.stop_recording(None)
                except Exception as e:
                    print(f"[Server] finalise on last disconnect failed: {e}")

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
                    # Audit P5: this used to clear _mv_raw unconditionally, so a
                    # duplicate start silently threw away a block the operator
                    # believed was being recorded. Refuse instead.
                    await websocket.send(json.dumps({
                        "type": "mv_train_status", "state": "error",
                        "message": ("a movement capture is already in progress "
                                    f"({len(self._mv_raw)} chunks); stop it with "
                                    "mv_record_stop before starting another")}))
                else:
                    self._mv_raw = []
                    self._mv_capturing = True
                    self._mv_paused = False
                    self._mv_meta = data      # label + class_order/sequence + montage
                    await self.broadcast({"type": "mv_train_status", "state": "recording"})

            elif command == "mv_record_pause":       # tutorial phase — exclude from data
                self._mv_paused = True

            elif command == "mv_record_resume":
                self._mv_paused = False

            elif command == "mv_record_stop":
                self._mv_capturing = False
                # Audit B2: stop MUST clear the pause latch. The frontend
                # deliberately skips resume-before-stop, so without this an
                # aborted session left _mv_paused set forever and every later
                # start_recording produced a file holding only the pre-trigger
                # buffer, with a cheerful "Recording started" and no error.
                self._mv_paused = False
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
                # The authority validates, binds the train to this socket, arms
                # the deadline and publishes stimulation_status itself.
                await self.stim.handle_start(data, websocket)

            elif command == "stimulate_stop":
                await self.stim.handle_stop(websocket)

            elif command == "set_otb_config":
                await self.apply_device_config(data.get("path"), websocket)

            elif command == "get_status":
                decomp_status = self.decomp.get_status()
                await websocket.send(json.dumps({
                    "type": "status",
                    "connected_to_device": self.device is not None,
                    "recording": self.recording.is_recording,
                    "stimulation_active": self.stim.active,
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
        except websockets.exceptions.ConnectionClosed:
            # Not a command failure: the socket dropped mid-command. Must be
            # re-raised so handle_client's dedicated handler sees it, rather
            # than logged as an alarming traceback and answered down a socket
            # that is already gone.
            raise
        except Exception as e:
            # Audit D3: this used to catch JSONDecodeError only, so any other
            # exception (int() on a non-numeric channel, a ragged hstack, a
            # pkl.dump failure) escaped and closed the socket with a 1011. That
            # is now actively dangerous: stop-on-disconnect would read the close
            # as "the operator went away" and stop a legitimate train, and it
            # would take recording control down with it. Report and stay up.
            import traceback
            print(f"[Server] Unhandled error while processing command "
                  f"{command!r}: {e}")
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Command '{command}' failed: {e}"
                }))
            except Exception:
                pass    # the socket is already gone; nothing left to report to

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
        """Filesystem-safe token for building informative recording filenames.

        Delegates to server.recover_recording so the recovery tool and the sim
        server spell subject/session tokens exactly the same way.
        """
        return safe_name(s, default)

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

    # ── recording spool (audit P5) ───────────────────────────────────────────

    def _recording_base_name(self, metadata: Optional[dict]) -> str:
        """`emg_recording_<subject>_<session>_<start-ts>`.

        The subject/session tokens come from the frontend metadata, sanitized
        the same way `_save_mv_recording` does. The old name was the timestamp
        alone, so recordings were anonymous on disk AND two stops in the same
        second overwrote each other (audit P5).
        """
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
            "stream_type": self.stream_type,
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
            # Flush every chunk, but do NOT fsync. The failure this protects
            # against is process death - a closed console window, a crash, a
            # taskkill - and a flushed write survives all of those in the OS
            # page cache. Without it the 1 MB buffer would hold back ~0.5s of
            # Quattrocento data, which is exactly what a crash would eat.
            # fsync per chunk would be a disk round trip 20x a second for a
            # power-cut guarantee nobody asked for.
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
            write_sidecar(path, info)          # final sample count
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

        # Audit B2: `_mv_paused` gates the general recorder and is only cleared
        # by mv_record_resume/start. A movement session aborted mid-tutorial left
        # it latched, and every later recording silently contained nothing but
        # the pre-trigger buffer. A fresh recording always starts unpaused.
        if self._mv_paused:
            print("[Recording] _mv_paused was still latched from an earlier movement "
                  "session; clearing it so this recording is not silently empty")
            self._mv_paused = False

        # Pre-trigger snapshot, unchanged: taken before anything else so it is
        # the buffer as it stood the instant recording was requested.
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

        # The pre-trigger goes in first, as part of the same stream, so the
        # reassembled array is exactly what hstack([pre, *chunks]) produced.
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

        The reassembly and the pickle run in a worker thread: at Quattrocento
        rates this is a multi-GB write, and doing it on the event loop stalled
        everything queued behind it - including `stimulate_stop` (audit P6).

        A failed save never destroys data. The spool is deleted only after the
        .pkl is durably on disk; if the save fails the spool stays put and the
        operator is told how to recover it (audit P5).

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
            # Named after the spool, so the two are obviously the same recording
            # if a save ever fails. The `timestamp` INSIDE the file stays the
            # stop time, exactly as before.
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
        """Handle a streaming failure: stop stimulation, drop the device, notify."""
        print(f"[Server] Lost connection to Ripple device: {error}")
        print(f"[Server] Will retry every {self.reconnect_interval_s}s...")
        # Audit S3: the closed loop only closes the gate when a NEW decision
        # says so. No device means no decisions, so an active train would run
        # until someone noticed. Stop it before anything else.
        try:
            await self.stim.on_device_lost()
        except Exception as e:
            print(f"[Server] stim stop on device loss failed: {e}")
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

                # If recording, append straight to the on-disk spool (audit P5:
                # this used to be `recorded_chunks.append(samples.copy())`, i.e.
                # unbounded RAM that a crash threw away in full).
                # Paused during a tutorial phase so it's excluded from the saved
                # recording too. `_mv_paused` really is False for any
                # non-movement-session recording: mv_record_stop clears it and
                # start_recording clears it defensively (audit B2).
                if self.recording.is_recording and not self._mv_paused:
                    self._spool_write(samples)

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
        """Synchronous resource release: stop the loops, drop the device and LSL.

        Unchanged in behaviour, and kept under this name for any caller that
        cannot await. It is NOT a safe shutdown on its own: being synchronous it
        structurally cannot await the stimulator stop, the recording flush or the
        writer close (audit S2/S8.4). `teardown()` does those first and then
        calls this as its last step — off the event-loop thread, because
        RippleDevice.__del__ sleeps ~1s (devices/ripple.py:75).
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

        Order is deliberate — the patient first, then the data that only exists
        in RAM, then hardware handles:

            1. stimulation off (the only thing here attached to a patient, and
               the only one the external controller keeps doing forever if we
               exit without saying anything);
            2. finalise an in-progress recording - the samples are already on
               disk in the spool, so this only reassembles them into the .pkl,
               and even a failure here loses nothing (audit P5);
            3. close the online-run writer so the .raw gets its .pkl sidecar —
               without the sidecar the shape/dtype are unrecorded and the file
               is unreadable;
            4. save an in-progress movement-training capture, same reason as 2;
            5. release the exo serial port (audit P4: never closed before);
            6. device + LSL outlet.

        Every step is individually guarded: a failure in one must not skip the
        rest, and step 1 must never be skipped by a failure in any other.
        """
        if self._torn_down:
            return
        self._torn_down = True
        print("\n[Teardown] shutting down...")
        # Stop the acquisition/reconnect and time-sync loops before their
        # resources go away.
        self._running = False

        try:
            await self.stim.shutdown()
        except Exception as e:
            print(f"[Teardown] stimulation stop FAILED: {e} — check the stimulator")

        await self._flush_recording_for_teardown()

        if self._mv_online_file is not None:
            self._mv_online_capturing = False
            try:
                self._close_online_writer(self._mv_online_dec, self._mv_online_meta)
            except Exception as e:
                print(f"[Teardown] closing the online writer failed: {e}")
            self._mv_online_dec = []

        if self._mv_raw:
            print(f"[Teardown] saving in-progress movement capture "
                  f"({len(self._mv_raw)} chunks)")
            try:
                raw = np.hstack(self._mv_raw)
                self._mv_capturing = False
                self._mv_raw = []
                await asyncio.to_thread(self._save_mv_recording, raw, self._mv_meta)
            except Exception as e:
                print(f"[Teardown] movement capture save failed: {e}")

        try:
            self.decomp.stop_classification()       # closes the exo COM port
        except Exception as e:
            print(f"[Teardown] stop_classification failed: {e}")

        # __del__ sleeps ~1s, so drop the device off the event-loop thread —
        # the reconnect path already does this (_on_device_lost).
        try:
            await asyncio.to_thread(self.shutdown)
        except Exception as e:
            print(f"[Teardown] device/LSL cleanup failed: {e}")

        print("[Teardown] done.")

    async def _flush_recording_for_teardown(self):
        """Finalise a running recording instead of dropping it.

        The samples are already in the spool, so this is just the normal stop:
        reassemble and write the .pkl. The old emergency in-RAM dump this used
        to fall back to is gone with the RAM buffer it dumped — if the save
        fails now, stop_recording keeps the spool and prints the recovery
        command, which is strictly better than a bespoke rescue format.
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
