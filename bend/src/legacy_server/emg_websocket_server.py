"""
EMG WebSocket Server  [SUPERSEDED — see server/websocket_server.py]
====================

A WebSocket server that streams EMG data from Trellis to a frontend
and supports triggered recording with pre-trigger buffer.

This consumed an LSL stream published by a separate process (legacy_server/cli.py).
It was replaced in February 2026 by server/websocket_server.py, which connects to
the device directly and publishes its own LSL outlet. Retained for reference; no .bat
file starts it.

Architecture:
    Trellis (LSL) --> This Server --> Frontend (WebSocket)
                          |
                          v
                    Saved .pkl files

Usage:
    python -m legacy_server.emg_websocket_server --port 8765 --pre-trigger 10

Frontend Integration:
    See the FRONTEND_API section below for WebSocket message formats.
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
import argparse

# WebSocket library
try:
    import websockets
    from websockets.server import serve
except ImportError:
    raise ImportError("Please install websockets: pip install websockets")

# LSL for Trellis connection
from pylsl import StreamInlet, resolve_streams, resolve_byprop

# Local imports for filtering
from dsp.processing import design_filters, init_filter_states, apply_filters


# =============================================================================
# FRONTEND API DOCUMENTATION
# =============================================================================
"""
FRONTEND_API
============

Connect to: ws://localhost:8765 (or your configured host:port)

MESSAGES FROM SERVER TO FRONTEND:
---------------------------------

1. Connection established:
   {
       "type": "connected",
       "sample_rate": 2048,
       "n_channels": 64,
       "pre_trigger_seconds": 10
   }

2. EMG data chunk (sent every ~50ms):
   {
       "type": "emg_chunk",
       "data": [[ch1_samples...], [ch2_samples...], ...],  // 2D array: [n_channels x n_samples]
       "timestamp": 1234567890.123,
       "n_samples": 102
   }

3. Recording status changed:
   {
       "type": "recording_status",
       "recording": true/false,
       "message": "Recording started" / "Recording stopped"
   }

4. Recording saved:
   {
       "type": "recording_saved",
       "filepath": "C:/path/to/recording_20240115_143022.pkl",
       "duration_seconds": 15.5,
       "n_channels": 64,
       "n_samples": 31744
   }

5. Error:
   {
       "type": "error",
       "message": "Error description"
   }


MESSAGES FROM FRONTEND TO SERVER:
---------------------------------

1. Start recording (with optional metadata):
   {
       "command": "start_recording",
       "metadata": {                    // Optional - trial metadata
           "subjectId": "subject_001",
           "sessionId": "session_001",
           "notes": "First trial",
           "sequenceName": "Default Training Sequence",
           "totalItems": 5,
           "settings": {
               "pauseBetweenItems": 0.5,
               "restBetweenReps": 2.0,
               "prepTime": 1.0,
               "playbackSpeed": 1.0
           },
           "items": [
               {"index": 0, "model": "wrist.glb", "animation": 0, "repetitions": 3},
               ...
           ],
           "startTime": "2024-01-15T14:30:22.123Z"
       }
   }

2. Stop recording (and save):
   {
       "command": "stop_recording"
   }

3. Get current status:
   {
       "command": "get_status"
   }

   Response:
   {
       "type": "status",
       "connected_to_trellis": true/false,
       "recording": true/false,
       "buffer_seconds": 10.0,
       "n_clients": 2
   }


EXAMPLE FRONTEND (JavaScript):
------------------------------

const ws = new WebSocket('ws://localhost:8765');

ws.onopen = () => {
    console.log('Connected to EMG server');
};

ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    switch(msg.type) {
        case 'connected':
            console.log(`Sample rate: ${msg.sample_rate} Hz, Channels: ${msg.n_channels}`);
            break;

        case 'emg_chunk':
            // msg.data is a 2D array [n_channels][n_samples]
            // Update your visualization here
            updatePlot(msg.data, msg.timestamp);
            break;

        case 'recording_status':
            updateRecordingUI(msg.recording);
            break;

        case 'recording_saved':
            console.log(`Saved: ${msg.filepath} (${msg.duration_seconds.toFixed(1)}s)`);
            break;
    }
};

// Start recording (captures last 10s + everything until stop)
function startRecording() {
    ws.send(JSON.stringify({ command: 'start_recording' }));
}

// Stop recording and save
function stopRecording() {
    ws.send(JSON.stringify({ command: 'stop_recording' }));
}
"""


@dataclass
class RecordingState:
    """Tracks the current recording state."""
    is_recording: bool = False
    pre_trigger_data: Optional[np.ndarray] = None  # Data from before trigger
    recorded_chunks: list = field(default_factory=list)  # Data after trigger
    start_time: Optional[datetime] = None
    metadata: Optional[dict] = None  # Trial metadata from frontend


class EMGWebSocketServer:
    """
    WebSocket server for streaming EMG data and handling triggered recordings.

    Attributes:
        host: Server host address
        port: Server port
        pre_trigger_seconds: Seconds of data to keep before recording trigger
        output_folder: Where to save recordings
        enable_filtering: Apply bandpass and notch filters
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        pre_trigger_seconds: float = 10.0,
        output_folder: str = "./recordings",
        enable_filtering: bool = True,
        chunk_interval_ms: int = 50
    ):
        self.host = host
        self.port = port
        self.pre_trigger_seconds = pre_trigger_seconds
        self.output_folder = output_folder
        self.enable_filtering = enable_filtering
        self.chunk_interval_ms = chunk_interval_ms

        # Will be set when Trellis connects
        self.sample_rate: Optional[float] = None
        self.n_channels: Optional[int] = None
        self.inlet: Optional[StreamInlet] = None

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

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    async def connect_to_trellis(self) -> bool:
        """Connect to the Trellis LSL stream."""
        print("Looking for Trellis LSL stream...")

        try:
            streams = resolve_byprop("source_id", "RippleTrellis", timeout=5.0)
        except Exception:
            streams = []

        if not streams:
            # Fallback: try any stream
            streams = resolve_streams(wait_time=5.0)

        if not streams:
            print("No LSL stream found. Make sure lsl-ripple is running.")
            return False

        self.inlet = StreamInlet(streams[0], max_buflen=360, processing_flags=1)
        info = self.inlet.info()

        self.sample_rate = info.nominal_srate()
        self.n_channels = info.channel_count()

        # Initialize rolling buffer (stores chunks, will be flattened when needed)
        buffer_samples = int(self.pre_trigger_seconds * self.sample_rate)
        self.rolling_buffer = deque(maxlen=buffer_samples)

        # Initialize filters
        if self.enable_filtering:
            self.bp_sos, self.notch_sos = design_filters(
                self.sample_rate, lowcut=20.0, highcut=500.0, notch_freq=50.0
            )
            bp_zi, notch_zi = init_filter_states(
                self.bp_sos, self.notch_sos, self.n_channels
            )
            self.filter_state = {'bp_zi': bp_zi, 'notch_zi': notch_zi}

        print(f"Connected to: {info.name()}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Channels: {self.n_channels}")
        print(f"  Pre-trigger buffer: {self.pre_trigger_seconds}s ({buffer_samples} samples)")

        return True

    async def broadcast(self, message: dict):
        """Send a message to all connected clients."""
        if not self.clients:
            return

        msg_str = json.dumps(message)
        # Create tasks for all clients and handle disconnections
        disconnected = set()
        for client in list(self.clients):
            try:
                await client.send(msg_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)

        # Remove disconnected clients
        self.clients -= disconnected

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        """Handle a single WebSocket client connection."""
        self.clients.add(websocket)
        client_id = id(websocket)
        print(f"Client {client_id} connected. Total clients: {len(self.clients)}")

        # Send connection info
        await websocket.send(json.dumps({
            "type": "connected",
            "sample_rate": self.sample_rate,
            "n_channels": self.n_channels,
            "pre_trigger_seconds": self.pre_trigger_seconds
        }))

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
                await self.stop_recording()

            elif command == "get_status":
                await websocket.send(json.dumps({
                    "type": "status",
                    "connected_to_trellis": self.inlet is not None,
                    "recording": self.recording.is_recording,
                    "buffer_seconds": self.pre_trigger_seconds,
                    "n_clients": len(self.clients)
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

    async def start_recording(self, metadata: Optional[dict] = None):
        """Start recording, capturing the pre-trigger buffer.

        Args:
            metadata: Optional trial metadata from frontend containing:
                - subjectId: Subject/participant identifier
                - sessionId: Session identifier
                - notes: Free-form notes
                - sequenceName: Name of the sequence being recorded
                - totalItems: Total items in sequence
                - settings: Dictionary of sequence settings
                - items: List of sequence items with model/animation/repetitions
                - startTime: ISO timestamp when sequence started
        """
        if self.recording.is_recording:
            await self.broadcast({
                "type": "error",
                "message": "Recording already in progress"
            })
            return

        # Capture current rolling buffer as pre-trigger data
        if self.rolling_buffer:
            buffer_list = list(self.rolling_buffer)
            if buffer_list:
                self.recording.pre_trigger_data = np.array(buffer_list).T  # Shape: [n_channels, n_samples]
            else:
                self.recording.pre_trigger_data = None

        self.recording.is_recording = True
        self.recording.recorded_chunks = []
        self.recording.start_time = datetime.now()
        self.recording.metadata = metadata

        if metadata:
            print(f"Recording started with metadata: subject={metadata.get('subjectId')}, session={metadata.get('sessionId')}")

        pre_duration = 0
        if self.recording.pre_trigger_data is not None:
            pre_duration = self.recording.pre_trigger_data.shape[1] / self.sample_rate

        print(f"Recording started (pre-trigger: {pre_duration:.1f}s)")

        await self.broadcast({
            "type": "recording_status",
            "recording": True,
            "message": f"Recording started with {pre_duration:.1f}s pre-trigger data"
        })

    async def stop_recording(self):
        """Stop recording and save the data."""
        if not self.recording.is_recording:
            await self.broadcast({
                "type": "error",
                "message": "No recording in progress"
            })
            return

        self.recording.is_recording = False

        # Combine pre-trigger and recorded data
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

        # Combine all data
        full_recording = np.hstack(all_chunks)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emg_recording_{timestamp}.pkl"
        filepath = os.path.join(self.output_folder, filename)

        # Save as pickle (compatible with existing codebase)
        recording_data = {
            "data": full_recording,
            "srate": self.sample_rate,
            "n_channels": self.n_channels,
            "filtered": self.enable_filtering,
            "pre_trigger_seconds": self.pre_trigger_seconds,
            "timestamp": timestamp,
            # Trial metadata from frontend
            "trial_metadata": self.recording.metadata,
        }

        with open(filepath, "wb") as f:
            pkl.dump(recording_data, f)

        duration = full_recording.shape[1] / self.sample_rate
        print(f"Recording saved: {filepath} ({duration:.1f}s, {self.n_channels} channels)")

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

        # Clear recording state
        self.recording.pre_trigger_data = None
        self.recording.recorded_chunks = []
        self.recording.start_time = None
        self.recording.metadata = None

    async def stream_emg(self):
        """Continuously read EMG from Trellis and broadcast to clients."""
        if self.inlet is None:
            print("Cannot stream: not connected to Trellis")
            return

        print(f"Starting EMG stream (chunk interval: {self.chunk_interval_ms}ms)")

        while True:
            # Pull available samples
            samples, timestamps = self.inlet.pull_chunk(
                timeout=0.0,
                max_samples=int(self.sample_rate)
            )

            if samples:
                samples = np.array(samples).T  # Shape: [n_channels, n_samples]
                n_new = samples.shape[1]

                # Apply filters if enabled
                if self.enable_filtering and self.filter_state:
                    samples, self.filter_state['bp_zi'], self.filter_state['notch_zi'] = apply_filters(
                        samples, self.bp_sos, self.notch_sos,
                        self.filter_state['bp_zi'], self.filter_state['notch_zi']
                    )

                # Update rolling buffer (sample by sample for deque)
                for i in range(n_new):
                    self.rolling_buffer.append(samples[:, i])

                # If recording, also store in recorded chunks
                if self.recording.is_recording:
                    self.recording.recorded_chunks.append(samples.copy())

                # Broadcast to clients
                await self.broadcast({
                    "type": "emg_chunk",
                    "data": samples.tolist(),
                    "timestamp": timestamps[-1] if timestamps else None,
                    "n_samples": n_new
                })

            # Wait before next chunk
            await asyncio.sleep(self.chunk_interval_ms / 1000.0)

    async def run(self):
        """Start the WebSocket server and EMG streaming."""
        # Connect to Trellis first
        if not await self.connect_to_trellis():
            print("Failed to connect to Trellis. Exiting.")
            return

        # Start WebSocket server
        print(f"\nStarting WebSocket server on ws://{self.host}:{self.port}")
        print("Waiting for frontend connections...")

        async with serve(self.handle_client, self.host, self.port):
            # Run EMG streaming in parallel
            await self.stream_emg()


def main():
    parser = argparse.ArgumentParser(
        description="EMG WebSocket Server for Trellis streaming with triggered recording"
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--pre-trigger", type=float, default=10.0,
        help="Seconds of pre-trigger data to capture (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default="./recordings",
        help="Output folder for recordings (default: ./recordings)"
    )
    parser.add_argument(
        "--no-filter", action="store_true",
        help="Disable bandpass and notch filtering"
    )
    parser.add_argument(
        "--chunk-interval", type=int, default=50,
        help="Milliseconds between EMG chunks (default: 50)"
    )

    args = parser.parse_args()

    server = EMGWebSocketServer(
        host=args.host,
        port=args.port,
        pre_trigger_seconds=args.pre_trigger,
        output_folder=args.output,
        enable_filtering=not args.no_filter,
        chunk_interval_ms=args.chunk_interval
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer stopped by user")


if __name__ == "__main__":
    main()
