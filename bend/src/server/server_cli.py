"""
CLI for the combined Ripple + WebSocket server.

Usage:
    uv run lsl-ripple-server [OPTIONS]
"""

import asyncio
import typer
from typing import Optional

# Imported at module scope (unlike the server itself, which stays lazy) because
# typer evaluates option defaults at decoration time. Cheap: stim_authority
# pulls in httpx and nothing else.
from server.stim_authority import DEFAULT_MAX_TRAIN_SECONDS

app = typer.Typer(help="Combined Ripple Trellis + WebSocket server")


@app.command()
def main(
    host: str = typer.Option("localhost", "--host", "-h", help="Server host"),
    port: int = typer.Option(8765, "--port", "-p", help="WebSocket server port"),
    stream_type: str = typer.Option(
        "hi-res",
        "--stream-type", "-s",
        help="Ripple stream type: hi-res, raw, or lfp"
    ),
    pre_trigger: float = typer.Option(
        10.0,
        "--pre-trigger",
        help="Seconds of pre-trigger data to capture for recordings"
    ),
    output: str = typer.Option(
        "./recordings",
        "--output", "-o",
        help="Output folder for recordings"
    ),
    no_filter: bool = typer.Option(
        False,
        "--no-filter",
        help="Disable bandpass and notch filtering"
    ),
    no_lsl: bool = typer.Option(
        False,
        "--no-lsl",
        help="Disable LSL outlet (WebSocket only)"
    ),
    chunk_interval: int = typer.Option(
        50,
        "--chunk-interval",
        help="Milliseconds between data chunks"
    ),
    lowcut: float = typer.Option(
        20.0,
        "--lowcut",
        help="Bandpass filter low cutoff frequency (Hz)"
    ),
    highcut: float = typer.Option(
        500.0,
        "--highcut",
        help="Bandpass filter high cutoff frequency (Hz)"
    ),
    notch: float = typer.Option(
        50.0,
        "--notch",
        help="Notch filter frequency (Hz)"
    ),
    stim_controller_url: Optional[str] = typer.Option(
        None,
        "--stim-controller-url",
        help="Stimulator controller base URL (default http://127.0.0.1:11051). "
             "Server-side only: a controller_url sent by a client is ignored."
    ),
    stim_max_seconds: float = typer.Option(
        DEFAULT_MAX_TRAIN_SECONDS,
        "--stim-max-seconds",
        help="Dead-man deadline for one stimulation train (s). A backstop against "
             "a stranded train, NOT a therapy parameter — see stim_authority.py."
    ),
):
    """
    Start the combined Ripple Trellis + WebSocket server.

    Connects directly to Ripple hardware and streams data over WebSocket
    to a frontend, with optional LSL outlet for other consumers.
    """
    from server.websocket_server import RippleWebSocketServer

    server = RippleWebSocketServer(
        host=host,
        port=port,
        pre_trigger_seconds=pre_trigger,
        output_folder=output,
        enable_filtering=not no_filter,
        enable_lsl=not no_lsl,
        stream_type=stream_type,
        chunk_interval_ms=chunk_interval,
        filter_lowcut=lowcut,
        filter_highcut=highcut,
        filter_notch=notch,
        stim_controller_url=stim_controller_url,
        stim_max_seconds=stim_max_seconds,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer stopped by user")


if __name__ == "__main__":
    app()
