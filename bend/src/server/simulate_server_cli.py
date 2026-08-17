"""
CLI for the Simulated WebSocket server.

Usage:
    uv run lsl-simulate-server data.npz [OPTIONS]
"""

import asyncio
import typer
from typing import Annotated

app = typer.Typer(help="Simulated EMG WebSocket server (plays back .npz files)")


@app.command()
def main(
    npz_path: Annotated[str, typer.Argument(help="Path to .npz file containing EMG data")],
    host: str = typer.Option("localhost", "--host", "-h", help="Server host"),
    port: int = typer.Option(8765, "--port", "-p", help="WebSocket server port"),
    srate: float = typer.Option(
        2000.0,
        "--srate", "-r",
        help="Sample rate of the data in Hz"
    ),
    loop: bool = typer.Option(
        False,
        "--loop", "-l",
        help="Loop the data continuously"
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
):
    """
    Start a simulated WebSocket server that plays back EMG data from an .npz file.

    This mimics the RippleWebSocketServer but uses pre-recorded data instead of
    real hardware, useful for testing the frontend without Ripple equipment.
    """
    from server.simulated_websocket_server import SimulatedWebSocketServer

    print(f"Starting simulated EMG WebSocket server")
    print(f"Data file: {npz_path}")
    print(f"Sample rate: {srate} Hz")
    print(f"Loop: {'enabled' if loop else 'disabled'}")
    print("-" * 50)

    server = SimulatedWebSocketServer(
        npz_path=npz_path,
        host=host,
        port=port,
        srate=srate,
        loop=loop,
        pre_trigger_seconds=pre_trigger,
        output_folder=output,
        enable_filtering=not no_filter,
        enable_lsl=not no_lsl,
        chunk_interval_ms=chunk_interval,
        filter_lowcut=lowcut,
        filter_highcut=highcut,
        filter_notch=notch,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n\nServer stopped by user")


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
