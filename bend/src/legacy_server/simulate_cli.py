from typing import Annotated

import typer

from server.devices.simulated import SimulatedDevice
from legacy_server.simulated_stream import SimulatedStream


def _main(
    npz_path: Annotated[str, typer.Argument(help="Path to .npz file containing EMG data")],
    srate: Annotated[float, typer.Option(help="Sample rate of the data in Hz")] = 2000.0,
    loop: Annotated[bool, typer.Option(help="Loop the data continuously")] = False,
):
    """
    Stream EMG data from an .npz file over LSL, simulating a real device.
    """
    print(f"Starting simulated EMG stream from: {npz_path}")
    print(f"Sample rate: {srate} Hz")
    print("-" * 50)

    device = SimulatedDevice(npz_path=npz_path, srate=srate)
    stream = SimulatedStream(device)

    print("-" * 50)
    print("Streaming... (Ctrl+C to stop)")
    print("-" * 50)

    try:
        stream.start(loop=loop)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        stream.shutdown()
        print("Stream closed.")


def cli():
    """Entry point for the CLI."""
    typer.run(_main)


if __name__ == "__main__":
    cli()
