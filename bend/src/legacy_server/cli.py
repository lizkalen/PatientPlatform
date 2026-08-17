import typer

from server.devices.ripple import RippleDevice
from legacy_server.stream import RippleStream


def main(targ_stream_type: str = "hi-res"):
    device = RippleDevice(targ_stream_type=targ_stream_type)
    stream = RippleStream(device)
    try:
        stream.start()
    except KeyboardInterrupt:
        print("Application interrupted. Exiting.")
        stream.shutdown()


if __name__ == "__main__":
    typer.run(main)
