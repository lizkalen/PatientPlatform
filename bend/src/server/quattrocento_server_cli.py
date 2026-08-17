"""
CLI for the Quattrocento (OTBioLab+) WebSocket server.

Streams the OTBioLab+ TCP feed to the frontend over the same WebSocket protocol
as the Ripple server, so the existing frontend connects and records unchanged.

Prerequisite: OTBioLab+ is running with its TCP/IP option enabled and
acquisition started (the montage/sample rate are chosen there; this server
only starts/stops the transmission and decodes it).

Usage:
    python src/server/quattrocento_server_cli.py --port 8765 --pre-trigger 10
    python src/server/quattrocento_server_cli.py --otb-host 127.0.0.1 --otb-port 31000 --fsamp 2048
"""

import asyncio
from typing import Optional

import typer

app = typer.Typer(help="Quattrocento (OTBioLab+) + WebSocket server")


def _parse_channels(spec: Optional[str]):
    """'0-191', '0-127,128-191' -> sorted unique list; None -> None (all)."""
    if spec is None:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


@app.command()
def main(
    host: str = typer.Option("localhost", "--host", "-h", help="WebSocket server host"),
    port: int = typer.Option(8765, "--port", "-p", help="WebSocket server port"),
    otb_host: str = typer.Option("127.0.0.1", "--otb-host", help="OTBioLab+ TCP host"),
    otb_port: int = typer.Option(31000, "--otb-port", help="OTBioLab+ TCP port"),
    config: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to the OTBioLab+ configuration file (extensionless XML). "
             "Sets channel count and sample rate deterministically."
    ),
    fsamp: Optional[float] = typer.Option(
        None, "--fsamp",
        help="OTBioLab+ sample rate (Hz). Overrides --config; defaults to 2048."
    ),
    nch: Optional[int] = typer.Option(
        None, "--nch", help="Channels in the stream frame (overrides --config)."
    ),
    channels: Optional[str] = typer.Option(
        None, "--channels",
        help="Channels to expose downstream, e.g. '0-191' (default: all). "
             "OTBioLab+ streams EMG first, then control/aux."
    ),
    emg_only: bool = typer.Option(
        False, "--emg-only",
        help="With --config, expose only EMG channels (drop ramp/buffer/aux)."
    ),
    auto_detect: bool = typer.Option(
        False, "--auto-detect",
        help="Fall back to byte-rate channel detection (unreliable; OTBioLab+ "
             "streams in bursts). Prefer --config or --nch."
    ),
    scale: bool = typer.Option(
        False, "--scale/--no-scale",
        help="Convert EMG counts to mV. Default: off — stream raw int16 counts, "
             "exactly as OTBioLab+ transmits them (no resolution change)."
    ),
    pre_trigger: float = typer.Option(
        10.0, "--pre-trigger", help="Seconds of pre-trigger data for recordings"
    ),
    output: str = typer.Option("./recordings", "--output", "-o", help="Recordings folder"),
    do_filter: bool = typer.Option(
        False, "--filter/--no-filter",
        help="Apply bandpass + notch filtering. Default: off — record the raw "
             "stream as sent to OTBioLab+ (no filtering, no downsampling)."
    ),
    no_lsl: bool = typer.Option(False, "--no-lsl", help="Disable LSL outlet"),
    live_emg: bool = typer.Option(
        False, "--live-emg/--no-live-emg",
        help="Stream live EMG chunks to the frontend plot. Off by default for the "
             "Quattrocento: serializing 256-channel chunks stalls the event loop. "
             "Recording and LSL are unaffected either way."
    ),
    chunk_interval: int = typer.Option(50, "--chunk-interval", help="ms between chunks"),
    lowcut: float = typer.Option(20.0, "--lowcut", help="Bandpass low cutoff (Hz)"),
    highcut: float = typer.Option(500.0, "--highcut", help="Bandpass high cutoff (Hz)"),
    notch: float = typer.Option(50.0, "--notch", help="Notch frequency (Hz)"),
):
    """Start the Quattrocento (OTBioLab+) + WebSocket server."""
    from server.websocket_server import RippleWebSocketServer
    from server.devices.quattrocento import QuattrocentoDevice, parse_otb_config, CLOCK_RATE

    # Channel selection (explicit --channels wins over --emg-only over "all").
    keep = _parse_channels(channels)

    def _factory_for(nch_, fsamp_, keep_):
        def device_factory():
            return QuattrocentoDevice(
                host=otb_host,
                port=otb_port,
                fsamp=fsamp_,
                nch=nch_,
                keep_channels=keep_,
                scale_to_mv=scale,
                clock_rate=CLOCK_RATE,
                auto_detect=auto_detect,
            )
        return device_factory

    def _keep_for(cfg):
        if keep is not None:
            return keep
        if emg_only:
            return cfg["emg_channels"]
        return None

    def config_loader(path):
        """Parse an OTBioLab+ config file -> (device_factory, info dict).

        Used both for --config at startup and for the frontend's runtime
        "Load OTB Config" button (server command ``set_otb_config``).
        """
        cfg = parse_otb_config(path)
        keep_ = _keep_for(cfg)
        info = {
            "device_name": cfg["device_name"],
            "model": cfg["model"],
            "nch": cfg["nch"],
            "n_emg": len(cfg["emg_channels"]),
            "fsamp": cfg["fsamp"],
            "n_exposed": cfg["nch"] if keep_ is None else len(keep_),
        }
        return _factory_for(cfg["nch"], cfg["fsamp"], keep_), info

    # Resolve the initial device: config file first, then explicit flags.
    if config is not None:
        device_factory, cfg_info = config_loader(config)
        print(f"Loaded config: {cfg_info['device_name']} (model {cfg_info['model']}) - "
              f"{cfg_info['nch']} channels, {cfg_info['n_emg']} EMG, {cfg_info['fsamp']:g} Hz")
        resolved_fsamp = cfg_info["fsamp"]
        resolved_nch = cfg_info["nch"]
    else:
        resolved_fsamp = fsamp if fsamp is not None else 2048.0
        resolved_nch = nch
        if resolved_nch is None and not auto_detect:
            raise typer.BadParameter(
                "Provide --config or --nch (or use --auto-detect). Automatic detection "
                "is off by default because OTBioLab+ streams in bursts."
            )
        device_factory = _factory_for(resolved_nch, resolved_fsamp, keep)

    print("Starting Quattrocento (OTBioLab+) WebSocket server")
    print(f"  OTBioLab+ : {otb_host}:{otb_port} @ {resolved_fsamp:g} Hz")
    print(f"  Channels  : {'auto-detect' if resolved_nch is None else resolved_nch}"
          f"{'' if keep is None else f', exposing {len(keep)}'}")
    print(f"  Scaling   : {'mV' if scale else 'raw int16 counts'}")
    print(f"  Filtering : {'on' if do_filter else 'off (raw stream)'}")
    print(f"  Live EMG  : {'on' if live_emg else 'off (use --live-emg to enable)'}")
    print("-" * 50)

    server = RippleWebSocketServer(
        host=host,
        port=port,
        pre_trigger_seconds=pre_trigger,
        output_folder=output,
        enable_filtering=do_filter,
        enable_lsl=not no_lsl,
        broadcast_emg=live_emg,
        stream_type="quattrocento",
        chunk_interval_ms=chunk_interval,
        filter_lowcut=lowcut,
        filter_highcut=highcut,
        filter_notch=notch,
        device_factory=device_factory,
        device_kind="quattrocento",
        clock_rate=CLOCK_RATE,
        lsl_source_id="Quattrocento",
        lsl_name="Quattrocento",
        config_loader=config_loader,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\nServer stopped by user")


def cli():
    app()


if __name__ == "__main__":
    cli()
