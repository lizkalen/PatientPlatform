"""
Test script for decomposition streaming over WebSocket.

Usage:
    1. Start the simulated server:
       cd bend
       python src/server/simulate_server_cli.py ../datafile1_filtered.npz --loop --no-lsl --port 8765

    2. Run this test script in another terminal:
       cd bend
       python test_decomp_ws.py --model path/to/results_combined.pkl

    The script connects to the WebSocket server, sends a load_model command,
    and prints decomposition_chunk messages as they arrive.
"""

import asyncio
import json
import argparse
import sys


async def test_decomposition(host, port, model_path, duration):
    """Connect to the server, load model, and print decomposition output."""
    try:
        import websockets
    except ImportError:
        print("Please install websockets: pip install websockets")
        sys.exit(1)

    url = f"ws://{host}:{port}"
    print(f"Connecting to {url}...")

    async with websockets.connect(url) as ws:
        # Wait for connected message
        msg = json.loads(await ws.recv())
        print(f"\n[connected] {msg.get('n_channels')} channels @ {msg.get('sample_rate')} Hz")
        print(f"  decomposition_active: {msg.get('decomposition_active', False)}")

        # Load model
        print(f"\nSending load_model: {model_path}")
        await ws.send(json.dumps({
            "command": "load_model",
            "model_path": model_path,
        }))

        # Listen for messages
        chunk_count = 0
        start_time = asyncio.get_event_loop().time()

        try:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if duration and elapsed > duration:
                    print(f"\n--- Duration ({duration}s) reached, stopping ---")
                    break

                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "decomposition_status":
                    print(f"\n[decomposition_status] active={msg.get('active')}, "
                          f"n_mus={msg.get('n_mus')}")
                    if msg.get("channels_to_remove"):
                        print(f"  channels_to_remove: {msg['channels_to_remove']}")

                elif msg_type == "decomposition_chunk":
                    chunk_count += 1
                    n_mus = len(msg["sources"])
                    n_samples = msg["n_samples"]
                    fr = msg["firing_rates"]
                    sil = msg["sil_scores"]
                    spikes = msg["spikes"]

                    spike_str = ", ".join(
                        f"MU{k}:{len(v)}" for k, v in spikes.items() if len(v) > 0
                    )

                    print(f"  [chunk #{chunk_count:4d}] {n_mus} MUs, {n_samples} samples | "
                          f"FR: [{', '.join(f'{f:.1f}' for f in fr)}] Hz | "
                          f"SIL: [{', '.join(f'{s:.2f}' for s in sil)}] | "
                          f"spikes: {spike_str or 'none'}")

                    if msg.get("classification"):
                        c = msg["classification"]
                        label_names = {0: "REST", 1: "MOVE1", 2: "MOVE2"}
                        print(f"         classification: {label_names.get(c['label'], c['label'])} "
                              f"(MU1={c['mu1_firing_rate']:.1f} Hz, MU2={c['mu2_firing_rate']:.1f} Hz)"
                              + (f" [{c['transition']}]" if c.get("transition") else ""))

                elif msg_type == "emg_chunk":
                    pass  # Skip EMG chunks (too verbose)

                elif msg_type == "error":
                    print(f"\n[ERROR] {msg.get('message')}")

                elif msg_type == "classification_status":
                    print(f"\n[classification_status] active={msg.get('active')}")

                else:
                    print(f"  [{msg_type}] {json.dumps(msg, indent=2)[:200]}")

        except asyncio.TimeoutError:
            print("\n--- No messages for 5s, stopping ---")

        except KeyboardInterrupt:
            pass

        # Unload model before disconnecting
        print(f"\nSending unload_model...")
        await ws.send(json.dumps({"command": "unload_model"}))
        try:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            if msg.get("type") == "decomposition_status":
                print(f"[decomposition_status] active={msg.get('active')}")
        except (asyncio.TimeoutError, Exception):
            pass

    print(f"\nDone. Received {chunk_count} decomposition chunks.")


def main():
    parser = argparse.ArgumentParser(description="Test decomposition WebSocket streaming")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--model", "-m", required=True,
                        help="Path to decomposition model .pkl file (results_combined.pkl)")
    parser.add_argument("--duration", "-d", type=float, default=30.0,
                        help="Listen duration in seconds (default: 30, 0=infinite)")
    parser.add_argument("--classify", "-c", action="store_true",
                        help="Also test classification (MU0 + MU1)")
    args = parser.parse_args()

    async def run():
        try:
            import websockets
        except ImportError:
            print("Please install websockets: pip install websockets")
            sys.exit(1)

        url = f"ws://{args.host}:{args.port}"
        print(f"Connecting to {url}...")

        async with websockets.connect(url) as ws:
            # Wait for connected message
            msg = json.loads(await ws.recv())
            print(f"\n[connected] {msg.get('n_channels')} chs @ {msg.get('sample_rate')} Hz")
            print(f"  decomposition_active: {msg.get('decomposition_active', False)}")

            # Load model
            print(f"\nLoading model: {args.model}")
            await ws.send(json.dumps({
                "command": "load_model",
                "model_path": args.model,
            }))

            # Optionally start classification
            if args.classify:
                # Wait for decomp status first
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    m = json.loads(raw)
                    if m.get("type") == "decomposition_status":
                        print(f"[decomposition_status] active={m.get('active')}, n_mus={m.get('n_mus')}")
                        break
                    elif m.get("type") == "error":
                        print(f"[ERROR] {m.get('message')}")
                        return

                print("\nStarting classification (MU0 + MU1)...")
                await ws.send(json.dumps({
                    "command": "start_classification",
                    "mu1_idx": 0,
                    "mu2_idx": 1,
                    "threshold": 10.0,
                    "window_sec": 1.0,
                }))

            # Listen
            chunk_count = 0
            start_time = asyncio.get_event_loop().time()

            try:
                while True:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if args.duration > 0 and elapsed > args.duration:
                        print(f"\n--- Duration ({args.duration}s) reached ---")
                        break

                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    msg = json.loads(raw)
                    msg_type = msg.get("type")

                    if msg_type == "decomposition_chunk":
                        chunk_count += 1
                        n_mus = len(msg["sources"])
                        fr = msg["firing_rates"]
                        sil = msg["sil_scores"]
                        spikes = msg["spikes"]
                        spike_str = ", ".join(
                            f"MU{k}:{len(v)}" for k, v in spikes.items() if len(v) > 0
                        )
                        print(f"  [#{chunk_count:4d}] {n_mus} MUs, {msg['n_samples']} samp | "
                              f"FR: [{', '.join(f'{f:.1f}' for f in fr)}] | "
                              f"SIL: [{', '.join(f'{s:.2f}' for s in sil)}] | "
                              f"spk: {spike_str or '-'}", end="")
                        if msg.get("classification"):
                            c = msg["classification"]
                            names = {0: "REST", 1: "MOVE1", 2: "MOVE2"}
                            print(f" | {names.get(c['label'], '?')}", end="")
                            if c.get("transition"):
                                print(f" [{c['transition']}]", end="")
                        print()

                    elif msg_type == "decomposition_status":
                        print(f"[decomposition_status] active={msg.get('active')}, n_mus={msg.get('n_mus')}")

                    elif msg_type == "classification_status":
                        print(f"[classification_status] active={msg.get('active')}")

                    elif msg_type == "error":
                        print(f"[ERROR] {msg.get('message')}")

                    elif msg_type == "emg_chunk":
                        pass  # skip

            except asyncio.TimeoutError:
                print("\n--- No messages for 5s ---")
            except KeyboardInterrupt:
                pass

            # Cleanup
            if args.classify:
                await ws.send(json.dumps({"command": "stop_classification"}))
            await ws.send(json.dumps({"command": "unload_model"}))

            print(f"\nReceived {chunk_count} decomposition chunks in {elapsed:.1f}s")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
