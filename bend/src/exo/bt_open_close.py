"""
Open-close test for Tenoexo via Bluetooth COM port (PySerial).

Tests trigger_open_close, key_grip, and midway_opening.

Usage:
    python bt_open_close.py COM5
    python bt_open_close.py COM5 --delay 10
"""

import sys
import json
import time
import argparse

import serial


def send(port, command, send_id, args=None):
    if args is None:
        args = []
    msg = json.dumps({"command": command, "send": send_id, "args": args})
    print(f"  TX -> {msg}")
    port.write(msg.encode("utf-8"))
    try:
        data = port.read(4096)
        text = data.decode("utf-8", errors="replace")
        if text:
            print(f"  RX <- {text}")
        return text
    except Exception as e:
        print(f"  RX <- [error: {e}]")
        return None


def main():
    parser = argparse.ArgumentParser(description="Tenoexo open/close via Bluetooth COM port")
    parser.add_argument("port", help="COM port (e.g. COM5)")
    parser.add_argument("--delay", type=float, default=10, help="Seconds between open and close (default: 10)")
    args = parser.parse_args()

    print(f"Connecting to {args.port}...")
    try:
        port = serial.Serial(args.port, timeout=5)
    except serial.SerialException as e:
        print(f"Failed to open {args.port}: {e}")
        sys.exit(1)

    print("Connected.\n")

    try:
        # Heartbeat to verify the link
        print("--- Heartbeat ---")
        resp = send(port, "heart_beat", 0)
        if resp and '"received"' in resp:
            print("Link OK.\n")
        else:
            print("No ack for heartbeat, continuing anyway...\n")

        send_id = 1

        # --- Normal open/close ---
        print("=== NORMAL OPEN/CLOSE ===\n")

        print("--- Triggering OPEN ---")
        send(port, "trigger_open_close", send_id); send_id += 1

        print(f"\nWaiting {args.delay}s...\n")
        time.sleep(args.delay)

        print("--- Triggering CLOSE ---")
        send(port, "trigger_open_close", send_id); send_id += 1

        print(f"\nWaiting {args.delay}s...\n")
        time.sleep(args.delay)

        # --- Key grip ---
        print("=== KEY GRIP ===\n")

        print("--- Enabling key grip ---")
        send(port, "key_grip", send_id, [True]); send_id += 1

        print(f"\nWaiting 2s...\n")
        time.sleep(2)

        print("--- Triggering OPEN (key grip) ---")
        send(port, "trigger_open_close", send_id); send_id += 1

        print(f"\nWaiting {args.delay}s...\n")
        time.sleep(args.delay)

        print("--- Triggering CLOSE (key grip) ---")
        send(port, "trigger_open_close", send_id); send_id += 1

        print(f"\nWaiting {args.delay}s...\n")
        time.sleep(args.delay)

        print("--- Disabling key grip ---")
        send(port, "key_grip", send_id, [False]); send_id += 1

        print(f"\nWaiting 2s...\n")
        time.sleep(2)

    

    finally:
        port.close()
        print("\nPort closed.")


if __name__ == "__main__":
    main()
