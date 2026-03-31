#!/usr/bin/env -S uv run --python 3.13 --with reachy-mini
"""Start the Reachy Mini daemon on USB serial."""

import glob
import subprocess
import sys

PORT = 9002


def find_serial_port():
    candidates = glob.glob("/dev/cu.usbmodem*") + glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
    if not candidates:
        print("No USB serial device found. Is Reachy plugged in?", file=sys.stderr)
        sys.exit(1)
    if len(candidates) > 1:
        print(f"Multiple devices found: {candidates}, using {candidates[0]}", file=sys.stderr)
    return candidates[0]


def main():
    port = find_serial_port()
    print(f"Starting reachy-mini-daemon on {port} (API port {PORT})")
    subprocess.run(
        [
            sys.executable, "-m", "reachy_mini.daemon.app.main",
            "-p", port,
            "--fastapi-port", str(PORT),
            "--no-media",
        ],
    )


if __name__ == "__main__":
    main()
