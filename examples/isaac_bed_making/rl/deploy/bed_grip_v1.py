"""Sensor-gated LEFT-hand grip on a bedsheet, coordinated with the arm-overlay pull.

Talks to the running ``brainco_bridge`` (TCP JSON on 127.0.0.1:9877) — NOT the serial port directly.
  * send  {"cmd":"set","left":[6 floats 0-1]}   0=open 1=closed
  * query {"cmd":"get"}  → per-finger left_touch_force [5 u16], left_proximity [5 u16], left_touch_ok

Sequence (sentinel handshake with g1_bed_pull_v1.py):
  1. wait for ``--arm-reached-file`` (the arm has reached the sheet and is holding)
  2. close the LEFT fingers in steps, gating on the touch force rising above the idle baseline;
     stop early once enough fingers register contact (a grip), else close to ``--grip-max``
  3. create ``--gripped-file``  → the arm starts the draw
  4. hold the grip (the hand servos hold the last commanded position) until ``--draw-done-file``
  5. release (open the fingers)

The LEFT hand is used because the right Brainco hand is mechanically damaged (kill-9 incident:
middle+pinky distal knuckles, thumb extension). Fingers/sensors are green-light (no leg/arm motion);
this commands only the hand.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time


class Bridge:
    def __init__(self, host, port):
        self._s = socket.socket()
        self._s.settimeout(5.0)
        self._s.connect((host, port))
        self._buf = b""

    def _rpc(self, obj):
        self._s.sendall((json.dumps(obj) + "\n").encode())
        while b"\n" not in self._buf:
            chunk = self._s.recv(8192)
            if not chunk:
                break
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        if not line.strip():
            raise RuntimeError("empty reply from brainco_bridge (connection closed mid-request — "
                               "usually a malformed command; note the bridge reads BOTH 'left' and "
                               "'right' on a 'set').")
        return json.loads(line)

    def get(self):
        return self._rpc({"cmd": "get"})

    def set_left(self, value6):
        # The bridge's "set" handler reads BOTH req["left"] and req["right"] unconditionally —
        # omitting "right" raises a server-side KeyError and the bridge closes the connection
        # (empty reply → JSONDecodeError). Always send both; keep the (damaged) right hand open.
        return self._rpc({"cmd": "set",
                          "left": [float(v) for v in value6],
                          "right": [0.0] * 6})

    def close(self):
        try:
            self._s.close()
        except OSError:
            pass


def _wait_for(path, timeout_s, log):
    if not path:
        return True
    log(f"[grip] waiting for {path} (arm to reach the sheet)…")
    t_end = time.time() + timeout_s
    while time.time() < t_end:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    log(f"[grip] timeout waiting for {path}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9877)
    ap.add_argument("--grip-max", type=float, default=0.85, help="max close fraction (0=open,1=closed)")
    ap.add_argument("--grip-step", type=float, default=0.1, help="close increment per step")
    ap.add_argument("--step-pause", type=float, default=0.35, help="seconds between close steps")
    ap.add_argument("--force-threshold", type=int, default=800,
                    help="per-finger touch-force rise (u16) that counts as contact")
    ap.add_argument("--fingers-needed", type=int, default=2, help="fingers in contact to call it a grip")
    ap.add_argument("--hold-timeout", type=float, default=120.0, help="max grip hold before auto-release")
    ap.add_argument("--arm-reached-file", default="/tmp/arm_reached")
    ap.add_argument("--gripped-file", default="/tmp/sheet_gripped")
    ap.add_argument("--draw-done-file", default="/tmp/draw_done")
    ap.add_argument("--no-wait", action="store_true", help="grip immediately (skip the arm-reached wait)")
    ap.add_argument("--dry", action="store_true", help="read-only: print touch/proximity, NO finger motion")
    args = ap.parse_args()

    try:
        br = Bridge(args.host, args.port)
    except OSError as e:
        raise SystemExit(f"[grip] cannot reach brainco_bridge at {args.host}:{args.port} ({e}). "
                         "Start the bridge first (brainco_touch).")

    try:
        st = br.get()
        print(f"[grip] left_touch_ok={st.get('left_touch_ok')}  "
              f"left_touch_force={st.get('left_touch_force')}  left_proximity={st.get('left_proximity')}")
        if args.dry:
            print("[grip:dry] OK (no finger motion commanded)")
            return

        if not args.no_wait and not _wait_for(args.arm_reached_file, args.hold_timeout, print):
            return

        base = list(st.get("left_touch_force") or [0] * 5)
        print(f"[grip] baseline touch force = {base}")
        gripped = False
        g = 0.0
        while g < args.grip_max - 1e-6:
            g = min(args.grip_max, g + args.grip_step)
            br.set_left([g] * 6)
            time.sleep(args.step_pause)
            force = list(br.get().get("left_touch_force") or [0] * 5)
            fired = [i for i in range(min(len(force), len(base))) if force[i] - base[i] > args.force_threshold]
            print(f"[grip] close={g:.2f}  force={force}  contact_fingers={fired}")
            if len(fired) >= args.fingers_needed:
                gripped = True
                break
        print(f"[grip] {'GRIP established (sensor contact)' if gripped else 'closed to grip-max (no clear contact — best-effort)'}")

        open(args.gripped_file, "w").close()
        print(f"[grip] signaled {args.gripped_file} — arm will draw. Holding grip until {args.draw_done_file}")
        t_end = time.time() + args.hold_timeout
        while not os.path.exists(args.draw_done_file):
            if time.time() > t_end:
                print("[grip] hold timeout — releasing")
                break
            time.sleep(0.1)

        br.set_left([0.0] * 6)
        print("[grip] released (fingers open)")
    finally:
        br.close()


if __name__ == "__main__":
    main()
