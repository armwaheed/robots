"""Sensor-gated LEFT-hand grip on a bedsheet, coordinated with the arm-overlay pull.

Talks to the running ``brainco_bridge`` (TCP JSON on 127.0.0.1:9877) — NOT the serial port directly.
  * send  {"cmd":"set","left":[6 floats 0-1]}   0=open 1=closed
  * query {"cmd":"get"}  → per-finger left_touch_force [5 u16], left_proximity [5 u16], left_touch_ok

Sequence (sentinel handshake with g1_bed_pull_v1.py):
  1. wait for ``--arm-reached-file`` (the arm has reached the sheet and is holding)
  2. oppose the thumb laterally into a CLAW (``thumb_aux``→``--thumb-claw``) and let it settle, then
     close the four fingers + thumb_curl in ONE smooth continuous flex (ramped over ``--close-s``) into
     the opposed thumb, gating on the touch force; stop on contact (a grip), else close to ``--grip-max``
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

from bed_grasp_confirm import GraspConfirmMonitor


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
    ap.add_argument("--close-s", type=float, default=1.5,
                    help="seconds to flex from open to grip-max — one smooth continuous close")
    ap.add_argument("--rate-hz", type=float, default=50.0, help="finger position command rate (Hz)")
    ap.add_argument("--thumb-claw", type=float, default=1.0,
                    help="thumb_aux opposition (0=slap/flat, 1=thumb opposed across palm = CLAW). The "
                         "thumb is moved here FIRST so the digits close into the opposed thumb (you "
                         "cannot pinch thin fabric with a flat hand).")
    ap.add_argument("--claw-settle-s", type=float, default=0.6,
                    help="seconds to let the thumb travel to the claw position BEFORE closing the digits")
    ap.add_argument("--force-threshold", type=int, default=40,
                    help="per-finger touch-force rise (u16) that counts as contact (soft fabric reads "
                         "low — the old 800 never fired; observed grabs are ~10-40)")
    ap.add_argument("--fingers-needed", type=int, default=1, help="fingers in contact to call it a grip")
    ap.add_argument("--hold-timeout", type=float, default=120.0, help="max grip hold before auto-release")
    ap.add_argument("--arm-reached-file", default="/tmp/arm_reached")
    ap.add_argument("--gripped-file", default="/tmp/sheet_gripped")
    ap.add_argument("--draw-done-file", default="/tmp/draw_done")
    # Empty-grab confirmation (did the claw close on FABRIC or on NOTHING?). See bed_grasp_confirm.py.
    ap.add_argument("--empty-file", default="/tmp/grab_empty",
                    help="sentinel written when the close is judged an EMPTY grab — the ask-the-human "
                         "trigger for 'closed on nothing' (only written under --require-fabric)")
    ap.add_argument("--require-fabric", action="store_true",
                    help="GATE the draw on fabric-in-hand: on an empty-grab verdict, write --empty-file, "
                         "skip --gripped-file, release. Default is measure-only (log the verdict, always "
                         "proceed) — calibrate the thresholds on hardware FIRST (analog of --abort-on-stall).")
    ap.add_argument("--confirm-force-rise", type=float, default=15.0,
                    help="per-finger touch-force rise (u16) over the open-claw baseline that counts as "
                         "fabric (LOWER than --force-threshold: a real soft-fabric grab reads ~10-40)")
    ap.add_argument("--confirm-prox-dev", type=float, default=150.0,
                    help="per-finger |proximity - baseline| (u16) that counts as something-in-claw "
                         "(FIRST-PASS — calibrate empty vs fabric on hardware)")
    ap.add_argument("--confirm-fingers", type=int, default=1,
                    help="fingers showing fabric (touch OR proximity) to confirm a grab")
    ap.add_argument("--no-wait", action="store_true", help="grip immediately (skip the arm-reached wait)")
    ap.add_argument("--present-claw", action="store_true",
                    help="show the OPEN claw (thumb opposed, fingers open) while waiting for the "
                         "arm-reached signal — for the human handoff: the hand is ready to receive the "
                         "sheet, then closes on the signal.")
    ap.add_argument("--dry", action="store_true", help="read-only: print touch/proximity, NO finger motion")
    args = ap.parse_args()

    try:
        br = Bridge(args.host, args.port)
    except OSError as e:
        raise SystemExit(f"[grip] cannot reach brainco_bridge at {args.host}:{args.port} ({e}). "
                         "Start the bridge first (brainco_touch).")

    # Clear OUR own coordination sentinels up front so a standalone/repeat run can't act on a stale
    # gripped/draw-done from a previous run (the arm-side script clears them too; this makes grip
    # safe to run on its own).
    for f in (args.gripped_file, args.draw_done_file, args.empty_file):
        try:
            os.remove(f)
        except OSError:
            pass

    try:
        st = br.get()
        print(f"[grip] left_touch_ok={st.get('left_touch_ok')}  "
              f"left_touch_force={st.get('left_touch_force')}  left_proximity={st.get('left_proximity')}")
        if args.dry:
            print("[grip:dry] OK (no finger motion commanded)")
            return

        if args.present_claw:                               # handoff: show the open claw while waiting
            claw0 = max(0.0, min(1.0, args.thumb_claw))
            print(f"[grip] presenting OPEN claw (thumb_aux={claw0:.2f}, fingers open) — place the sheet")
            br.set_left([0.0, claw0, 0.0, 0.0, 0.0, 0.0])

        if not args.no_wait and not _wait_for(args.arm_reached_file, args.hold_timeout, print):
            return

        base = list(st.get("left_touch_force") or [0] * 5)
        print(f"[grip] baseline touch force = {base}")

        # Motor order for set/get is [thumb_curl, thumb_aux, index, middle, ring, pinky]; thumb_aux is
        # the LATERAL thumb (0=slap/flat, 1=opposed across the palm = claw). You cannot pinch thin
        # fabric with a flat hand, so FIRST move the thumb laterally into the claw and let it settle,
        # THEN close only the four fingers + thumb_curl into the opposed thumb (thumb_aux held).
        claw = max(0.0, min(1.0, args.thumb_claw))

        def cmd(g):  # finger close fraction g, with the thumb held in the claw (thumb_aux=claw)
            return [g, claw, g, g, g, g]

        print(f"[grip] pre-positioning thumb → claw (thumb_aux={claw:.2f}), settling {args.claw_settle_s:.1f}s")
        br.set_left(cmd(0.0))                              # fingers open, thumb opposed
        time.sleep(args.claw_settle_s)

        # Empty-grab confirmation: baseline the OPEN claw now (thumb opposed, fingers open — nothing
        # loaded yet), then watch touch + proximity through the close to decide fabric-vs-air.
        gc = GraspConfirmMonitor(force_rise=args.confirm_force_rise, prox_dev=args.confirm_prox_dev,
                                 fingers_needed=args.confirm_fingers)
        gc.baseline(br.get())

        # One smooth, continuous flex: ramp the commanded close fraction at a fixed rate so the
        # fingers move in a single motion (the old stepped setpoints + per-step pauses caused the
        # intermittent flexion). Position is commanded every tick; the touch force is polled ~10 Hz
        # so contact still stops the close early, just as before.
        def _fired(force):
            return [i for i in range(min(len(force), len(base))) if force[i] - base[i] > args.force_threshold]

        gripped = False
        g = 0.0
        force = base
        dt = 1.0 / max(1.0, args.rate_hz)
        speed = args.grip_max / max(1e-3, args.close_s)   # close fraction per second
        t0 = time.time()
        next_poll = 0.0
        last_report = -1.0
        while True:
            elapsed = time.time() - t0
            g = min(args.grip_max, speed * elapsed)
            br.set_left(cmd(g))                            # thumb stays opposed; fingers ramp closed
            if elapsed >= next_poll:                       # poll touch ~10 Hz (don't gate every tick)
                reading = br.get()
                gc.update(reading)                         # feed the empty-grab confirm (touch + proximity)
                force = list(reading.get("left_touch_force") or [0] * 5)
                next_poll = elapsed + 0.1
                fired = _fired(force)
                if elapsed - last_report > 0.2:
                    print(f"[grip] close={g:.2f}  force={force}  contact_fingers={fired}")
                    last_report = elapsed
                if len(fired) >= args.fingers_needed:
                    gripped = True
                    break
            if g >= args.grip_max - 1e-6:                  # reached max — final force check, then stop
                reading = br.get()
                gc.update(reading)
                force = list(reading.get("left_touch_force") or [0] * 5)
                gripped = len(_fired(force)) >= args.fingers_needed
                print(f"[grip] close={g:.2f}  force={force}  contact_fingers={_fired(force)}")
                break
            time.sleep(dt)
        print(f"[grip] {'GRIP established (sensor contact)' if gripped else 'closed to grip-max (no clear contact — best-effort)'}")

        # Empty-grab verdict — fabric-in-hand or closed-on-nothing? The pull-failure detector is BLIND
        # to this (a draw on air reads perfectly free), so confirm it here before signaling the draw.
        gv = gc.verdict()
        print(f"[grip] GRASP VERDICT: {gc.summary()}")
        if args.require_fabric and not gv["fabric_present"]:
            # Enforced: do NOT signal the draw on an empty hand. Write the empty-grab sentinel (the
            # ask-the-human trigger), release, and exit — the arm-side script breaks its grip-wait on
            # this file and skips the draw, and the orchestrator escalates exactly like /tmp/pull_failed.
            try:
                open(args.empty_file, "w").close()
            except OSError:
                pass
            print(f"[grip] ⚠ EMPTY GRAB — closed on nothing. wrote {args.empty_file}; NOT signaling the "
                  f"draw. THIS is the trigger to ask the human for help.")
            br.set_left([0.0] * 6)
            print("[grip] released (fingers open)")
            return
        if not gv["fabric_present"]:
            print("[grip] (measure-only: empty-grab verdict logged but NOT enforced — pass "
                  "--require-fabric once thresholds are calibrated)")

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
