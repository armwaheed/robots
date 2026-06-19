"""Calibrate the empty-grab confirm thresholds on hardware — turn the manual procedure into one command.

`bed_grasp_confirm.py` ships with FIRST-PASS thresholds (--confirm-force-rise ~15, --confirm-prox-dev
~150) that MUST be calibrated before --require-fabric is trusted to gate the live demo. The documented
procedure was "close on air ~5×, close on sheet ~5×, eyeball the logged peaks, pick a split" — manual
and easy to get wrong, and it never actually confirmed proximity is usable at all. This script does it:

  * runs N closes on an EMPTY claw and N closes on a REAL sheet (prompting you to set each up),
  * captures, per trial, the strongest-finger peak touch-force rise and peak proximity deviation
    (the exact quantities GraspConfirmMonitor's verdict keys on, max across fingers, fingers_needed=1),
  * reports the air-vs-sheet clusters and RECOMMENDS a --confirm-force-rise and --confirm-prox-dev set
    midway between max(air) and min(sheet) — and tells you when a signal does NOT separate (clusters
    overlap) or proximity is dead (so you know to fall back to touch-only or to vision).

It deliberately does a FULL close every trial (no touch-gated early stop, unlike bed_grip_v1) so the
peaks are captured consistently across trials. It commands only the LEFT hand via the same
brainco_bridge; no arm/leg motion. Bias of the recommendation: leave a margin BELOW min(sheet) so a
slightly weaker real grab still reads as fabric (we would rather occasionally ask for help on a real
grab than ever claim a grab we don't have).

  python bed_grasp_calibrate.py --air 5 --sheet 5
  # then, if a signal separates cleanly:
  python bed_grip_v1.py --require-fabric --confirm-force-rise <F> --confirm-prox-dev <P> ...
"""
from __future__ import annotations

import argparse
import time

from bed_grasp_confirm import GraspConfirmMonitor
from bed_grip_v1 import Bridge


def close_and_measure(br, *, grip_max, close_s, rate_hz, thumb_claw, claw_settle_s, log=print):
    """Present the open claw, baseline, ramp the fingers FULLY closed once, return the peak verdict."""
    claw = max(0.0, min(1.0, thumb_claw))

    def cmd(g):  # [thumb_curl, thumb_aux, index, middle, ring, pinky]; thumb held opposed
        return [g, claw, g, g, g, g]

    br.set_left(cmd(0.0))                                  # fingers open, thumb opposed into the claw
    time.sleep(claw_settle_s)
    gc = GraspConfirmMonitor()
    gc.baseline(br.get())                                  # OPEN-claw baseline (nothing loaded yet)

    dt = 1.0 / max(1.0, rate_hz)
    speed = grip_max / max(1e-3, close_s)
    t0 = time.time()
    next_poll = 0.0
    while True:
        elapsed = time.time() - t0
        g = min(grip_max, speed * elapsed)
        br.set_left(cmd(g))
        if elapsed >= next_poll:
            gc.update(br.get())
            next_poll = elapsed + 0.1
        if g >= grip_max - 1e-6:
            gc.update(br.get())
            break
        time.sleep(dt)
    br.set_left([0.0] * 6)                                 # release
    v = gc.verdict()
    # the verdict keys on the STRONGEST finger (fingers_needed=1), so cluster on the max across fingers
    return {"force": max(v["peak_force_rise"]), "prox": max(v["peak_prox_dev"]),
            "prox_usable": v["prox_usable"], "raw": v}


def recommend_threshold(air, sheet, *, margin_frac=0.25, floor=1.0):
    """Pick a threshold that separates the air cluster (low) from the sheet cluster (high).

    Returns (threshold, separates, note). The threshold sits below min(sheet) by `margin_frac` of the
    gap to max(air) — biased toward calling marginal grabs FABRIC (min(sheet) side), since a false
    "empty" only costs an unnecessary ask-for-help while a false "fabric" is the failure we are killing.
    `separates` is False when the clusters overlap (max(air) >= min(sheet)) — that signal can't gate.
    """
    if not air or not sheet:
        return None, False, "no samples"
    hi_air, lo_sheet = max(air), min(sheet)
    if hi_air >= lo_sheet:
        return None, False, (f"OVERLAP: max(air)={hi_air:.1f} >= min(sheet)={lo_sheet:.1f} — this signal "
                             f"does NOT separate empty from fabric")
    gap = lo_sheet - hi_air
    thr = max(floor, lo_sheet - margin_frac * gap)
    return round(thr, 1), True, (f"clean gap {gap:.1f} (air≤{hi_air:.1f}, sheet≥{lo_sheet:.1f})")


def _fmt(vals):
    return "[" + ", ".join(f"{v:.1f}" for v in vals) + "]" if vals else "[]"


def _run_set(br, label, n, args, auto):
    out = []
    for i in range(n):
        if not auto:
            input(f"\n  >> {label} trial {i + 1}/{n}: set up the claw ({label}), then press Enter…")
        m = close_and_measure(br, grip_max=args.grip_max, close_s=args.close_s, rate_hz=args.rate_hz,
                              thumb_claw=args.thumb_claw, claw_settle_s=args.claw_settle_s)
        print(f"     trial {i + 1}: force_rise={m['force']:.1f}  prox_dev={m['prox']:.1f}"
              f"{'' if m['prox_usable'] else '  [proximity dead]'}")
        out.append(m)
        time.sleep(args.between_s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9877)
    ap.add_argument("--air", type=int, default=5, help="empty-claw closes to sample")
    ap.add_argument("--sheet", type=int, default=5, help="real-sheet closes to sample")
    ap.add_argument("--grip-max", type=float, default=0.85)
    ap.add_argument("--close-s", type=float, default=1.5)
    ap.add_argument("--rate-hz", type=float, default=50.0)
    ap.add_argument("--thumb-claw", type=float, default=1.0)
    ap.add_argument("--claw-settle-s", type=float, default=0.6)
    ap.add_argument("--between-s", type=float, default=0.5, help="pause between trials")
    ap.add_argument("--margin-frac", type=float, default=0.25,
                    help="fraction of the air→sheet gap to drop below min(sheet) for the threshold")
    ap.add_argument("--auto", action="store_true",
                    help="don't prompt between trials (you stage the sheet some other way) — for scripted runs")
    args = ap.parse_args()

    try:
        br = Bridge(args.host, args.port)
    except OSError as e:
        raise SystemExit(f"[cal] cannot reach brainco_bridge at {args.host}:{args.port} ({e}). "
                         "Start the bridge first (brainco_touch).")
    try:
        print(f"[cal] sampling {args.air} EMPTY + {args.sheet} SHEET closes — keep the bridge running.")
        air = _run_set(br, "EMPTY (nothing in the claw)", args.air, args, args.auto)
        sheet = _run_set(br, "SHEET (lay the sheet edge in the claw)", args.sheet, args, args.auto)
    finally:
        br.set_left([0.0] * 6)
        br.close()

    af, sf = [t["force"] for t in air], [t["force"] for t in sheet]
    ap_, sp = [t["prox"] for t in air], [t["prox"] for t in sheet]
    prox_usable = any(t["prox_usable"] for t in air + sheet)

    print("\n" + "═" * 72)
    print("  EMPTY-GRAB THRESHOLD CALIBRATION")
    print("═" * 72)
    print(f"  touch force_rise   air={_fmt(af)}  sheet={_fmt(sf)}")
    print(f"  proximity dev      air={_fmt(ap_)}  sheet={_fmt(sp)}"
          f"{'' if prox_usable else '   [proximity read 0 on every trial — DEAD]'}")

    f_thr, f_ok, f_note = recommend_threshold(af, sf, margin_frac=args.margin_frac)
    print(f"\n  touch-force:  {f_note}")
    if f_ok:
        print(f"    → --confirm-force-rise {f_thr}")
    if prox_usable:
        p_thr, p_ok, p_note = recommend_threshold(ap_, sp, margin_frac=args.margin_frac)
        print(f"  proximity:    {p_note}")
        if p_ok:
            print(f"    → --confirm-prox-dev {p_thr}")
    else:
        p_ok = False
        print("  proximity:    DEAD — fall back to touch-only (and consider a vision fabric-in-hand cue).")

    print("\n  verdict:")
    if f_ok or (prox_usable and p_ok):
        usable = ", ".join(s for s, ok in [("touch", f_ok), ("proximity", prox_usable and p_ok)] if ok)
        print(f"    {usable} separate(s) cleanly — safe to enable --require-fabric with the values above.")
    else:
        print("    NEITHER signal separates empty from fabric on these samples. Do NOT enable "
              "--require-fabric yet — re-check the claw geometry / sheet placement, take more samples, "
              "or move fabric-in-hand confirmation to vision.")
    print("═" * 72)


if __name__ == "__main__":
    main()
