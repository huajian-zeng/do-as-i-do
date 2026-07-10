"""Stage B (rsviewer env, needs pyserial): stream Inspire hardware commands to
the real RH56 hand over RS485.

Reads inspire_cmd_*.npz (from extract_inspire_cmd.py) and sends each frame's
6-DOF command to the hand via the InspireHand driver, paced at the trajectory
fps (optionally scaled). THIS MOVES THE HAND — keep it clear of obstacles.

    conda activate rsviewer
    python stream_inspire.py --cmd inspire_cmd_whisking.npz [--speed 1.0] [--dry-run]
"""
import argparse
import sys
import time
import numpy as np

# InspireHand driver lives in the standalone project created earlier
sys.path.insert(0, "/home/huajianzeng/project/inspire_hand")
from inspire_hand import InspireHand, FINGERS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmd", required=True, help="inspire_cmd_*.npz from extract_inspire_cmd.py")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--hand-speed", type=int, default=800, help="RH56 SPEED_SET 0..1000")
    ap.add_argument("--dry-run", action="store_true", help="print frames, do not open serial")
    args = ap.parse_args()

    d = np.load(args.cmd)
    cmd = d["cmd"]                                  # (H,6) order matches FINGERS
    fps = float(d["fps"]) * args.speed
    dt = 1.0 / fps
    H = cmd.shape[0]
    print(f"{H} frames @ {fps:.1f} fps, order={list(d['joint_names'])}")
    assert list(d["joint_names"]) == FINGERS, f"order mismatch: {d['joint_names']} vs {FINGERS}"

    if args.dry_run:
        for i in range(0, H, max(1, H // 10)):
            print(f"frame {i:4d}: {cmd[i].tolist()}")
        print("(dry-run: nothing sent)")
        return

    with InspireHand(args.port, args.baud, args.id) as h:
        h.set_speed([args.hand_speed] * 6)
        # ease into the first pose before playing at speed
        h.set_angle(cmd[0].tolist())
        time.sleep(1.0)
        print("streaming... (Ctrl-C to stop)")
        t0 = time.time()
        for i in range(H):
            h.set_angle(cmd[i].tolist())
            target = t0 + (i + 1) * dt
            now = time.time()
            if target > now:
                time.sleep(target - now)
        print("done. actual angles:", h.read("ANGLE_ACT"))


if __name__ == "__main__":
    main()
