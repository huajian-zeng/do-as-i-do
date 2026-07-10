"""Stage A (retargeting env, needs mujoco): convert a retargeted kinematic
trajectory into per-frame Inspire RH56 hardware commands.

Hand-only replay: we ignore the 6-DOF wrist/base trajectory (the hand is fixed
on the desk) and only reproduce finger flexion. For each of the hand's 6
ACTUATED joints we read its qpos column + range from the scene model, normalize
to [0,1], and map to the Inspire 0..1000 command range (1000 = fully open at the
joint's lower/extended limit, 0 = fully closed at the upper/flexed limit).

    conda activate retargeting
    python extract_inspire_cmd.py \
        --scene ../../retargeting/outputs/inspire/right/whisking/0/scene.xml \
        --traj  ../../retargeting/outputs/inspire/right/whisking/0/trajectory_kinematic.npz \
        --out   inspire_cmd_whisking.npz

Output npz: cmd (H,6) uint16 in [0,1000], order [pinky,ring,middle,index,
thumbBend,thumbRot]; fps (float); joint_names (6,).
"""
import argparse
import numpy as np
import mujoco

# Inspire hardware DOF order  ->  MJCF actuated joint name (right hand)
HW_ORDER = [
    ("pinky",     "right_pinky_proximal"),
    ("ring",      "right_ring_proximal"),
    ("middle",    "right_middle_proximal"),
    ("index",     "right_index_proximal"),
    ("thumbBend", "right_thumb_proximal_pitch"),
    ("thumbRot",  "right_thumb_proximal_yaw"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="scene.xml or right.xml with the inspire hand")
    ap.add_argument("--traj", required=True, help="trajectory_kinematic.npz (qpos, frequency)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--open-cmd", type=int, default=1000, help="Inspire cmd at extended limit")
    ap.add_argument("--close-cmd", type=int, default=0, help="Inspire cmd at flexed limit")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    d = np.load(args.traj)
    qpos = d["qpos"]                       # (H, nq)
    fps = float(d["frequency"]) if "frequency" in d.files else 50.0
    H = qpos.shape[0]

    cmd = np.zeros((H, 6), dtype=np.uint16)
    names = []
    for k, (hw, jname) in enumerate(HW_ORDER):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise SystemExit(f"joint {jname} not found in {args.scene}")
        adr = model.jnt_qposadr[jid]
        lo, hi = model.jnt_range[jid]
        q = np.clip(qpos[:, adr], lo, hi)
        frac = (q - lo) / (hi - lo)        # 0 = extended, 1 = fully flexed
        val = args.open_cmd + frac * (args.close_cmd - args.open_cmd)
        cmd[:, k] = np.clip(np.round(val), 0, 1000).astype(np.uint16)
        names.append(hw)
        print(f"{hw:10} <- {jname:28} range[{lo:.3f},{hi:.3f}] "
              f"cmd[{cmd[:,k].min()}..{cmd[:,k].max()}]")

    np.savez(args.out, cmd=cmd, fps=np.float64(fps), joint_names=np.array(names))
    print(f"\nsaved {args.out}: {H} frames @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
