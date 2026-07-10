# Porting do-as-i-do to the Inspire RH56 hand (single right hand, hardware replay)

This documents how the sharpa-only pipeline was extended to retarget reconstructed
hand-object demos onto an **Inspire RH56** 6-DOF dexterous hand, and how the
resulting trajectory is streamed to the **physical** right hand on `/dev/ttyUSB0`.

Scope of this port (per the current hardware setup):
- **single right hand**, no arm — the hand is fixed on the desk, so the 6-DOF
  wrist/base trajectory is **ignored** and only finger flexion is reproduced.
- RH56 model sourced from open source (dex-urdf), converted to MJCF.

## TL;DR — how the pipeline chooses a hand

The retargeting pipeline is **model-driven**: `robot_type` selects an asset
directory `retargeting/assets/robots/{robot_type}/{embodiment}.xml`, and the IK
(`solve_ik.py`) targets sites **by name** — only `{side}_palm` (wrist, pos+ori)
and `{side}_{finger}_tip` for thumb/index/middle/ring/pinky (fingertip pos).
So adding a hand is almost entirely **"add a correctly-named MJCF"** — no
per-hand branching in the pipeline. The one exception we had to patch is the
IK rollout's fully-actuated assumption (see below).

## What was added / changed

### 1. New robot model — `retargeting/retargeting/assets/robots/inspire/right.xml`
Generated from dex-urdf `inspire_hand_right.urdf`
(<https://github.com/dexsuite/dex-urdf>, `robots/hands/inspire_hand`) by
`scratchpad/make_inspire_mjcf.py`. It contains:
- a 6-DOF floating base named `right_pos_{x,y,z}` / `right_rot_{x,y,z}`
  (same convention as sharpa, so `solve_ik`/`generate_scene` work unchanged);
- the RH56 finger tree: **6 actuated joints** + **6 coupled (mimic) joints**
  reproduced as MuJoCo `<equality joint>` constraints;
- sites `right_palm` and `right_{finger}_tip` (the correspondence the IK needs);
- position actuators on the 6 base DOF + 6 actuated finger joints (`nu=12`);
- mesh collision geoms `collision_hand_right_*` (collision `.obj` meshes copied
  into `assets/robots/inspire/meshes/`; the visual `.glb` are unused — MuJoCo
  can't load glTF).

Model stats: `nq=18` (6 base + 12 finger), `nu=12`, 6 sites, 6 equality couplings.
FK check confirms fingers extend when open and curl inward on flexion.

> Only `right.xml` exists. `left.xml` / `bimanual.xml` were not needed for a
> single right hand. For `right` embodiment `generate_scene` loads `right.xml`.

Actuated joint → Inspire hardware DOF:

| Inspire DOF (hardware order) | MJCF actuated joint         | range (rad)   |
|------------------------------|-----------------------------|---------------|
| pinky                        | `right_pinky_proximal`      | 0 .. 1.47     |
| ring                         | `right_ring_proximal`       | 0 .. 1.47     |
| middle                       | `right_middle_proximal`     | 0 .. 1.47     |
| index                        | `right_index_proximal`      | 0 .. 1.47     |
| thumbBend                    | `right_thumb_proximal_pitch`| 0 .. 0.6      |
| thumbRot                     | `right_thumb_proximal_yaw`  | 0 .. 1.308    |

### 2. Pipeline fix — underactuated hands in `solve_ik.py`
The IK rollout set `ctrl = qpos[: nq - nq_obj]`, assuming **joints == actuators**
(true for sharpa's 22 = 22, false for inspire's 18 joints vs 12 actuators). It now
maps each actuator to the qpos slot of the joint it drives
(`model.actuator_trnid` → `jnt_qposadr`), which is correct for both fully- and
under-actuated hands. This is the **only** code change; sharpa is unaffected.

### 3. Deployment (hand-only finger replay) — `deployment/inspire_replay/`
Two stages, split so the MuJoCo dependency and the serial dependency live in
different envs:
- `extract_inspire_cmd.py` (**retargeting** env): reads `scene_ik.xml` +
  `trajectory_kinematic.npz`, reads each actuated joint's qpos column and range,
  maps radians → Inspire `0..1000` (1000 = extended limit / open, 0 = flexed /
  closed), saves `inspire_cmd_*.npz` = `cmd (H,6) uint16` + `fps`.
- `stream_inspire.py` (**rsviewer** env): streams those commands to the real hand
  via the `InspireHand` driver at `~/project/inspire_hand`, paced at `fps`.

## How to run (end-to-end)

```bash
# 0. one-time: create the retargeting env
conda create -y -n retargeting python=3.12
conda activate retargeting
cd ~/project/do-as-i-do/retargeting && pip install -e .

# 1. retarget onto the Inspire hand (stages 1-4; skips GPU physics MPC)
python run_inspire_ik.py --task whisking --raw-dir ../reconstruction/whisking
#    -> outputs/inspire/right/whisking/0/trajectory_kinematic.npz

# 2. convert the IK trajectory to Inspire 0..1000 commands
cd ../deployment/inspire_replay
python extract_inspire_cmd.py \
    --scene ../../retargeting/outputs/inspire/right/whisking/0/scene_ik.xml \
    --traj  ../../retargeting/outputs/inspire/right/whisking/0/trajectory_kinematic.npz \
    --out   inspire_cmd_whisking.npz

# 3a. sanity check without touching hardware
conda activate rsviewer
python stream_inspire.py --cmd inspire_cmd_whisking.npz --dry-run

# 3b. THIS MOVES THE HAND — keep it clear of obstacles
python stream_inspire.py --cmd inspire_cmd_whisking.npz --speed 0.5
```

`run_inspire_ik.py` stops before stage 5 (physics MPC, needs a GPU +
mujoco-warp). For hand-only finger replay the kinematic trajectory is enough.
To run the full pipeline incl. physics, use `launch.py --robot-type inspire`.

## Validation status (whisking demo)

- Pipeline stages 1–4 run clean for `--robot-type inspire`; 137 frames of IK.
- Fingertip tracking error vs the MANO targets averaged **~42 mm**
  (middle 13 mm best; index 62 mm / pinky 58 mm worst). The `whisk` object is
  ~19 cm; error is dominated by **hand-morphology/scale mismatch** — the RH56 is
  smaller than a human hand and each finger has a single coupled flexion DOF, so
  absolute fingertip positions can't fully match. The **flexion pattern** is
  captured. `extract` showed index saturating near full-flex and thumb-rotation
  pinned, consistent with unreachable targets.
- `extract` + `stream --dry-run` validated: 135 frames @ 50 fps, 6-DOF commands
  in `[0,1000]`, order `[pinky,ring,middle,index,thumbBend,thumbRot]`.

## Tuning levers if you want better tracking

- **Palm site placement** in `right.xml` (`site name="right_palm"`) biases the
  wrist alignment and thus all fingertips; nudely it sits at `0 -0.08 0`.
- **Per-hand scale**: scale the MANO fingertip targets toward the wrist to match
  the RH56's shorter reach (dex-retargeting-style), or lower `finger_pos_cost` /
  raise `posture_cost` in `solve_ik.py` so the solver doesn't saturate joints.
- **Hardware sign/calibration**: the 1000=open / 0=closed mapping in
  `extract_inspire_cmd.py` (`--open-cmd/--close-cmd`) should be confirmed against
  the real hand (read `ANGLE_ACT` at known open/closed poses).

## Related

- Inspire driver + protocol: `~/project/inspire_hand` (RS485, 115200, id 1;
  reply header is reversed `90 EB`).
- MJCF generator: `scratchpad/make_inspire_mjcf.py` (re-run to regenerate/left).
