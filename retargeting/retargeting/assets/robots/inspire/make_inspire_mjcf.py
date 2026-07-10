"""Generate a do-as-i-do-compatible MJCF for the Inspire RH56 right hand
from the dex-urdf inspire_hand_right.urdf.

Produces a hand that plugs into the retargeting pipeline unchanged:
  - 6-DOF floating base named right_pos_{x,y,z} / right_rot_{x,y,z}
    (matches the sharpa convention so solve_ik / generate_scene work as-is)
  - the 12-joint RH56 finger tree (6 actuated + 6 mimic) with right_ prefix
  - equality constraints reproducing the URDF <mimic> couplings
  - sites right_palm and right_{finger}_tip  (the ONLY correspondence solve_ik needs)
  - position actuators on the 6 base DOF + 6 actuated finger DOF
  - mesh collision geoms named collision_hand_right_* for pair generation

URDF rpy is extrinsic-xyz -> scipy Rotation.from_euler('xyz', ...) matches it.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "inspire_hand_right.urdf")
OUT = os.path.join(HERE, "right.xml")

SIDE = "right"

# link -> collision obj (per URDF collision refs; several reuse index meshes)
LINK_MESH = {
    "thumb_proximal_base": "right_thumb_proximal_base",
    "thumb_proximal": "right_thumb_proximal",
    "thumb_intermediate": "right_thumb_intermediate",
    "thumb_distal": "right_thumb_distal",
    "index_proximal": "right_index_proximal",
    "index_intermediate": "right_index_intermediate",
    "middle_proximal": "right_index_proximal",
    "middle_intermediate": "right_middle_intermediate",
    "ring_proximal": "right_index_proximal",
    "ring_intermediate": "right_index_intermediate",
    "pinky_proximal": "right_index_proximal",
    "pinky_intermediate": "right_pinky_intermediate",
}
# fingertip sites: parent link -> (site_name, xyz) from URDF fixed tip joints
TIP_SITES = {
    "thumb_distal": (f"{SIDE}_thumb_tip", (0.0202, 0.0140, -0.006)),
    "index_intermediate": (f"{SIDE}_index_tip", (-0.0008, 0.045, -0.005)),
    "middle_intermediate": (f"{SIDE}_middle_tip", (-0.001, 0.048, -0.005)),
    "ring_intermediate": (f"{SIDE}_ring_tip", (-0.0008, 0.045, -0.005)),
    "pinky_intermediate": (f"{SIDE}_pinky_tip", (-0.0008, 0.037, -0.005)),
}
ACTUATED = ["thumb_proximal_yaw", "thumb_proximal_pitch",
            "index_proximal", "middle_proximal", "ring_proximal", "pinky_proximal"]
# equality couplings: dependent_joint -> (driver_joint, mult, offset)
MIMIC = {
    "thumb_intermediate": ("thumb_proximal_pitch", 1.334, 0.0),
    "thumb_distal": ("thumb_proximal_pitch", 0.667, 0.0),
    "index_intermediate": ("index_proximal", 1.06399, -0.04545),
    "middle_intermediate": ("middle_proximal", 1.06399, -0.04545),
    "ring_intermediate": ("ring_proximal", 1.06399, -0.04545),
    "pinky_intermediate": ("pinky_proximal", 1.06399, -0.04545),
}

tree = ET.parse(URDF)
root = tree.getroot()
joints = {j.get("name"): j for j in root.findall("joint")}
links = {l.get("name"): l for l in root.findall("link")}

# build child map: parent_link -> list of joints
children = {}
for j in root.findall("joint"):
    p = j.find("parent").get("link")
    children.setdefault(p, []).append(j)


def origin_of(j):
    o = j.find("origin")
    xyz = np.fromstring(o.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ") if o is not None else np.zeros(3)
    quat_xyzw = R.from_euler("xyz", rpy).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    return xyz, quat_wxyz


def fmt(a):
    return " ".join(f"{v:.6g}" for v in a)


def emit_link_body(link_name, joint, indent):
    """Recursively emit an MJCF <body> for link_name attached via `joint`."""
    pad = "  " * indent
    xyz, quat = origin_of(joint)
    lines = [f'{pad}<body name="{SIDE}_{link_name}" pos="{fmt(xyz)}" quat="{fmt(quat)}">']
    # inertial
    li = links[link_name].find("inertial")
    if li is not None:
        ixyz = np.fromstring(li.find("origin").get("xyz", "0 0 0"), sep=" ")
        mass = li.find("mass").get("value")
        ic = li.find("inertia")
        lines.append(f'{pad}  <inertial pos="{fmt(ixyz)}" mass="{mass}" '
                     f'diaginertia="{ic.get("ixx")} {ic.get("iyy")} {ic.get("izz")}" />')
    # joint (revolute only; fixed handled by not emitting a joint)
    if joint.get("type") == "revolute":
        axis = joint.find("axis").get("xyz")
        lim = joint.find("limit")
        lo, hi = lim.get("lower"), lim.get("upper")
        jname = f'{SIDE}_{joint.get("name").replace("_joint", "")}'
        lines.append(f'{pad}  <joint name="{jname}" pos="0 0 0" axis="{axis}" '
                     f'range="{lo} {hi}" />')
    # collision + visual geom
    if link_name in LINK_MESH:
        mesh = LINK_MESH[link_name]
        gname = f"collision_hand_{SIDE}_{link_name}"
        lines.append(f'{pad}  <geom type="mesh" mesh="{mesh}" name="{gname}" '
                     f'group="3" rgba="0.75 0.78 0.92 1" />')
        lines.append(f'{pad}  <geom type="mesh" mesh="{mesh}" contype="0" conaffinity="0" '
                     f'group="1" density="0" rgba="0.75 0.78 0.92 1" />')
    # fingertip site
    if link_name in TIP_SITES:
        sname, spos = TIP_SITES[link_name]
        lines.append(f'{pad}  <site name="{sname}" pos="{fmt(spos)}" />')
    # recurse to children
    for cj in children.get(link_name, []):
        if cj.get("type") == "fixed":
            continue  # tip frames captured as sites above
        lines += emit_link_body(cj.find("child").get("link"), cj, indent + 1)
    lines.append(f'{pad}</body>')
    return lines


# --- assemble ---
meshes = sorted(set(LINK_MESH.values()))
asset_lines = [f'    <mesh name="{m}" file="{m}.obj" />' for m in meshes]

# hand_base_link is attached to `base` via fixed base_joint; bake its transform
base_joint = joints["base_joint"]
hb_xyz, hb_quat = origin_of(base_joint)

# palm collision primitives from URDF hand_base_link
palm_geoms = [
    '        <geom type="cylinder" size="0.028 0.0139" pos="0 -0.0136 0" '
    'euler="1.5708 0 0" name="collision_hand_right_palm_0" group="3" rgba="0.6 0.6 0.7 1" />',
    '        <geom type="box" size="0.01955 0.0101 0.0287" pos="-0.0032 -0.038 0" '
    'name="collision_hand_right_palm_1" group="3" rgba="0.6 0.6 0.7 1" />',
    '        <geom type="box" size="0.01955 0.02 0.0407" pos="-0.0032 -0.0682 0" '
    'name="collision_hand_right_palm_2" group="3" rgba="0.6 0.6 0.7 1" />',
]

# finger subtrees: children of hand_base_link that are revolute
finger_lines = []
for cj in children.get("hand_base_link", []):
    if cj.get("type") == "revolute":
        finger_lines += emit_link_body(cj.find("child").get("link"), cj, 4)

# actuators
base_act = []
for ax in ["pos_x", "pos_y", "pos_z", "rot_x", "rot_y", "rot_z"]:
    base_act.append(f'    <position name="{SIDE}_{ax}_position" joint="{SIDE}_{ax}" kp="1000" />')
finger_act = [f'    <position name="{SIDE}_{a}_position" joint="{SIDE}_{a}" />' for a in ACTUATED]

# equality couplings
eq_lines = []
for dep, (drv, mult, off) in MIMIC.items():
    eq_lines.append(f'    <joint joint1="{SIDE}_{dep}" joint2="{SIDE}_{drv}" '
                    f'polycoef="{off} {mult} 0 0 0" />')

xml = f'''<mujoco model="inspire_right">
  <compiler angle="radian" meshdir="meshes/" autolimits="true" />
  <default>
    <geom density="800" condim="1" contype="0" conaffinity="0" />
    <position kp="300" dampratio="1.0" inheritrange="1" />
    <joint damping="0.0" armature="1.0" frictionloss="0.0" />
    <site size="0.005" type="sphere" rgba="1 0 0 1" group="3" />
  </default>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1" width="512" height="3072" />
    <texture type="2d" name="right_groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="right_groundplane" texture="right_groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2" />
{chr(10).join(asset_lines)}
  </asset>
  <worldbody>
    <body name="{SIDE}_base_tx">
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01" />
      <joint name="{SIDE}_pos_x" pos="0 0 0" axis="1 0 0" type="slide" range="-5 5" />
      <body name="{SIDE}_base_ty">
        <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01" />
        <joint name="{SIDE}_pos_y" pos="0 0 0" axis="0 1 0" type="slide" range="-5 5" />
        <body name="{SIDE}_base_tz">
          <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01" />
          <joint name="{SIDE}_pos_z" pos="0 0 0" axis="0 0 1" type="slide" range="-5 5" />
          <body name="{SIDE}_base_roll">
            <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01" />
            <joint name="{SIDE}_rot_x" pos="0 0 0" axis="1 0 0" type="hinge" range="-6.28 6.28" />
            <body name="{SIDE}_base_pitch">
              <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01" />
              <joint name="{SIDE}_rot_y" pos="0 0 0" axis="0 1 0" type="hinge" range="-6.28 6.28" />
              <body name="{SIDE}_base_yaw">
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.01 0.01 0.01" />
                <joint name="{SIDE}_rot_z" pos="0 0 0" axis="0 0 1" type="hinge" range="-6.28 6.28" />
                <body name="{SIDE}_hand_base_link" pos="{fmt(hb_xyz)}" quat="{fmt(hb_quat)}">
                  <inertial pos="-0.0025264 -0.066047 0.0019598" mass="0.14143" diaginertia="0.00012281 8.3832e-05 7.6663e-05" />
                  <site name="{SIDE}_palm" pos="0 -0.08 0" />
{chr(10).join(palm_geoms)}
{chr(10).join(finger_lines)}
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
{chr(10).join(eq_lines)}
  </equality>
  <actuator>
{chr(10).join(base_act)}
{chr(10).join(finger_act)}
  </actuator>
</mujoco>
'''

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(xml)
print("wrote", OUT, len(xml.splitlines()), "lines")

# validate
import mujoco
m = mujoco.MjModel.from_xml_path(OUT)
print(f"compiled OK: nq={m.nq} nv={m.nv} nu={m.nu} nsite={m.nsite} neq={m.neq}")
for s in [f"{SIDE}_palm"] + [f"{SIDE}_{f}_tip" for f in ["thumb","index","middle","ring","pinky"]]:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s)
    print(f"  site {s}: id={sid}")
print("actuated joints nu:", m.nu)
