"""Run the retargeting pipeline stages 1-4 for the Inspire hand and stop before
the GPU physics-optimization stage.

Produces outputs/inspire/right/{task}/0/trajectory_kinematic.npz, which is the
kinematic IK trajectory that the hand-only deployment adapter streams to the
real RH56 hand.

    conda activate retargeting
    python run_inspire_ik.py --task whisking --raw-dir ../reconstruction/whisking
"""
import tyro
import loguru
from dataclasses import dataclass

from retargeting.pipeline.process_dataset import main as process_dataset
from retargeting.pipeline.decompose_mesh import main as decompose_mesh
from retargeting.pipeline.generate_scene import main as generate_scene
from retargeting.pipeline.solve_ik import main as solve_ik


@dataclass
class Cfg:
    task: str = "whisking"
    raw_dir: str = "../reconstruction/whisking"
    robot_type: str = "inspire"
    hand_type: str = "right"       # single right hand (see AskUserQuestion answer)
    data_id: int = 0
    dataset_name: str = "do_as_i_do"
    output_root_dir: str = "outputs"
    force: bool = True
    add_ur3_arm: bool = False      # tabletop hand-only: no UR arm behind the palm


def main(cfg: Cfg):
    task = process_dataset(
        raw_dir=cfg.raw_dir, output_root_dir=cfg.output_root_dir, task=cfg.task,
        data_id=cfg.data_id, embodiment_type=cfg.hand_type,
        dataset_name=cfg.dataset_name, force=cfg.force,
    )
    if task is None:
        raise SystemExit("process_dataset failed")

    decompose_mesh(
        task=task, dataset_name=cfg.dataset_name, data_id=cfg.data_id,
        embodiment_type=cfg.hand_type, thicken=0.002, dilate=0.002, force=cfg.force,
    )

    generate_scene(
        task=task, dataset_name=cfg.dataset_name, data_id=cfg.data_id,
        embodiment_type=cfg.hand_type, robot_type=cfg.robot_type,
        show_viewer=False, friction_scale=1.5,
        use_pedestal=True, use_support=True, force=cfg.force,
        add_ur3_arm=cfg.add_ur3_arm,
    )

    solve_ik(
        task=task, dataset_name=cfg.dataset_name, data_id=cfg.data_id,
        embodiment_type=cfg.hand_type, robot_type=cfg.robot_type,
        show_viewer=False, force=cfg.force, smoothing=True,
    )
    loguru.logger.info("Stages 1-4 done. Kinematic trajectory ready for deployment.")


if __name__ == "__main__":
    main(tyro.cli(Cfg))
