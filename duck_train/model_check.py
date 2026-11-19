"""Load the upstream Microduck walking model headless and print a sanity summary.

Run from the repo root:
    uv run --project third_party/microduck_rl python duck_train/model_check.py
"""
from pathlib import Path

import mujoco

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "microduck_rl"
XML = UPSTREAM / "src/mjlab_microduck/robot/microduck/robot_walk.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    hinge = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        for j in range(model.njnt)
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
    ]
    actuators = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in range(model.nu)]
    print(f"model:           {XML.relative_to(UPSTREAM)}")
    print(f"joints (total):  {model.njnt}  (non-free: {len(hinge)})")
    print(f"actuators:       {model.nu}")
    print(f"bodies:          {model.nbody}")
    print(f"total mass [kg]: {model.body_mass.sum():.4f}")
    print(f"timestep [s]:    {model.opt.timestep}")
    for i, (j, a) in enumerate(zip(hinge, actuators)):
        print(f"  {i:2d} joint={j:<16} actuator={a}")


if __name__ == "__main__":
    main()
