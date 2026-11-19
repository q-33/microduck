"""Replica walking task: upstream's flat walking task with the replica's interface.

Built from upstream's make_microduck_velocity_env_cfg() (third_party/microduck_rl,
task Mjlab-Velocity-Flat-MicroDuck). Changed, and only these:

1. Policy observation: 51 entries in the order the replica's walk.py builds them.
   The head-pose (4) and body-pose (6) commands are removed, with their rewards and
   curricula. The neck/head joints stay policy outputs and join the posture reward,
   as in the replica's own training env.
2. IMU frame: the policy sees gyro and the world up-vector in the vertically mounted
   sensor's frame, as the replica runtime feeds them (see _to_sensor_frame below).
3. Servo stiffness: BAM firmware kp 125 instead of 200.
4. Supply voltage: per-env range from replica_config.PACK_VOLTAGE_RANGE_V.

All replica values live in duck_train/replica_config.py.
Importing this module registers the task (idempotent).
"""
from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.utils.lab_api.math import quat_apply

# Upstream, imported unmodified.
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROBOT_CFG
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MicroduckRlCfg,
    make_microduck_velocity_env_cfg,
)

from duck_train import replica_config as rc

_ROBOT = SceneEntityCfg("robot")


# ---------------------------------------------------------------------------
# IMU frame
# ---------------------------------------------------------------------------
# Real robot (replica runtime):
#   gyro      = raw BNO08x gyro, sensor frame, no rotation
#               (robot_controller.py:97-99 -> observer.py:78 -> walk.py:131)
#   gravity   = quat_apply_inverse(q_sensor, [0, 0, +1]): the world UP vector in the
#               SENSOR frame (observer.py:85 -> walk.py:132). The mount quaternion is
#               used only for body_quat diagnostics and the fall check (observer.py:84,
#               walk.py:101-106), NOT for the policy input.
# With q_body = q_sensor * m (imu_reader.py:69-73), R_sensor = R_body R_m^T, so
#   up_sensor   = R_m (R_body^T up)   = quat_apply(m, up_body)
#   gyro_sensor = R_m (R_body^T w)    = quat_apply(m, w_body)
# The sim's upstream terms are in the BODY frame (and gravity points DOWN), so the sim
# DOES need a change: rotate by m and negate gravity. This matches the replica's own
# training functions (mjlab_microduck/.../tasks/mdp.py:142-179).
# Upstream's per-env zero-centred IMU misalignment noise is kept: applied in the body
# frame first, then the fixed mount rotation.


def _to_sensor_frame(vec_b: torch.Tensor, mount_quat: tuple[float, float, float, float]) -> torch.Tensor:
    q = torch.tensor(mount_quat, dtype=vec_b.dtype, device=vec_b.device)
    q = (q / q.norm()).expand(vec_b.shape[0], 4)
    return quat_apply(q, vec_b)


def sensor_frame_up(
    env: ManagerBasedRlEnv,
    mount_quat: tuple[float, float, float, float],
    max_angle_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """World up-vector in the mounted IMU frame (upright robot -> R_m @ (0, 0, 1))."""
    if max_angle_deg > 0.0:
        g_b = microduck_mdp.projected_gravity_imu_misaligned(env, max_angle_deg, asset_cfg)
    else:
        g_b = env.scene[asset_cfg.name].data.projected_gravity_b
    return _to_sensor_frame(-g_b, mount_quat)


def sensor_frame_gyro(
    env: ManagerBasedRlEnv,
    mount_quat: tuple[float, float, float, float],
    max_angle_deg: float = 0.0,
    asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
    """Trunk angular velocity in the mounted IMU frame."""
    if max_angle_deg > 0.0:
        w_b = microduck_mdp.base_ang_vel_imu_misaligned(env, max_angle_deg, asset_cfg)
    else:
        w_b = env.scene[asset_cfg.name].data.root_link_ang_vel_b
    return _to_sensor_frame(w_b, mount_quat)


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------


def _replica_robot_cfg():
    """Upstream walk robot with replica servo kp and pack voltage range."""
    robot = deepcopy(MICRODUCK_WALK_ROBOT_CFG)
    (act,) = robot.articulation.actuators
    lo, hi = rc.PACK_VOLTAGE_RANGE_V
    if not (0.0 < lo <= hi):
        raise ValueError(f"PACK_VOLTAGE_RANGE_V must be (min, max) with 0 < min <= max, got {rc.PACK_VOLTAGE_RANGE_V}")
    robot.articulation.actuators = (
        dataclasses.replace(
            act,
            kp_fw=float(rc.SERVO_KP_REGISTER),
            vin_range=(float(lo), float(hi)),
            vin_min=float(rc.SAG_FLOOR_V),
        ),
    )
    return robot


def make_replica_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_microduck_velocity_env_cfg(play=play, rough=False)

    # 3 + 4: actuator stiffness and supply voltage.
    cfg.scene.entities = {"robot": _replica_robot_cfg()}

    # 1: drop head/body commands and everything that depends on them.
    for name in ("head_pose", "body_pose"):
        cfg.commands.pop(name)
    for name in ("head_pose_tracking", "head_pose_bias", "body_pose_tracking"):
        cfg.rewards.pop(name)
    for name in ("head_pose_range", "body_pose_range", "head_pose_bias_weight"):
        cfg.curriculum.pop(name)
    for group in ("actor", "critic"):
        for name in ("head_command", "body_command"):
            cfg.observations[group].terms.pop(name)

    # Head joints: mirror the replica's training env (its microduck_velocity_env_cfg.py
    # :237-263) - posture reward over ALL servos, neck/head pulled to home.
    pose = cfg.rewards["pose"]
    pose.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    head_std = {
        "standing": rc.HEAD_POSE_STD_STANDING,
        "walking": rc.HEAD_POSE_STD_WALKING,
    }
    for key, std in (("std_standing", head_std["standing"]),
                     ("std_walking", head_std["walking"]),
                     ("std_running", head_std["walking"])):
        pose.params[key] = {r".*neck.*": std, r".*head.*": std, **pose.params[key]}

    # 2: IMU terms in the mounted sensor frame (actor only; critic stays privileged
    # body-frame). Keeps upstream's noise, delay and misalignment settings.
    actor = cfg.observations["actor"].terms
    for term_name, func in (("projected_gravity", sensor_frame_up),
                            ("base_ang_vel", sensor_frame_gyro)):
        term = deepcopy(actor[term_name])
        old = dict(term.params or {})
        term.func = func
        term.params = {
            "mount_quat": rc.IMU_MOUNT_QUAT,
            "max_angle_deg": float(old.get("max_angle_deg", 0.0)),
        }
        actor[term_name] = term

    # Guard: the actor layout must be exactly the replica's.
    names = tuple(actor.keys())
    expected = tuple(n for n, _ in rc.POLICY_OBS_LAYOUT)
    if names != expected:
        raise RuntimeError(f"replica actor obs terms {names} != {expected}")
    return cfg


ReplicaRlCfg = dataclasses.replace(
    deepcopy(MicroduckRlCfg),
    experiment_name=rc.EXPERIMENT_NAME,
    run_name="velocity_replica",
)


def register() -> None:
    if rc.TASK_ID in list_tasks():
        return
    register_mjlab_task(
        task_id=rc.TASK_ID,
        env_cfg=make_replica_velocity_env_cfg(),
        play_env_cfg=make_replica_velocity_env_cfg(play=True),
        rl_cfg=ReplicaRlCfg,
        runner_cls=MicroduckOnPolicyRunner,
    )


register()
