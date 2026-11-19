"""Tests for the replica task (duck_train/replica_task.py).

Run from the repo root, in upstream's environment:
    uv run --project third_party/microduck_rl --with pytest pytest tests/ -q -s
The 2-iteration training smoke test takes about 10 s on an M-series CPU, so it is not marked slow.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import mjlab  # noqa: F401  (loads upstream tasks through its entry point)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg

from duck_train import replica_config as rc
from duck_train import replica_task as rt

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "microduck_rl"
REPLICA = ROOT / "external" / "Microduck-build-tutorial"
REPLICA_SRC = REPLICA / "microduck" / "src"
REPLICA_ONNX = REPLICA_SRC / "agents" / "walk.onnx"
UPSTREAM_TASK = "Mjlab-Velocity-Flat-MicroDuck"

needs_replica = pytest.mark.skipif(not REPLICA_SRC.exists(), reason="external/Microduck-build-tutorial not cloned")


def _build(cfg, num_envs: int) -> ManagerBasedRlEnv:
    cfg.scene.num_envs = num_envs
    return ManagerBasedRlEnv(cfg=cfg, device="cpu")


@pytest.fixture(scope="module")
def replica_env():
    env = _build(load_env_cfg(rc.TASK_ID), num_envs=4)
    env.reset()
    yield env
    env.close()


def _robot_actuator(env):
    acts = env.scene["robot"]._actuators  # mjlab 1.3.0 keeps built actuators here
    assert len(acts) == 1
    return acts[0]


# --------------------------------------------------------------------------- layout


def test_task_registered_beside_upstream():
    tasks = list_tasks()
    assert rc.TASK_ID in tasks and UPSTREAM_TASK in tasks


def test_policy_observation_is_51_in_replica_order(replica_env):
    om = replica_env.observation_manager
    assert om.group_obs_dim["actor"] == (51,)
    terms = om.active_terms["actor"]
    dims = [d[0] for d in om.group_obs_term_dim["actor"]]
    assert list(zip(terms, dims)) == [tuple(x) for x in rc.POLICY_OBS_LAYOUT]
    assert rc.POLICY_OBS_SIZE == 51
    assert "head_pose" not in replica_env.command_manager.active_terms
    assert "body_pose" not in replica_env.command_manager.active_terms


def test_action_joint_order_matches_replica(replica_env):
    names = list(replica_env.scene["robot"].joint_names)
    servo = [n for n in names if not n.startswith("passive_")]
    assert tuple(servo) == rc.JOINT_ORDER
    assert replica_env.action_manager.total_action_dim == 14


@needs_replica
def test_config_joint_order_equals_replica_constants():
    sys.path.insert(0, str(REPLICA_SRC))
    try:
        constants = importlib.import_module("constants")
        assert tuple(constants.OBSERVATION_DOF_ORDER) == rc.JOINT_ORDER
        assert constants.KP_RL == rc.SERVO_KP_REGISTER
        assert tuple(constants.IMU_MOUNT_QUAT) == rc.IMU_MOUNT_QUAT
    finally:
        sys.path.remove(str(REPLICA_SRC))
        sys.modules.pop("constants", None)


def test_upstream_task_unchanged_by_import():
    env = _build(load_env_cfg(UPSTREAM_TASK), num_envs=1)
    try:
        assert env.observation_manager.group_obs_dim["actor"] == (61,)
        assert "head_command" in env.observation_manager.active_terms["actor"]
        act = _robot_actuator(env)
        assert act.cfg.kp_fw == 200.0
        assert tuple(act.cfg.vin_range) == (6.5, 8.2)
    finally:
        env.close()


# --------------------------------------------------------------------------- actuator


def test_actuator_kp_and_voltage_from_config(replica_env):
    act = _robot_actuator(replica_env)
    assert act.cfg.kp_fw == rc.SERVO_KP_REGISTER == 125.0
    assert act._base_kp == pytest.approx(125.0)
    assert tuple(act.cfg.vin_range) == tuple(rc.PACK_VOLTAGE_RANGE_V)
    assert act.cfg.vin_min == rc.SAG_FLOOR_V
    lo, hi = rc.PACK_VOLTAGE_RANGE_V
    v = act.vin_tensor
    assert float(v.min()) >= lo - 1e-6 and float(v.max()) <= hi + 1e-6
    # And they really differ from upstream's defaults.
    assert act.cfg.kp_fw != 200.0
    assert tuple(act.cfg.vin_range) != (6.5, 8.2)


# --------------------------------------------------------------------------- IMU frame


def test_upright_robot_sees_mounted_up_vector(replica_env):
    """Upright trunk: replica policy input is R_m @ (0,0,1) = (1,0,0), not (0,0,-1)."""
    replica_env.reset()
    obs = replica_env.observation_manager.compute()["actor"]
    up = obs[:, 3:6]
    # Allow reset tilt, 6 deg misalignment DR and +/-0.01 noise.
    assert torch.all(up[:, 0] > 0.9), up
    assert torch.all(up[:, 2].abs() < 0.3), up


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


@needs_replica
def test_sim_imu_terms_equal_replica_runtime_formulas():
    """For random trunk poses, our sim terms equal what the replica runtime computes.

    Runtime: imu_reader.quat_apply_inverse(q_sensor, [0, 0, 1]) (observer.py:85) and the
    raw sensor-frame gyro, with q_body = imu_quat_to_body(q_sensor) (imu_reader.py:69-73).
    Assumes the BNO08x quaternion maps sensor->world and the gyro is in the same sensor
    frame; that convention is to be confirmed on hardware.
    """
    sys.path.insert(0, str(REPLICA_SRC))
    try:
        imu_reader = importlib.import_module("imu_reader")
    finally:
        sys.path.remove(str(REPLICA_SRC))
    try:
        rng = np.random.default_rng(0)
        m = np.array(rc.IMU_MOUNT_QUAT) / np.linalg.norm(rc.IMU_MOUNT_QUAT)
        m_conj = m * np.array([1, -1, -1, -1])
        for _ in range(50):
            q_b = rng.normal(size=4)
            q_b /= np.linalg.norm(q_b)
            q_s = _quat_mul(q_b, m_conj)  # so that q_b = q_s * m
            assert np.allclose(imu_reader.imu_quat_to_body(tuple(q_s)), q_b, atol=1e-6) or \
                np.allclose(imu_reader.imu_quat_to_body(tuple(q_s)), -q_b, atol=1e-6)
            w_world = rng.normal(size=3)

            real_up = np.array(imu_reader.quat_apply_inverse(q_s, [0.0, 0.0, 1.0]))
            real_gyro = np.array(imu_reader.quat_apply_inverse(q_s, w_world))

            gravity_b = np.array(imu_reader.quat_apply_inverse(q_b, [0.0, 0.0, -1.0]))
            w_b = np.array(imu_reader.quat_apply_inverse(q_b, w_world))
            sim_up = rt._to_sensor_frame(-torch.tensor(gravity_b)[None], rc.IMU_MOUNT_QUAT)[0].numpy()
            sim_gyro = rt._to_sensor_frame(torch.tensor(w_b)[None], rc.IMU_MOUNT_QUAT)[0].numpy()
            assert np.allclose(sim_up, real_up, atol=1e-5)
            assert np.allclose(sim_gyro, real_gyro, atol=1e-5)
    finally:
        for name in ("imu_reader", "constants"):
            sys.modules.pop(name, None)


# --------------------------------------------------------------------------- replica walk.onnx


def _yaw(robot) -> float:
    w, x, y, z = robot.data.root_link_quat_w[0].tolist()
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


@needs_replica
def test_replica_walk_onnx_runs_in_replica_task():
    import onnxruntime as ort

    sess = ort.InferenceSession(str(REPLICA_ONNX))
    inp = sess.get_inputs()[0]
    meta = sess.get_modelmeta().custom_metadata_map
    print(f"\n[walk.onnx] input {inp.name} {inp.shape}, output {sess.get_outputs()[0].shape}")
    print(f"[walk.onnx] observation_names={meta.get('observation_names')} command_names={meta.get('command_names')}")
    assert inp.shape[-1] == 51
    assert tuple(meta["joint_names"].split(",")) == rc.JOINT_ORDER

    cfg = load_env_cfg(rc.TASK_ID)
    vx = 0.2
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = (vx, vx)
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.rel_standing_envs = 0.0
    tw.rel_turn_in_place_envs = 0.0
    cfg.events.pop("push_robot", None)
    env = _build(cfg, num_envs=1)
    try:
        robot = env.scene["robot"]
        default = robot.data.default_joint_pos[0, :14].numpy()
        meta_default = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
        print(f"[walk.onnx] max |default_joint_pos(onnx) - task default| = {np.abs(default - meta_default).max():.4f} rad")

        obs, _ = env.reset()
        steps, falls, first_fall = 500, 0, None
        start_xy = robot.data.root_link_pos_w[0, :2].clone()
        start_yaw = _yaw(robot)
        heights, tilts = [], []
        for k in range(steps):
            a = sess.run(None, {inp.name: obs["actor"].numpy().astype(np.float32)})[0]
            assert a.shape == (1, 14)
            obs, _, terminated, _, _ = env.step(torch.from_numpy(a))
            assert obs["actor"].shape == (1, 51)
            if bool(terminated[0]):
                falls += 1
                first_fall = first_fall if first_fall is not None else k
                start_xy = robot.data.root_link_pos_w[0, :2].clone()
                start_yaw = _yaw(robot)
            heights.append(float(robot.data.root_link_pos_w[0, 2]))
            gz = float(robot.data.projected_gravity_b[0, 2])
            tilts.append(float(np.degrees(np.arccos(np.clip(-gz, -1, 1)))))
        d = (robot.data.root_link_pos_w[0, :2] - start_xy).numpy()
        c, s_ = np.cos(start_yaw), np.sin(start_yaw)
        disp = np.array([c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]])  # in the start heading frame
        report = (
            f"[walk.onnx] {steps} steps ({steps * env.step_dt:.1f} s) at vx={vx}: "
            f"terminations={falls} (first at step {first_fall}), "
            f"base z min/max {min(heights):.3f}/{max(heights):.3f} m, max tilt {max(tilts):.1f} deg, "
            f"displacement in start-heading frame forward={disp[0]:+.3f} left={disp[1]:+.3f} m, "
            f"yaw change {np.degrees(_yaw(robot) - start_yaw):+.0f} deg"
        )
        print(report)
        out = ROOT / "out"
        out.mkdir(exist_ok=True)
        (out / "replica_walk_onnx_report.txt").write_text(report + "\n")
    finally:
        env.close()


# --------------------------------------------------------------------------- training smoke


def test_training_smoke_exports_51_wide_onnx():
    import onnxruntime as ort

    work = ROOT / "out" / "pytest_replica_smoke"
    work.mkdir(parents=True, exist_ok=True)
    before = set(work.rglob("*.onnx"))
    cmd = [
        sys.executable, str(ROOT / "duck_train" / "train.py"), rc.TASK_ID,
        "--gpu-ids", "None", "--env.scene.num-envs", "16",
        "--agent.max-iterations", "2", "--agent.logger", "tensorboard",
    ]
    res = subprocess.run(cmd, cwd=work, capture_output=True, text=True, timeout=900,
                         env={**os.environ, "MJLAB_WARP_QUIET": "1"})
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    new = sorted(set(work.rglob("*.onnx")) - before)
    assert new, "no ONNX exported"
    onnx = new[-1]
    assert f"logs/rsl_rl/{rc.EXPERIMENT_NAME}/" in onnx.as_posix()
    sess = ort.InferenceSession(str(onnx))
    meta = sess.get_modelmeta().custom_metadata_map
    print(f"\n[smoke] {onnx.relative_to(ROOT)} input {sess.get_inputs()[0].shape}")
    assert sess.get_inputs()[0].shape[-1] == 51
    assert tuple(meta["joint_names"].split(",")) == rc.JOINT_ORDER
    assert meta["observation_names"] == ",".join(n for n, _ in rc.POLICY_OBS_LAYOUT)
    assert meta["command_names"] == "twist"
