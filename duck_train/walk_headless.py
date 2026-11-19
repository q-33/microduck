"""Run an official Microduck ONNX walking policy headless and record a video + trace.

Reuses upstream's scripts/infer_policy.py (PolicyInference class, DEFAULT_POSE,
scene path, timestep, current limit, initial pose) without its viewer and
keyboard loop, which need a window and a raw terminal.

Run from the repo root:
    uv run --project third_party/microduck_rl python duck_train/walk_headless.py
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "microduck_rl"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", default="pretrained/pollen-robotics/alpha_walking.onnx",
                   help="walking policy, relative to the upstream repo")
    # alpha_walking stands still from rest at <=0.2 m/s; it walks from 0.25 up (measured).
    p.add_argument("--lin-vel-x", type=float, default=0.3, help="forward command, m/s (upstream keyboard max 0.3)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--video", default=str(ROOT / "out" / "walk_official.mp4"))
    p.add_argument("--trace", default=str(ROOT / "out" / "walk_official_trace.csv"))
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()
    video = Path(args.video).resolve()
    trace = Path(args.trace).resolve()
    video.parent.mkdir(parents=True, exist_ok=True)

    # Upstream uses repo-relative paths, so work from its root.
    os.chdir(UPSTREAM)
    sys.path.insert(0, str(UPSTREAM / "scripts"))
    import mujoco
    import infer_policy as up  # upstream module, unmodified
    from bam.model import load_model

    # Same setup as upstream main() with defaults (projected gravity, 1.75 A limit).
    model = mujoco.MjModel.from_xml_path(up.MICRODUCK_XML)
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    kt = load_model(motor_name="xl330", model="m6").kt.value
    model.actuator_forcerange[:, 0] = -kt * 1.75
    model.actuator_forcerange[:, 1] = kt * 1.75
    model.actuator_forcelimited[:] = 1

    policy = up.PolicyInference(model, data, walking_onnx_path=args.onnx,
                                use_projected_gravity=True, new_cmd_obs=True)
    policy.set_vel_cmd(args.lin_vel_x, 0.0, 0.0)

    fj = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[fj]
    data.qpos[qa:qa + 3] = [0.0, 0.0, 0.125]
    data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    for i, q in enumerate(policy.joint_qpos_indices):
        data.qpos[q] = policy.default_pose[i]
    data.ctrl[:] = policy.default_pose
    mujoco.mj_forward(model, data)
    obs_len = policy.get_observations().size
    print(f"obs size {obs_len}, policy input {policy.ort_session.get_inputs()[0].shape}")

    renderer = writer = None
    if not args.no_video:
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = policy.trunk_base_id
        cam.distance, cam.azimuth, cam.elevation = 0.6, 135.0, -20.0
        writer = imageio.get_writer(str(video), fps=50, codec="libx264", quality=7)

    decimation, rows, fell_at = 4, [], None
    steps = int(round(args.seconds / (decimation * model.opt.timestep)))
    for k in range(steps):
        policy.apply_action(policy.infer())
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        t = (k + 1) * decimation * model.opt.timestep
        x, y, z = data.qpos[qa:qa + 3]
        gz = float(policy.get_projected_gravity()[2])  # -1 = upright
        tilt = float(np.degrees(np.arccos(np.clip(-gz, -1, 1))))
        if fell_at is None and (z < 0.07 or tilt > 60):
            fell_at = t
        rows.append((round(t, 3), x, y, z, tilt))
        if renderer is not None:
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())
    if writer is not None:
        writer.close()

    with open(trace, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "base_x_m", "base_y_m", "base_z_m", "tilt_deg"])
        w.writerows(rows)

    a = np.array(rows)
    print(f"simulated {a[-1, 0]:.2f} s at 50 Hz control, command vx={args.lin_vel_x} m/s")
    for t in (0.02, 1, 2, 4, 6, 8, args.seconds):
        r = a[np.argmin(abs(a[:, 0] - t))]
        print(f"  t={r[0]:5.2f}s  x={r[1]:+.3f}  y={r[2]:+.3f}  z={r[3]:.4f}  tilt={r[4]:4.1f} deg")
    print(f"base z min/mean/max: {a[:, 3].min():.4f} / {a[:, 3].mean():.4f} / {a[:, 3].max():.4f} m")
    print(f"max tilt: {a[:, 4].max():.1f} deg")
    print(f"final displacement: x={a[-1, 1]:+.3f} m, y={a[-1, 2]:+.3f} m "
          f"(mean vx {a[-1, 1] / a[-1, 0]:.3f} m/s)")
    print("fell: " + ("no" if fell_at is None else f"yes, at t={fell_at:.2f}s") +
          "  (fall = base z < 0.07 m or tilt > 60 deg)")
    print(f"trace: {trace}")
    if writer is not None:
        print(f"video: {video}")


if __name__ == "__main__":
    main()
