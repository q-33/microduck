"""Replay an exported ONNX policy headless in our task and report how many robots fell.

Same approach as tests/test_replica_task.py (walk.onnx check): the task's training
config with pushes off and a fixed forward command, CPU, ONNX run with onnxruntime.
The env is seeded (mjlab ManagerBasedRlEnvCfg.seed), so a replay with the same seed
repeats. A policy trained for a few iterations will fall; this checks that the file
loads and steps without shape errors ("runs"), not that it walks.

    uv run --project third_party/microduck_rl python duck_train/replay_policy.py --onnx <file.onnx>
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duck_train import replica_config as rc  # noqa: E402
from duck_train.run_training import OneLineParser, positive_int  # noqa: E402

DEFAULT_SEED = 1
DEFAULT_ROBOTS = 32


def replay(onnx_path: Path, task: str = rc.TASK_ID, seconds: float = 10.0,
           lin_vel_x: float = 0.2, num_envs: int = DEFAULT_ROBOTS, seed: int = DEFAULT_SEED) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch

    import duck_train.replica_task  # noqa: F401  (registers the replica task)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    sess = ort.InferenceSession(str(onnx_path))
    inp = sess.get_inputs()[0]
    cfg = load_env_cfg(task)
    tw = cfg.commands["twist"]
    tw.ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
    tw.ranges.lin_vel_y = (0.0, 0.0)
    tw.ranges.ang_vel_z = (0.0, 0.0)
    tw.rel_standing_envs = 0.0
    tw.rel_turn_in_place_envs = 0.0
    cfg.events.pop("push_robot", None)
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    try:
        obs, _ = env.reset()
        width = obs["actor"].shape[-1]
        if inp.shape[-1] != width:
            raise ValueError(f"ONNX input {inp.shape} does not fit task {task} observation width {width}")
        tm = env.termination_manager
        terms = tm.active_terms
        steps = int(round(seconds / env.step_dt))
        fell = torch.zeros(num_envs, dtype=torch.bool)
        falls = nan_resets = other_resets = 0
        first_fall, min_z = None, float("inf")
        robot = env.scene["robot"]
        for k in range(steps):
            # The exported graph takes batch 1, as the robot feeds it; run robot by robot.
            o = obs["actor"].numpy().astype(np.float32)
            a = np.concatenate([sess.run(None, {inp.name: o[i:i + 1]})[0] for i in range(num_envs)])
            if a.shape != (num_envs, env.action_manager.total_action_dim):
                raise ValueError(f"action shape {a.shape}")
            obs, _, terminated, _, _ = env.step(torch.from_numpy(a))
            # Per-term flags of this step (TerminationManager keeps them until the next compute).
            f = tm.get_term("fell_over").clone() if "fell_over" in terms else terminated.clone()
            n = tm.get_term("nan_state").clone() if "nan_state" in terms else torch.zeros_like(f)
            f &= ~n
            falls += int(f.sum())
            nan_resets += int(n.sum())
            other_resets += int((terminated & ~f & ~n).sum())
            if f.any() and first_fall is None:
                first_fall = k
            fell |= f
            min_z = min(min_z, float(robot.data.root_link_pos_w[:, 2].min()))
        return {"steps": steps, "seconds": steps * env.step_dt, "num_envs": num_envs, "seed": seed,
                "robots_fell": int(fell.sum()), "falls": falls, "first_fall_step": first_fall,
                "nan_resets": nan_resets, "other_resets": other_resets, "min_base_z": min_z,
                "input_shape": list(inp.shape), "obs_width": width}
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    p = OneLineParser(description=__doc__.splitlines()[0])
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--task", default=rc.TASK_ID)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--lin-vel-x", type=float, default=0.2, help="forward command, m/s")
    p.add_argument("--envs", type=positive_int, default=DEFAULT_ROBOTS, help="robots simulated side by side")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)
    r = replay(args.onnx, args.task, args.seconds, args.lin_vel_x, args.envs, args.seed)
    print(f"[replay] {args.onnx}")
    print(f"[replay] runs: {r['steps']} steps ({r['seconds']:.1f} s) x {r['num_envs']} robots, "
          f"input {r['input_shape']} = task obs width {r['obs_width']}, no shape errors")
    first = "" if r["first_fall_step"] is None else f" (first at step {r['first_fall_step']})"
    print(f"[replay] seed {r['seed']}: {r['robots_fell']} of {r['num_envs']} robots fell{first}; "
          f"{r['falls']} falls in total, {r['nan_resets']} NaN resets, {r['other_resets']} other resets; "
          f"lowest base height {r['min_base_z']:.3f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
