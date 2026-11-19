"""Export a checkpoint to ONNX with upstream's unchanged export code, then check it.

Upstream's scripts/export.py does not import duck_train, so it cannot see our replica
task. This registers the task, then calls the same function upstream's CLI calls,
run_export(task, ExportConfig(...)), so input normalisation and metadata are baked in
exactly as upstream does it. The file is then checked against what the replica's
on-robot code reads (external/Microduck-build-tutorial/microduck/src/moves/walk.py:36-54):
input [1, 51], 14 outputs, joint_names in OBSERVATION_DOF_ORDER, 14 default_joint_pos
values. Any failed check exits non-zero.

    uv run --project third_party/microduck_rl python duck_train/export_policy.py --run smoke
    ... --checkpoint out/smoke/logs/rsl_rl/velocity_replica/<run>/model_4.pt --output x.onnx
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_EXPORT = ROOT / "third_party" / "microduck_rl" / "scripts" / "export.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duck_train import replica_config as rc  # noqa: E402

N_JOINTS = len(rc.JOINT_ORDER)


class ExportCheckError(RuntimeError):
    pass


def newest_checkpoint(run_name: str) -> Path:
    """Highest-numbered model_*.pt in the newest upstream run folder under out/<run_name>/."""
    runs = sorted(p for p in (ROOT / "out" / run_name).glob("logs/rsl_rl/*/*") if p.is_dir())
    runs = [r for r in runs if any(r.glob("model_*.pt"))]
    if not runs:
        raise SystemExit(f"no model_*.pt under out/{run_name}/logs/rsl_rl/")
    ckpts = sorted(runs[-1].glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return ckpts[-1]


def _load_upstream_export():
    spec = importlib.util.spec_from_file_location("microduck_rl_export", UPSTREAM_EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_onnx(onnx_path: Path, task: str) -> list[str]:
    """Check the exported file; return report lines, raise ExportCheckError on any failure."""
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path))
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    meta = sess.get_modelmeta().custom_metadata_map
    lines, errors = [], []

    def check(ok: bool, what: str) -> None:
        lines.append(f"  [{'ok' if ok else 'FAIL'}] {what}")
        if not ok:
            errors.append(what)

    replica = task == rc.TASK_ID
    if replica:
        check(list(inp.shape) == [1, rc.POLICY_OBS_SIZE], f"input shape {inp.shape} == [1, {rc.POLICY_OBS_SIZE}]")
    else:
        lines.append(f"  [info] input shape {inp.shape} (no fixed width for task {task})")
    check(len(out.shape) == 2 and out.shape[-1] == N_JOINTS, f"output shape {out.shape}, width {N_JOINTS}")

    names = meta.get("joint_names", "").split(",")
    check(tuple(names) == rc.JOINT_ORDER, f"joint_names == replica OBSERVATION_DOF_ORDER: {meta.get('joint_names')}")
    try:
        pose = [float(v) for v in meta["default_joint_pos"].split(",")]
    except (KeyError, ValueError) as e:
        pose = []
        lines.append(f"  default_joint_pos unreadable: {e!r}")
    check(len(pose) == N_JOINTS and all(math.isfinite(v) for v in pose),
          f"default_joint_pos has {N_JOINTS} finite values: {[round(v, 4) for v in pose]}")
    # walk.py hard-codes action_scale = 1.0 (walk.py:34); training must match.
    try:
        scale = [float(v) for v in meta["action_scale"].split(",")]
    except (KeyError, ValueError):
        scale = []
    check(bool(scale) and all(abs(s - 1.0) < 1e-9 for s in scale),
          f"action_scale all 1.0 (replica walk.py uses 1.0): {sorted(set(scale))}")
    if replica:
        want_obs = ",".join(n for n, _ in rc.POLICY_OBS_LAYOUT)
        check(meta.get("observation_names") == want_obs, f"observation_names == {want_obs}")
        check(meta.get("command_names") == "twist", f"command_names {meta.get('command_names')!r} == 'twist'")

    width = inp.shape[-1] if isinstance(inp.shape[-1], int) else rc.POLICY_OBS_SIZE
    y = sess.run(None, {inp.name: np.zeros((1, width), dtype=np.float32)})[0]
    check(y.shape == (1, N_JOINTS) and bool(np.all(np.isfinite(y))), f"zero input runs, output {y.shape} finite")
    lines.append(f"  [info] run_path metadata: {meta.get('run_path')}")
    if errors:
        raise ExportCheckError("ONNX check failed:\n" + "\n".join(lines))
    return lines


def export(task: str, checkpoint: Path, output: Path, device: str = "cpu") -> Path:
    import duck_train.replica_task  # noqa: F401  (registers the replica task)
    import mjlab.tasks  # noqa: F401  (registers upstream tasks, as export.py's main() does)

    up = _load_upstream_export()
    checkpoint, output = checkpoint.resolve(), output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Upstream writes the checkpoint path into the ONNX metadata (run_path). Run from the
    # repo root with a repo-relative path so no machine-specific path ends up in the file.
    os.chdir(ROOT)
    ckpt_arg = os.path.relpath(checkpoint, ROOT) if checkpoint.is_relative_to(ROOT) else str(checkpoint)
    cfg = up.ExportConfig(onnx_file=str(output), checkpoint_file=ckpt_arg, num_envs=1, device=device)
    up.run_export(task, cfg)
    return output


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run name: newest checkpoint under out/<run>/")
    src.add_argument("--checkpoint", type=Path, help="a model_*.pt file")
    p.add_argument("--task", default=rc.TASK_ID)
    p.add_argument("--output", type=Path, help="default: <checkpoint folder>/<checkpoint name>.onnx")
    p.add_argument("--device", default="cpu", help="torch device for loading/export (default cpu)")
    args = p.parse_args(argv)

    ckpt = newest_checkpoint(args.run) if args.run else args.checkpoint.resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    out = (args.output or ckpt.with_suffix(".onnx")).resolve()
    print(f"[export_policy] task={args.task}\n[export_policy] checkpoint={ckpt}\n[export_policy] output={out}", flush=True)
    export(args.task, ckpt, out, args.device)
    try:
        lines = check_onnx(out, args.task)
    except ExportCheckError as e:
        print(f"[export_policy] {e}", file=sys.stderr)
        return 1
    print("[export_policy] checks:\n" + "\n".join(lines))
    print(f"[export_policy] OK: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
