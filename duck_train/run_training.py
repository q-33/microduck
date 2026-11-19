"""Train with one setting for the number of parallel simulations; logs go to out/<name>/.

Runs duck_train/train.py (which registers our replica task, then calls upstream's
unchanged train entry point) in a child process whose working folder is out/<name>/.
Upstream writes logs/rsl_rl/<experiment>/<date>_<run>/ under the CURRENT folder, so
every run lands in out/<name>/logs/... wherever this script is started from.

Presets (run from anywhere):
    smoke test, any machine:      --envs 64   --iterations 5    --name smoke
    Jetson Orin Nano 8 GB, first: --envs 512  --iterations 1000 --name orin512
    Jetson Orin Nano 8 GB, then:  --envs 1024 --iterations 1000 --name orin1024
    continue a run:               --name orin1024 --resume --iterations 1000

    uv run --project third_party/microduck_rl python duck_train/run_training.py \
        --envs 64 --iterations 5 --name smoke

Anything after the known options is passed to upstream unchanged, for example
`--agent.save-interval 100`. Upstream flags this script sets itself are refused there.
"""
from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "duck_train" / "train.py"
OUT = ROOT / "out"

REPLICA_TASK = "Mjlab-Velocity-Flat-MicroDuck-Replica"
OFFICIAL_TASK = "Mjlab-Velocity-Flat-MicroDuck"

# Upstream TrainConfig.gpu_ids (mjlab/scripts/train.py) is `list[int] | "all" | None`,
# parsed with mjlab.TYRO_FLAGS (Python syntax for collections): "None" = CPU,
# "[0]" = first visible GPU (upstream's default). A bare "0" is rejected by tyro.
GPU_IDS = {"cpu": "None", "cuda": "[0]"}

# Upstream flags this launcher owns; passing them again as extras would be ambiguous.
OWNED_FLAGS = ("--gpu-ids", "--env.scene.num-envs", "--agent.max-iterations",
               "--agent.max_iterations", "--agent.logger", "--agent.resume")

_NAME_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def run_dir_for(name: str) -> Path:
    if not _NAME_OK.match(name):
        raise ValueError(f"run name {name!r}: use letters, digits, '.', '_' or '-' only")
    return OUT / name


def build_command(task: str, envs: int, iterations: int, device: str, resume: bool,
                  logger: str = "tensorboard", extra: list[str] | None = None,
                  python: str | None = None) -> list[str]:
    """The exact upstream command (argv), without side effects."""
    if envs < 1 or iterations < 1:
        raise ValueError("--envs and --iterations must be at least 1")
    if device not in GPU_IDS:
        raise ValueError(f"device must be one of {sorted(GPU_IDS)}, got {device!r}")
    extra = list(extra or [])
    for arg in extra:
        flag = arg.split("=", 1)[0]
        if flag in OWNED_FLAGS:
            raise ValueError(f"{flag} is set by this launcher; use its own option instead")
    cmd = [python or sys.executable, str(TRAIN_PY), task,
           "--gpu-ids", GPU_IDS[device],
           "--env.scene.num-envs", str(envs),
           "--agent.max-iterations", str(iterations),
           "--agent.logger", logger]
    if resume:
        # Upstream then loads the newest model_*.pt of the newest run under
        # logs/rsl_rl/<experiment>/ in the working folder (--agent.load-run/-checkpoint
        # can narrow that; pass them as extras).
        cmd += ["--agent.resume", "True"]
    return cmd + extra


def check_run_dir(run_dir: Path, resume: bool) -> None:
    has_ckpt = any(run_dir.glob("logs/rsl_rl/*/*/model_*.pt"))
    if resume and not has_ckpt:
        raise SystemExit(f"--resume: no checkpoint (model_*.pt) under {run_dir}/logs/rsl_rl/")
    if not resume and run_dir.exists():
        raise SystemExit(f"{run_dir} already exists. Use a new --name, or --resume to continue it.")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                epilog="Unknown arguments are passed to upstream's train unchanged.")
    p.add_argument("--envs", type=int, default=64, help="parallel simulations (default 64 = smoke test)")
    p.add_argument("--iterations", type=int, default=5,
                   help="PPO iterations (with --resume: further iterations). Default 5 = smoke test")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto = cuda if torch sees a GPU, else cpu")
    p.add_argument("--name", required=True, help="run folder name: logs go to out/<name>/")
    p.add_argument("--task", default=REPLICA_TASK,
                   help=f"default {REPLICA_TASK}; the official one is {OFFICIAL_TASK}")
    p.add_argument("--resume", action="store_true", help="continue the newest checkpoint in out/<name>/")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    p.add_argument("--dry-run", action="store_true", help="print the command and stop")
    return p.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    args, extra = parse(argv)
    run_dir = run_dir_for(args.name)
    device = resolve_device(args.device)
    cmd = build_command(args.task, args.envs, args.iterations, device, args.resume,
                        args.logger, extra)
    print(f"[run_training] device={device} envs={args.envs} iterations={args.iterations} task={args.task}")
    print(f"[run_training] working folder: {run_dir}")
    print(f"[run_training] command: {shlex.join(cmd)}", flush=True)
    check_run_dir(run_dir, args.resume)
    if args.dry_run:
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, cwd=run_dir).returncode


if __name__ == "__main__":
    sys.exit(main())
