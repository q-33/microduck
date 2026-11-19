"""Train with one setting for the number of parallel simulations; logs go to out/<name>/.

Runs duck_train/train.py (which registers our replica task, then calls upstream's
unchanged train entry point) in a child process whose working folder is out/<name>/.
Upstream writes logs/rsl_rl/<experiment>/<date>_<run>/ under the CURRENT folder, so
every run lands in out/<name>/logs/... wherever this script is started from.

Presets (from the repo root):
    smoke test, any machine:      --envs 64   --iterations 5    --name smoke
    Jetson Orin Nano 8 GB, first: --envs 512  --iterations 1000 --name orin512
    Jetson Orin Nano 8 GB, then:  --envs 1024 --iterations 1000 --name orin1024
    continue a run:               --name orin1024 --resume --iterations 1000

    uv run --project third_party/microduck_rl python duck_train/run_training.py \
        --envs 64 --iterations 5 --name smoke

Anything after the known options is passed to upstream unchanged, for example
`--agent.save-interval 100`. Upstream flags this script sets itself are refused there,
in any spelling upstream accepts (`-` or `_`, `--flag value` or `--flag=value`).
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT / "duck_train" / "train.py"
OUT = ROOT / "out"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duck_train import replica_config as rc  # noqa: E402  (plain constants, no heavy imports)

REPLICA_TASK = rc.TASK_ID
OFFICIAL_TASK = "Mjlab-Velocity-Flat-MicroDuck"

# Upstream TrainConfig.gpu_ids (mjlab/scripts/train.py) is `list[int] | "all" | None`,
# parsed with mjlab.TYRO_FLAGS (Python syntax for collections): "None" = CPU,
# "[0]" = first visible GPU (upstream's default). A bare "0" is rejected by tyro.
GPU_IDS = {"cpu": "None", "cuda": "[0]"}

# Upstream flags this launcher owns, in canonical (hyphen) spelling. tyro accepts
# both "-" and "_" in flag names and "--flag=value"; the last repeat silently wins.
OWNED_FLAGS = ("--gpu-ids", "--env.scene.num-envs", "--agent.max-iterations",
               "--agent.logger", "--agent.resume")
LOAD_RUN = "--agent.load-run"            # regex, re.match on run folder names (mjlab/utils/os.py)
LOAD_CHECKPOINT = "--agent.load-checkpoint"
DEFAULT_CHECKPOINT_PATTERN = "model_.*.pt"  # RslRlBaseRunnerCfg.load_checkpoint default

_NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _canonical(arg: str) -> str | None:
    """'--env.scene.num_envs=8' -> '--env.scene.num-envs'; non-flags -> None."""
    if not arg.startswith("--"):
        return None
    return arg.split("=", 1)[0].replace("_", "-")


def extra_value(extra: list[str], flag: str) -> str | None:
    """Last value given for an upstream flag (any spelling) in the pass-through args."""
    value = None
    for i, arg in enumerate(extra):
        if _canonical(arg) == flag:
            if "=" in arg:
                value = arg.split("=", 1)[1]
            elif i + 1 < len(extra):
                value = extra[i + 1]
    return value


def run_dir_for(name: str) -> Path:
    if not isinstance(name, str) or not _NAME_OK.fullmatch(name):
        raise ValueError(f"run name {name!r}: use letters, digits, '.', '_' or '-', starting with a letter or digit")
    return OUT / name


def build_command(task: str, envs: int, iterations: int, device: str, resume: bool,
                  logger: str = "tensorboard", extra: list[str] | None = None,
                  python: str | None = None, load_run: str | None = None) -> list[str]:
    """The exact upstream command (argv), without side effects."""
    if envs < 1 or iterations < 1:
        raise ValueError("--envs and --iterations must be at least 1")
    if device not in GPU_IDS:
        raise ValueError(f"device must be one of {sorted(GPU_IDS)}, got {device!r}")
    extra = list(extra or [])
    for arg in extra:
        if _canonical(arg) in OWNED_FLAGS:
            raise ValueError(f"{arg.split('=', 1)[0]} is set by this launcher; use its own option instead")
    cmd = [python or sys.executable, str(TRAIN_PY), task,
           "--gpu-ids", GPU_IDS[device],
           "--env.scene.num-envs", str(envs),
           "--agent.max-iterations", str(iterations),
           "--agent.logger", logger]
    if resume:
        cmd += ["--agent.resume", "True"]
        if load_run is not None:
            cmd += [LOAD_RUN, load_run]
    return cmd + extra


# --------------------------------------------------------------------------- resume


def _run_folders(exp_dir: Path) -> list[Path]:
    # Same set and order upstream uses (get_checkpoint_path: alphabetical, no wandb cache).
    return sorted(p for p in exp_dir.iterdir() if p.is_dir() and p.name != "wandb_checkpoints")


def _checkpoints(run: Path, pattern: str) -> list[str]:
    names = [f.name for f in run.iterdir() if re.match(pattern, f.name)]
    return sorted(names, key=lambda m: f"{m:0>15}")  # upstream's sort


def choose_resume(exp_dir: Path, ckpt_pattern: str = DEFAULT_CHECKPOINT_PATTERN,
                  user_load_run: str | None = None) -> tuple[Path, str]:
    """Run folder and checkpoint upstream will load.

    Upstream takes the NEWEST run folder whose name matches load_run, then the newest
    checkpoint in it, and fails if that folder has none (e.g. an interrupted resumed run
    that never reached a save). Without a user load_run we pick the newest folder that
    HAS a checkpoint and pin upstream to it; with one, we predict upstream's own choice.
    """
    runs = _run_folders(exp_dir) if exp_dir.is_dir() else []
    if user_load_run is not None:
        runs = [r for r in runs if re.match(user_load_run, r.name)]
        if not runs:
            raise SystemExit(f"--resume: no run folder in {exp_dir} matches {LOAD_RUN} {user_load_run!r}")
        ckpts = _checkpoints(runs[-1], ckpt_pattern)
        if not ckpts:
            raise SystemExit(f"--resume: newest folder matching {user_load_run!r}, {runs[-1].name}, "
                             f"has no checkpoint matching {ckpt_pattern!r}")
        return runs[-1], ckpts[-1]
    with_ckpt = [r for r in runs if _checkpoints(r, ckpt_pattern)]
    if not with_ckpt:
        raise SystemExit(f"--resume: no checkpoint matching {ckpt_pattern!r} in any run folder under {exp_dir}")
    run = with_ckpt[-1]
    return run, _checkpoints(run, ckpt_pattern)[-1]


def experiment_name(task: str, extra: list[str]) -> str:
    given = extra_value(extra, "--agent.experiment-name")
    if given:
        return given
    if task == REPLICA_TASK:
        return rc.EXPERIMENT_NAME
    import duck_train.replica_task  # noqa: F401  (registers our task beside upstream's)
    from mjlab.tasks.registry import list_tasks, load_rl_cfg
    if task not in list_tasks():
        raise SystemExit(f"unknown task {task!r}")
    return load_rl_cfg(task).experiment_name


def plan_resume(run_dir: Path, task: str, extra: list[str]) -> tuple[str | None, Path, str]:
    """(load_run value to pass or None, chosen run folder, checkpoint name)."""
    rsl = run_dir / "logs" / "rsl_rl"
    if not rsl.is_dir():
        raise SystemExit(f"--resume: {run_dir} has no logs/rsl_rl/ to resume from")
    exp = experiment_name(task, extra)
    if not (rsl / exp).is_dir():
        found = sorted(p.name for p in rsl.iterdir() if p.is_dir())
        raise SystemExit(f"--resume: task {task} logs to '{exp}', but {run_dir.name} has {found}; "
                         f"pass the --task this run was trained with")
    user_run = extra_value(extra, LOAD_RUN)
    pattern = extra_value(extra, LOAD_CHECKPOINT) or DEFAULT_CHECKPOINT_PATTERN
    run, ckpt = choose_resume(rsl / exp, pattern, user_run)
    load_run = None if user_run is not None else re.escape(run.name) + "$"
    return load_run, run, ckpt


# --------------------------------------------------------------------------- launch


def has_checkpoint(folder: Path) -> bool:
    return any(folder.rglob("model_*.pt"))


def check_run_dir(run_dir: Path, resume: bool) -> None:
    if resume and not has_checkpoint(run_dir):
        raise SystemExit(f"--resume: no checkpoint (model_*.pt) under {run_dir}/logs/rsl_rl/")
    if not resume and run_dir.exists():
        raise SystemExit(f"{run_dir} already exists. Use a new --name, or --resume to continue it.")


def cleanup_failed_launch(run_dir: Path, existed_before: bool, runs_before: set[Path]) -> list[Path]:
    """After a failed launch, remove only checkpoint-free folders this launch created."""
    removed = []
    if not existed_before:
        if run_dir.exists() and not has_checkpoint(run_dir):
            shutil.rmtree(run_dir)
            removed.append(run_dir)
        return removed
    for run in sorted(set(run_dir.glob("logs/rsl_rl/*/*")) - runs_before):
        if run.is_dir() and not has_checkpoint(run):
            shutil.rmtree(run)
            removed.append(run)
    return removed


def resolve_device(device: str) -> str:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() == "":
        print("[run_training] WARNING: CUDA_VISIBLE_DEVICES is set but empty, so no GPU is visible "
              "(upstream then trains on the CPU). Unset it to use the GPU.", file=sys.stderr)
    if device == "cpu":
        return "cpu"
    import torch
    has_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    if device == "cuda" and not has_cuda:
        raise SystemExit("[run_training] --device cuda: torch sees no CUDA device on this machine; "
                         "use --device cpu (or auto)")
    return "cuda" if has_cuda else "cpu"


class OneLineParser(argparse.ArgumentParser):
    """argparse with a one-line error (exit code 2) instead of usage + error."""

    def error(self, message: str):
        self.exit(2, f"{self.prog}: error: {message}\n")


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def run_name(text: str) -> str:
    try:
        run_dir_for(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return text


def make_parser() -> OneLineParser:
    p = OneLineParser(description=__doc__.splitlines()[0],
                      epilog="Unknown arguments are passed to upstream's train unchanged.")
    p.add_argument("--envs", type=positive_int, default=64, help="parallel simulations (default 64 = smoke test)")
    p.add_argument("--iterations", type=positive_int, default=5,
                   help="PPO iterations (with --resume: further iterations). Default 5 = smoke test")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto = cuda if torch sees a GPU, else cpu")
    p.add_argument("--name", type=run_name, required=True, help="run folder name: logs go to out/<name>/")
    p.add_argument("--task", default=REPLICA_TASK,
                   help=f"default {REPLICA_TASK}; the official one is {OFFICIAL_TASK}")
    p.add_argument("--resume", action="store_true",
                   help="continue the newest checkpoint in out/<name>/ (newest run folder that has one)")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    p.add_argument("--dry-run", action="store_true", help="print the command and stop")
    return p


def parse(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    return make_parser().parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    parser = make_parser()
    args, extra = parser.parse_known_args(argv)
    run_dir = run_dir_for(args.name)
    try:  # flag clashes before anything slow
        build_command(args.task, args.envs, args.iterations, "cpu", args.resume, args.logger, extra)
    except ValueError as e:
        parser.error(str(e))
    load_run = None
    if args.resume:
        check_run_dir(run_dir, resume=True)
        load_run, run, ckpt = plan_resume(run_dir, args.task, extra)
        print(f"[run_training] resuming from {run.relative_to(ROOT)}/{ckpt}")
    else:
        check_run_dir(run_dir, resume=False)
    device = resolve_device(args.device)
    cmd = build_command(args.task, args.envs, args.iterations, device, args.resume,
                        args.logger, extra, load_run=load_run)
    print(f"[run_training] device={device} envs={args.envs} iterations={args.iterations} task={args.task}")
    print(f"[run_training] working folder: {run_dir}")
    print(f"[run_training] command: {shlex.join(cmd)}", flush=True)
    if args.dry_run:
        return 0

    existed_before = run_dir.exists()
    runs_before = set(run_dir.glob("logs/rsl_rl/*/*"))
    run_dir.mkdir(parents=True, exist_ok=True)
    code = 130
    try:
        code = subprocess.run(cmd, cwd=run_dir).returncode
    except KeyboardInterrupt:
        print("\n[run_training] interrupted", file=sys.stderr)
    finally:
        if code != 0:
            for path in cleanup_failed_launch(run_dir, existed_before, runs_before):
                print(f"[run_training] removed {path} (created by this failed launch, no checkpoint)",
                      file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
