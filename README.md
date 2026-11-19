# MicroDuck

Build, print and training files for a Microduck biped (community replica, 14 XL330 servos).
Training builds on the upstream repo [jjjadand/microduck_rl](https://github.com/jjjadand/microduck_rl)
(mjlab / MuJoCo Warp + rsl_rl PPO). Upstream is pinned as a submodule and is never edited;
our code sits beside it.

## The project

- **Robot:** the community replica from
  [AI-FanGe/Microduck-build-tutorial](https://github.com/AI-FanGe/Microduck-build-tutorial).
  14 Dynamixel XL330-M288-T servos, OpenRB-150 servo controller, BNO08x IMU mounted vertically.
- **Printing:** first robot in PETG on a Prusa XL, with PLA supports. Glass-filled nylon may
  replace the most loaded parts later. Plates and part list are in `print/`.
- **Training:** a Jetson Orin Nano on the bench trains the walking policy in simulation. It does
  not go on the robot. The policy is exported to ONNX and runs on the robot's own small computer.
- **Until the Jetson arrives:** everything here runs on an Apple Silicon Mac, CPU only.

## Status (2026-11-18)

| Piece | State |
|---|---|
| Print plates (Prusa XL, and a 250 x 210 mm fallback set) | Made and checked by script and PrusaSlicer's command line. Not sliced, nothing printed. |
| Mac environment and upstream tests | Working: 149 tests pass. |
| Official policy walking in simulation | Working, headless, with a video clip. |
| CPU training smoke test | Working. Too slow for real training. |
| Part-weight tool (`tools/duck_mass/`) | Built and tested. Needs real part weights and a part-to-body mapping. |
| Replica training task (`duck_train/`) | Built; 10 tests pass on the Mac. IMU frame and battery voltage still to be confirmed on the real robot. |
| Jetson setup script, full training | Not started. Waits for the board. |

Open questions: the battery (and so the supply voltage to train for); which JetPack the Orin
Nano gets; whether the replica's part dimensions and weights match the official simulation
model (asked upstream in
[issue 17](https://github.com/AI-FanGe/Microduck-build-tutorial/issues/17)).

The plan and running notes are in `tasks/todo.md`. `MICRODUCK_HANDOFF.md` is the original
planning note; where it disagrees with this file, this file is newer.

## Printing

`print/README.md` has the print order, the slicer settings and what was verified.

- `print/prusa_xl_petg/`: 3 plates for the Prusa XL. Print `XL01_test_piece` first and check the
  servo fit. The back-right corner of each plate is kept clear for the wipe tower.
- `print/prusa_all_pla/`: the same 67 parts on 7 plates for 250 x 210 mm beds.
- `print/individual_STL/`: every part on its own.
- `print/parts.csv`: each part's size, volume and estimated PLA and PETG weight.

The plate files hold geometry and placement only. Toolheads, supports and the wipe tower
position are set in PrusaSlicer.

## Part weights into the simulation

`tools/duck_mass/` writes a generated copy of the simulated robot with measured part weights,
without editing the upstream model. It has its own README and its own small Python
environment. It needs three inputs: each printed part's weight, which parts sit on which of
the model's bodies, and the non-printed weight (servos, screws, boards) on each body.

## Licences of what this builds on

- jjjadand/microduck_rl: Apache-2.0.
- AI-FanGe replica: MIT at the root, but its on-robot code (`microduck/`) is GPL-3.0 and its
  CAD is CC BY-NC-SA. Check these before publishing or selling anything derived from them.

## Layout

```text
third_party/microduck_rl/  upstream training repo, git submodule at commit 25f7624 (read-only)
duck_train/                our small Python scripts that reuse upstream code
  model_check.py           load the robot model headless, print joints/actuators/mass
  walk_headless.py         run an official ONNX walking policy headless, write video + trace
  replica_config.py        every replica-specific value (IMU mount, servo kp, battery PLACEHOLDER)
  replica_task.py          registers Mjlab-Velocity-Flat-MicroDuck-Replica (51-input policy)
  train.py                 upstream's train command, plus our replica task
  run_training.py          launcher: --envs/--iterations/--device/--name, logs in out/<name>/
  export_policy.py         upstream's export, plus our task, plus checks for the robot's walk.py
  replay_policy.py         run an exported ONNX headless in our task, report falls
tests/                     our tests (run in upstream's environment)
print/                     print plates and part list (see print/README.md)
tools/                     standalone helper tools
tasks/todo.md              plan and status
out/                       generated output (git-ignored)
external/                  read-only reference clones (git-ignored)
```

## Setup (Mac, Apple Silicon)

Needs `uv` (installed with Homebrew). uv fetches Python 3.12 itself.

```bash
git submodule update --init                     # after a fresh clone
cd third_party/microduck_rl && uv sync          # creates third_party/microduck_rl/.venv (Python 3.12)
```

The environment is `third_party/microduck_rl/.venv`, built from upstream's `uv.lock`.
Run our scripts in it from the repo root with `uv run --project third_party/microduck_rl ...`.
Do not use upstream's Jetson install script on the Mac.

## Jetson setup

`jetson/setup_jetson.sh` sets up a Jetson Orin on JetPack 7.2 for GPU training; see
`jetson/README.md`. It is a dry run unless given `--apply`. It never runs `sudo` itself: it
prints the system-package commands for you to run. Tested only as a dry run on the Mac so far.

## Tests (CPU)

```bash
cd third_party/microduck_rl && uv run --with pytest pytest tests/ -q     # 149 passed, ~20 s
```

## See it walk (headless)

From the repo root:

```bash
uv run --project third_party/microduck_rl python duck_train/model_check.py
uv run --project third_party/microduck_rl python duck_train/walk_headless.py
```

`walk_headless.py` runs `pretrained/pollen-robotics/alpha_walking.onnx` for 10 s at a 0.3 m/s
forward command. It writes `out/walk_official.mp4` and `out/walk_official_trace.csv`.
Options: `--onnx`, `--lin-vel-x`, `--seconds`, `--no-video`. Starting from standing, this
policy does not step at commands of 0.2 m/s or less.

## Training smoke test (CPU)

Upstream defaults to GPU 0; `--gpu-ids None` selects the CPU. Logs go to `logs/` under the
current folder, so run it from `out/`:

```bash
mkdir -p out/train_smoke && cd out/train_smoke
uv run --project ../../third_party/microduck_rl train Mjlab-Velocity-Flat-MicroDuck \
  --gpu-ids None --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard
```

This took about 2.7 s per iteration on the CPU. That is fine for checking that a task builds;
it is too slow for real training.

## Replica task (51-input policy for the community replica)

`Mjlab-Velocity-Flat-MicroDuck-Replica` is upstream's flat walking task with four changes for
the replica's on-robot code: 51 policy inputs (no head/body commands), gyro and gravity in the
vertically mounted IMU's frame, servo kp 125 instead of 200, and a battery voltage range.
All replica values, with their sources, are in `duck_train/replica_config.py`. The battery range
there is a PLACEHOLDER (6.0-7.5 V) until the battery is chosen.

`duck_train` is not installed into upstream's environment (a `uv sync` there would remove it, and
upstream's `train` only finds installed packages). `duck_train/train.py` registers the task and
then runs upstream's own train entry point with all arguments passed through. Run it from `out/`:

```bash
mkdir -p out/replica_smoke && cd out/replica_smoke
uv run --project ../../third_party/microduck_rl python ../../duck_train/train.py \
  Mjlab-Velocity-Flat-MicroDuck-Replica \
  --gpu-ids None --env.scene.num-envs 16 --agent.max-iterations 2 --agent.logger tensorboard
```

This finished in about 10 s and wrote `logs/rsl_rl/velocity_replica/<run>/<run>.onnx` with a
51-wide input. Upstream's own `uv run train` / `list-envs` do not list the replica task.

Tests (from the repo root; about 1 minute, including two small training runs and an export):

```bash
uv run --project third_party/microduck_rl --with pytest pytest tests/ -q -s     # 67 passed, ~60 s
```

Add `-m "not slow"` to skip the train-export-replay tests in `tests/test_training_tools.py`.

## Train, export, replay (any machine; run from the repo root)

`duck_train/run_training.py` wraps the command above. Each run's logs go to `out/<name>/`
whatever the current folder is. The commands below use repo-relative paths, so run them from
the repo root; from elsewhere, give both paths in full
(`uv run --project <repo>/third_party/microduck_rl python <repo>/duck_train/run_training.py ...`).
It refuses to reuse a name unless `--resume` is given, prints the exact upstream command it
runs, and passes unknown options to upstream unchanged (except the flags it sets itself, in
any spelling). If a launch fails before any checkpoint is saved, it removes the folder that
launch created, so the name can be used again.
`--device auto` (the default) uses the GPU if torch sees one; `cpu` maps to upstream's
`--gpu-ids None`, `cuda` to `--gpu-ids [0]`; `--device cuda` stops at once if torch sees no GPU. `--task Mjlab-Velocity-Flat-MicroDuck` trains the
official 61-input task instead. `--dry-run` prints the command only.

```bash
# Smoke test, 64 simulations x 5 iterations (about 15 s on the Mac CPU)
uv run --project third_party/microduck_rl python duck_train/run_training.py --envs 64 --iterations 5 --name smoke
# Jetson Orin Nano 8 GB (NOT RUN YET): first 512 simulations, then 1024 if memory allows.
# On the Jetson EVERY `uv run` needs `--no-sync` (export and replay too): a plain `uv run`
# re-syncs the lock file and puts the CPU-only torch back (see jetson/README.md).
uv run --no-sync --project third_party/microduck_rl python duck_train/run_training.py --envs 512 --iterations 1000 --device cuda --name orin512
uv run --no-sync --project third_party/microduck_rl python duck_train/run_training.py --envs 1024 --iterations 1000 --device cuda --name orin1024
# Continue a run for 1000 more iterations from the newest run folder that has a checkpoint
# (it prints which folder and checkpoint; an interrupted resume that saved nothing is skipped)
uv run --no-sync --project third_party/microduck_rl python duck_train/run_training.py --envs 1024 --iterations 1000 --device cuda --name orin1024 --resume
```

The iteration counts for the Jetson are a starting point to time, not a tuned value (upstream's
own default is 50,000 at 4,096 simulations). Checkpoints are saved every 250 iterations and at
the end.

Export the newest checkpoint of a run to ONNX with upstream's export code (input
normalisation and metadata included), then check it against what the robot's `walk.py` reads
(input [1, 51], 14 outputs, joint names and default pose, action scale 1.0). It writes
`model_<N>.onnx` next to the checkpoint and exits non-zero if a check fails:

```bash
uv run --project third_party/microduck_rl python duck_train/export_policy.py --run smoke
```

Replay it headless in the replica task: 32 robots side by side for 10 s at 0.2 m/s forward, no
pushes, seed 1 (`--envs`, `--seconds`, `--lin-vel-x`, `--seed`). It reports "N of 32 robots
fell (first at step S)", with NaN resets counted apart from falls; the same seed gives the same
counts. After 5 iterations the robots fall; this checks the file runs, not that it walks. For
comparison, the replica's published `walk.onnx` measured 2 of 32 falling with seed 1 (about 6 %):

```bash
uv run --project third_party/microduck_rl python duck_train/replay_policy.py \
  --onnx out/smoke/logs/rsl_rl/velocity_replica/<run>/model_4.onnx
```

The `-s` output includes what the replica's published `walk.onnx` does in this task; the same
line is written to `out/replica_walk_onnx_report.txt`.

## Interactive viewer (NOT RUN — no GUI access when this was written)

On macOS the MuJoCo viewer window needs `mjpython`. Upstream's keyboard demo reads keys
from the terminal (arrow keys to steer, Space to stop, Q to quit):

```bash
cd third_party/microduck_rl
uv run mjpython scripts/infer_policy.py \
  --walking pretrained/pollen-robotics/alpha_walking.onnx \
  --standing pretrained/pollen-robotics/alpha_stand.onnx \
  --new-cmd-obs
```
