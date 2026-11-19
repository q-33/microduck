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
| Replica training task | In progress. |
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
