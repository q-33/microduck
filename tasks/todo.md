# MicroDuck: plan and status

## Status (2026-11-18)

- Own git repo created here (branch `main`). Nothing committed. No GitHub remote.
- Print plates made and checked by script and PrusaSlicer's command line: see `print/README.md`.
  Not sliced, not opened in the PrusaSlicer window, nothing printed.
- Both training codebases cloned read-only into `external/` (git-ignored) and compared.
  Nothing from them installed or run.
- Asked the replica's author for the missing training robot model:
  https://github.com/AI-FanGe/Microduck-build-tutorial/issues/17
- Owner decisions: first robot in PETG on the club's Prusa XL (head 4 PETG, head 5 PLA supports);
  nylon maybe later for critical parts; IMU mounted vertically; battery not chosen yet.

## Plan: training software for the Jetson Orin Nano (AWAITING OWNER'S YES)

Goal: a walking policy for OUR replica, trained on the Orin Nano, exported to ONNX with the
51-input interface the replica's on-robot code expects.

Approach: build on jjjadand/microduck_rl (pinned at commit 25f7624) without editing it; our code
is a thin layer beside it. Chosen over the replica's own training folder because that one is
missing its robot model and cannot train as published.

- [x] 1. Mac environment. Pin microduck_rl as a submodule; create the Python 3.12 environment with uv.
      Check: the repo's own CPU tests pass on the Mac.
      Verified 2026-11-18: submodule staged (not committed) at `third_party/microduck_rl`, commit
      25f7624. `uv sync` there (uv 0.13.9) made `third_party/microduck_rl/.venv` (Python 3.12.16,
      ignored by upstream's own .gitignore; submodule stays clean). `uv run --with pytest pytest
      tests/ -q`: 149 passed, 0 failed, ~20 s.
- [x] 2. See it move on the Mac. Open the robot model in the MuJoCo viewer and play the official
      ONNX policy in simulation.
      Check: a recorded clip of the duck walking in sim.
      Verified headless 2026-11-18: robot_walk.xml has 14 hinge joints, 14 actuators, 0.7372 kg.
      `duck_train/walk_headless.py` (reuses upstream's infer_policy.py) ran alpha_walking.onnx for
      10 s at vx command 0.3 m/s: no fall, base height 0.113-0.123 m, max tilt 4.0 deg, moved
      +1.22 m forward and 0.27 m sideways (about 0.12 m/s). From standing it does not step at commands
      of 0.2 m/s or less (stays put); it walks at 0.25-0.4. Clip: `out/walk_official.mp4` (10 s, 640x480).
      Interactive viewer NOT run (no GUI access); command in README.
      CPU training works: walking task, 64 envs x 5 iterations, device=cpu, ~2.7 s per iteration
      (~560 env-steps/s); output in `out/train_smoke/`.
- [ ] 3. Replica task. A new training task that differs from the official one in four ways:
      51 observations (no head/body commands), the vertical-IMU rotation, servo stiffness 125,
      and a supply-voltage range set from one battery setting (placeholder until the battery is chosen).
      Check: a test that the observation vector has 51 entries in the replica's order, and that
      the replica's published walk.onnx runs in this task without a shape error.
- [ ] 4. Part-weight tool. Reads `print/parts.csv` plus weighed masses and writes a generated copy
      of the robot model with those masses (never hand-editing the exported model files).
      Check: total mass of the generated model equals the sum of the inputs.
- [ ] 5. Jetson setup script, our own (the upstream one uses a third-party mirror, takes a sudo
      password from the environment and can delete a folder). It checks the JetPack/CUDA version
      first and stops with a clear message if it is not the one the GPU install steps assume.
      Check: shell lint and a dry-run mode on the Mac; real run waits for the Jetson.
- [ ] 6. Training wrapper with one setting for parallel simulations (64 for the smoke test, then
      512 to 1,024 on the 8 GB board), and ONNX export.
      Check on arrival: the 64-simulation, five-iteration smoke test, then export and replay in sim.

Open, not blocking steps 1 to 4: battery voltage; which JetPack the Orin Nano gets; whether the
replica's dimensions match the official model (issue 17); real weights of PETG parts.
