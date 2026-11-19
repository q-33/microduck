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
- [x] 3. Replica task. A new training task that differs from the official one in four ways:
      51 observations (no head/body commands), the vertical-IMU rotation, servo stiffness 125,
      and a supply-voltage range set from one battery setting (placeholder until the battery is chosen).
      Check: a test that the observation vector has 51 entries in the replica's order, and that
      the replica's published walk.onnx runs in this task without a shape error.
      Done 2026-11-18 (sim only): `duck_train/replica_config.py`, `replica_task.py`, `train.py`;
      task `Mjlab-Velocity-Flat-MicroDuck-Replica`. `tests/test_replica_task.py`: 10 passed
      (51 inputs in walk.py order; kp 125 and voltage range in the built env; official task still
      61; sim IMU terms equal the replica runtime's formulas; walk.onnx [1,51] runs 500 steps;
      16-env x 2-iteration CPU training exports a 51-wide ONNX). Upstream tests still 149 passed.
      Finding: the review had the IMU backwards. The replica feeds the policy gyro and the
      world UP vector in the SENSOR frame (observer.py:81-85, walk.py:101-102), and its training
      rotates the sim's body-frame values into that frame. So the sim DOES need the rotation (and a
      sign flip on gravity); implemented that way.
      walk.onnx in our task (no pushes, 8 envs, 10 s): never fell; with body-frame IMU inputs instead
      it fell ~15 times per env, which supports the frame choice. It walks forward ~0.2-0.3 m/s
      even when commanded to stand and drifts in yaw, so it does not track commands on this model.
      NOT verified, needs the real robot: the BNO08x quaternion/gyro axis convention and the mount
      quaternion for OUR build (stand the robot upright, check the policy input reads about (1, 0, 0));
      the battery range (placeholder 6.0-7.5 V from the replica's constants; its README says a 6 V
      pack); whether upstream's robot model matches the replica's (issue 17).
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

## RESUME HERE (written 2026-11-18, 22:35)

State: steps 1-4 done, committed and pushed (public repo, `main` at f492949). Steps 5-6 not started.

Next, in order:
1. Owner: slice and print `print/prusa_xl_petg/XL01_test_piece.3mf` (parts on head 4 PETG, supports on
   head 5 PLA, wipe tower moved to the back-right corner). Check XL330 fit and M2 screw bite. Confirm the
   slicer's extruder numbers match heads 4 and 5 on the printer.
2. If the bracket fits: print XL02 then XL03 (or split XL03; it is one long 44-part print). Weigh every
   part and record it in a copy of `tools/duck_mass/weights.template.csv`.
3. Step 5: our own Jetson setup script (checks JetPack/CUDA first; no mirror, no sudo password in the
   environment, no deleting folders). Dry-run on the Mac.
4. Step 6: training wrapper with one setting for parallel simulations (64, then 512-1024), and export
   through upstream's `scripts/export.py` so input normalisation is baked in.
5. Small fixes: duck_mass writes absolute paths (generated model and template header) - make them
   relative/configurable so it works on the Jetson; make the tests skip cleanly when `external/` is absent;
   load a duck_mass model through the training task, not only plain MuJoCo.
6. Have a separate reviewer read `duck_train/replica_task.py` against the replica's on-robot code
   (the IMU frame change especially). It was tested but not independently reviewed.

Waiting on others / owner:
- Battery choice (sets PACK_VOLTAGE_RANGE_V; check the XL330 voltage rating in the ROBOTIS manual first).
- Reply on https://github.com/AI-FanGe/Microduck-build-tutorial/issues/17 (missing training robot model).
  Read closed issue 12 there (which battery) in a browser.
- Which Orin Nano variant and JetPack version, on arrival.
- Licence for our own code (NOTICE.md says none chosen).
- On the real robot: upright, the policy's gravity input should read about (1, 0, 0).
