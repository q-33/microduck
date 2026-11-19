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

## RESUME HERE (written 2026-11-18, 23:30) — read this first

State: `main` pushed at 34ea2a6. Everything below is UNCOMMITTED (commit only when the owner asks):
`duck_train/run_training.py`, `export_policy.py`, `replay_policy.py`, `tests/test_training_tools.py`,
`tests/conftest.py`, `jetson/` (setup script + README), `README.md`, this file.

Done 2026-11-18 (late):
- Step 6, training wrapper + export + replay: built, run on the Mac CPU only, committed (bafa412).
  INDEPENDENT REVIEW DONE 23:40: export is bit-identical to upstream's training-end ONNX with normalisation
  inside the graph; 32 tests pass. DEFECTS TO FIX (not fixed yet, do first):
  1. Flag guard misses underscore spellings that upstream's parser accepts (`--env.scene.num_envs 8`,
     `--gpu_ids None`): the launcher prints one thing and trains another. Normalise `_` to `-`; add tests.
  2. `--resume` crashes after an interrupted resumed run: upstream picks the newest run folder, which has no
     checkpoint until the next multiple of 250 iterations. Pass `--agent.load-run <newest folder WITH a
     checkpoint>`. Will hit long Jetson runs.
  3. A failed start leaves an empty `out/<name>/` that then blocks the name; `--device cuda` without a GPU
     fails with an upstream IndexError instead of a clear message.
  4. Exporting a checkpoint outside the repo writes its absolute path into the ONNX `run_path` metadata.
  5. Replay has no `--seed` and defaults to one robot, so "stayed up" is noisy.
  6. Small: use `fullmatch` for run names (trailing newline accepted); clean argparse errors for bad
     `--envs`/names; warn when exporting a 61-input official policy (the replica runtime would misread it);
     README says "from anywhere" but the `uv run --project` path is relative; README timing line is stale.
  Review's throwaway runs are in `out/verify-*` (ignored).
- walk.onnx loose end SETTLED: the task is unseeded; the replica's published policy falls in about 6 % of
  10 s single-robot runs without pushes (8/128) and about 23 % with pushes (15/64). "Never fell" earlier was
  chance. Correct the step 3 note above when next editing.
- Step 5, `jetson/setup_jetson.sh`: dry-run by default, `--apply` only on a Jetson, never runs sudo (prints
  the commands), deletes nothing, official sources only. shellcheck clean; simulated-Jetson dry-runs accept
  JetPack 7.2 / L4T 39.2.0 / CUDA 13.2.x and stop on 7.2.1, JetPack 6, wrong CUDA, non-Orin.
  NOT independently reviewed yet (held back to save usage): do that before first use on the board.
  Upstream only ran this stack on an Orin NX 16 GB; the Orin Nano 8 GB is unconfirmed.
- On the Jetson EVERY `uv run` needs `--no-sync` (a plain one reinstalls CPU-only torch). README fixed.
- Step 6 of the old list (independent review of `replica_task.py`): done, all four requirements met, IMU
  frame agrees with the robot code numerically. Finding to act on: the robot's stick limits (vx +0.7/-0.5,
  vtheta 1.5 moving / 3.0 in place) are wider than the trained command range (0.4, 0.3, 1.0): cap on the
  robot or widen training. Symmetry augmentation must stay off for this task.

Jetson on the duck (plan further down, APPROVED): A fit study RUNNING at time of writing (result not seen);
B licence request posted (issue 1 on the mount repo, awaiting the author); C payload-in-training NOT started
(needs the wrapper reviewed and the mount's mass figures from the fit study); D power and E print not started.

Next, in order:
1. Fix the six wrapper defects above (opus), re-run the tests, commit. Read the fit study report if it
   arrived, or re-run it.
2. Independent review of `jetson/setup_jetson.sh` (opus).
3. Step C: payload model via `tools/duck_mass` + task `...-Replica-Loaded`.
4. Owner: slice and print `print/prusa_xl_petg/XL01_test_piece.3mf` (head 4 PETG, head 5 PLA supports);
   check XL330 fit and M2 screw bite; then XL02, XL03; weigh parts into a copy of
   `tools/duck_mass/weights.template.csv`.
5. Small fixes still open: duck_mass writes absolute paths; tests should skip cleanly when `external/` is
   absent; load a duck_mass model through the training task.

Waiting on others / owner:
- Battery choice (sets PACK_VOLTAGE_RANGE_V; check the XL330 voltage rating in the ROBOTIS manual first).
- Reply on https://github.com/AI-FanGe/Microduck-build-tutorial/issues/17 (missing training robot model).
  Read closed issue 12 there (which battery) in a browser.
- Which Orin Nano variant and JetPack version, on arrival.
- Owner pointed (2026-11-18) at an existing on-robot Jetson mount: https://github.com/adammong/microduck-jetson-mount
  (v07: five PETG parts in `clearance_mount/STL/`, saddle + front gate + tray, Jetson in front with its OEM base,
  4S LiHV 850 mAh pack at the rear for the Jetson; MuJoCo payload model and mass properties included).
  Read 2026-11-18 (README, clearance_mount/README, THIRD_PARTY only; no files copied). Facts that matter to us:
  payload estimate 447.6 g on a ~737 g robot; everything is simulation only, its author says physical fit,
  strength and hardware walking are untested; it is built on the OFFICIAL Pollen model and 61-input alpha
  policies, so its ONNX files do not fit our replica's 51-input runtime; the repo has NO licence, so link to
  it, do not copy its files into our public repo.
  OWNER DECIDED 2026-11-18 23:20: the Jetson RIDES ON THE DUCK. Plan below (AWAITING OWNER'S YES).

## Plan: Jetson on the duck (written 2026-11-18, APPROVED by the owner 23:21)

Status 23:25: B posted as https://github.com/adammong/microduck-jetson-mount/issues/1 (awaiting the author).
A (fit study) running as a sub-agent. C starts when the training wrapper (step 6) has landed.

Goal: our replica walks carrying the Orin Nano and its battery on the adammong v07 mount, with a 51-input
policy trained for that load.

- [ ] A. Fit study, read-only: download the mount repo outside ours (git-ignored `external/`), compare its
      shell/torso reference (official Pollen model) with OUR replica's printed torso and front shell
      (`print/parts.csv`, the replica STEP/STLs). Check: a written list of the contact surfaces the mount uses
      (shoulder seats, front shell lobes, two torso M4 bosses at +-40 mm) and whether each exists on our parts.
- [ ] B. Licence: the mount repo has none. Ask its author (GitHub issue, owner's yes needed before posting)
      to add one. Until then: print for our own use, link only, nothing of theirs in our repo.
- [ ] C. Payload in training: add the mount + Jetson + battery as a rigid payload on the trunk in a GENERATED
      copy of the model (extend `tools/duck_mass`; mass, centre of mass and inertia from the mount's
      `simulation/mass_properties.json`, replaced by weighed values later), and a task variant
      `...-Replica-Loaded` that randomises payload mass about +-10 %. Check: generated model total mass =
      bare mass + payload; the bare task is unchanged; smoke training runs.
- [ ] D. Power: decide how the Jetson and the servos are fed (mount assumes a 4S LiHV 850 mAh pack for the
      Jetson only; the servo battery is still not chosen; XL330 supply limits from the ROBOTIS manual). Owner
      decision, no code.
- [ ] E. Print: add the five mount STLs to a new XL plate (PETG; saddle and gate need supports) AFTER the
      bare duck's test piece and plates have printed and fitted.
- [ ] F. Order of proof: bare duck walks on hardware first; then loaded policy in sim; then mount on the robot.

Risks to keep in view: payload is about 60 % of the bare robot's mass and is untested on hardware by anyone;
the mount was designed against the official shell, not the replica's; its policies are 61-input and do not
apply to us.
- Licence for our own code (NOTICE.md says none chosen).
- On the real robot: upright, the policy's gravity input should read about (1, 0, 0).
