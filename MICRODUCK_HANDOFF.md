# Microduck build — project handoff

Exported 2026-11-18 (second export, 21:38 PT; adds the Jetson training
station) from a planning chat that began 2026-11-15, so Claude Code can pick
up where it left off. Everything below was decided or found in that chat. Items
marked UNVERIFIED were not tested and should be checked before relying on them.

## Goal

Build a Microduck biped robot at the East Bay Makers Club, then go beyond the
stock build by adding AI (vision, voice, cloud LLM).

## Key facts about the source design

- The official Microduck (Pollen Robotics, ~$399 assembled) has open software
  but the mechanical/hardware files are NOT published.
- We are building the community replica instead:
  https://github.com/AI-FanGe/Microduck-build-tutorial (MIT licence).
  Build video: https://www.youtube.com/watch?v=Vep8AjoCnEM
- Stock replica electronics: 14x Dynamixel XL330-M288-T servos (15 with the
  beak), ROBOTIS OpenRB-150 servo controller, Raspberry Pi Zero 2 W, BNO085 IMU.
- Walking is a neural-network policy (ONNX) run on the main board; the repo
  also contains the training environment.
- The repo's README has older instructions for a different joint layout and
  battery mixed in. Follow the root README.
- The repo's gamepad mode turns Wi-Fi off. That conflicts with cloud AI and
  needs changing.

## Decisions made

1. **Printing split by material**
   - NylonG (glass-filled nylon) for structural parts, on the user's own
     MatterHackers Pulse XE (250 x 220 mm bed), hosted at the club.
   - PLA for cosmetic parts and small bearing parts, on the club's Prusas.
2. **Main board: Radxa Zero 3W (4 GB, with GPIO header)** instead of the Pi
   Zero 2 W. Same 65 x 30 mm footprint, RK3566 with a 0.8 TOPS NPU.
   - Rejected: Raspberry Pi 5 + AI HAT+ 2. Too big for the body, needs a
     redesign, more power, and probably a retrained walking policy; its
     on-board LLMs (1-1.5B parameters) are weak anyway.
3. **AI architecture**
   - Walking policy stays on the robot, on the CPU (so Wi-Fi loss can't make
     it fall).
   - Vision on the Radxa NPU (RKNN format).
   - Conversation / heavy reasoning via a cloud model over Wi-Fi. The robot
     sends audio and frames, receives high-level commands (walk, turn, look).
4. **Recommended (not yet confirmed by the user):** also buy a Pi Zero 2 W
   (~$15) and get the robot walking on the repo's prebuilt image first, so
   hardware faults and software-port faults are not debugged at the same time.

5. **Training station: NVIDIA Jetson Orin Nano (ordered 2026-11-18, in
   transit).** It stays on the bench and is used only for reinforcement
   learning. It does NOT go on the robot and does not do motion control:
   the dev kit is about 100 x 79 x 21 mm and draws 7-25 W. Policies trained
   on it are exported to ONNX and run on the Radxa's CPU.

## Jetson Orin Nano: RL training station

- Exact variant not stated by the user. Confirm on arrival (Orin Nano Super
  dev kit has 8 GB shared CPU/GPU memory; the box has no storage, so a
  microSD of 64 GB+ or an NVMe SSD is needed).
- Reference workflow: Seeed's "Microduck RL on Jetson" tutorial
  https://wiki.seeedstudio.com/ai_robotics_microduck_rl_on_jetson/
  using https://github.com/jjjadand/microduck_rl (source, deployment script,
  official ONNX policies, Jetson-trained checkpoints). It does PPO training
  in MuJoCo, checkpoint visualisation, ONNX inference and custom motions.
- That tutorial was verified on a Jetson Orin NX 16GB with JetPack 7.2,
  Ubuntu 24.04, CUDA 13.2, Python 3.12, MuJoCo 3.10.
- UNVERIFIED on an Orin Nano: the tutorial does not mention it. Shared 8 GB
  will limit the number of parallel environments. Start from the tutorial's
  64-environment, five-iteration smoke test and scale up.
- The tutorial gives no training times or throughput figures.
- Its one-leg balance demo failed; the only validated custom motion is a
  front-back split.
- The tutorial uses the OFFICIAL Microduck model. Our robot is the AI-FanGe
  replica with NylonG structural parts, so the simulation model needs our
  measured part masses before a trained policy can be trusted on hardware.
  The AI-FanGe repo also ships its own training environment; decide which
  of the two codebases to train from.
- A Jetson is a slow trainer next to a desktop NVIDIA GPU or a rented cloud
  GPU. That comparison was raised but not done.

## Print files (already produced)

`microduck_split.zip` contains:

| File | Material / printer | Contents |
|---|---|---|
| `NylonG_PulseXE/N0_test_piece.3mf` | NylonG / Pulse XE | One servo bracket. Print first. |
| `NylonG_PulseXE/N1_frame_1.3mf`, `N1_frame_2.3mf` | NylonG / Pulse XE | Other 18 "gray structural" parts |
| `NylonG_PulseXE/N2_legs_feet_1.3mf` | NylonG / Pulse XE | 2 leg links + 2 foot housings |
| `PLA_Prusa/P1_shells_beak_1.3mf` | PLA / Prusa | Big shell, 2 covers, 2 beak parts |
| `PLA_Prusa/P2_soles_ring_1.3mf` | PLA / Prusa | 2 soles, 1 ring |
| `PLA_Prusa/P3_bearings_horns_1.3mf` | PLA / Prusa | 34 rings/horns/caps + 2 spare idle horns |
| `*/individual_STL/` | — | Every part as its own STL, print-oriented |

Totals: 23 NylonG parts, 44 PLA parts (67 in the source file).

`microduck_print_plates.3mf` is the untouched original Bambu Studio project
from the repo (5 plates, sliced for a Bambu P1S).

How the split was made:
- Parts keep the author's print orientation.
- The 3MF files hold geometry and placement only, NO slicer settings.
  Author's settings: 0.4 mm nozzle, 0.2 mm layers, 2 walls, 15% infill,
  supports on.
- File names are `plate<orig plate>_<label>_<orig name>_id<n>`. The labels
  (frame, leg-link, cover...) are a reading of the geometry, not the author's
  names. The original part names are gibberish.
- Largest part is about 123 x 92 x 46 mm. Tallest PLA part is 116 mm.

UNVERIFIED about the print files:
- Never opened in a slicer. Check the first import.
- Not sure MatterControl opens 3MF; the STLs are the fallback.
- PLA beds were laid out for 250 x 210 mm (Prusa MK3/MK4 size); the club's
  actual Prusa models are unknown.
- The two triangular "covers" were left in PLA because they only have three
  screw bosses. If the build video shows them carrying a servo, print them in
  NylonG.
- The two spare idle horns sat off-plate in the original; may not be needed.
- Plate 4 (rings/horns) was labelled structural by the author but kept in PLA
  on the judgment that glass fibre makes a poor sliding surface and thin
  rings warp in nylon. Not tested.

## NylonG print notes

- MatterHackers figures: extruder 250-265 C, bed 60-65 C (65 for NylonG),
  glue stick on garolite, no layer fan, no drafts.
- Dry the filament before every print; print from the dryer if possible.
- Abrasive: needs a hardened or ruby nozzle, not brass.
- The design was tuned for PLA. Use the test piece to check XL330 fit, M2
  screw bite and shrink; if undersized, apply an XY scale to all NylonG parts
  so the PLA shells still line up.
- The walking policy was trained around the PLA build's mass. Weigh nylon
  parts against the slicer's PLA estimate.

## Shopping list (current state)

Prices are rough recollections from the chat, NOT checked against vendors.

| Part | Qty | Rough cost | Notes |
|---|---|---|---|
| Dynamixel XL330-M288-T | 14 (15 with beak) | ~$25 each | Longest lead time; order first |
| ROBOTIS OpenRB-150 | 1 | ~$25 | Servo bus controller |
| Radxa Zero 3W, 4 GB, GPIO header | 1 | — | One US seller was out of stock |
| Small heatsink for the Radxa | 1 | — | Board runs very hot without one |
| Pi Zero 2 W (optional baseline) | 1 | ~$15 | Was reported widely sold out in Sept 2026 |
| BNO085 IMU breakout (I2C) | 1 | ~$20-25 | |
| microSD card, 32 GB | 1-2 | ~$8 | |
| "6 V" rechargeable pack | 1 | ? | See open issues |
| 5 V regulator, 3 A or better | 1 | ~$5 | Radxa spec: 5 V / 2 A or higher |
| Power switch | 1 | ~$3 | |
| USB-C to USB-C data cable | 1 | ~$6 | OpenRB-150 to Radxa USB 3.0 host port |
| Camera, 22-pin 0.5 mm ribbon | 1 | — | IMX219 was used in Radxa NPU tests; confirm support |
| I2S MEMS mic + I2S amp + speaker | 1 set | — | For voice |
| NVIDIA Jetson Orin Nano | 1 | ORDERED | Bench RL trainer; dev kit listed at $399 in Nov 2026 |
| microSD 64 GB+ or NVMe SSD for the Jetson | 1 | — | Not included with the dev kit |
| M2 / M2.5 self-tapping screws, POM/steel shims | assorted | ~$15 | |
| Dynamixel 3-pin cables, JST wire | spares | ~$15 | |
| U2D2 + Power Hub (optional) | 1 | ~$50 | For setting servo IDs |

Not yet done: a priced sheet with US vendor links (was offered, not accepted
yet).

## Radxa Zero 3W wiring notes

- Power: 5 V into GPIO pins 2 and 4 (a supported input), leaving both USB
  ports free.
- OpenRB-150 on the USB 3.0 host port (capped at 500 mA; servos are powered
  from the battery side).
- IMU on the header's I2C pins; the bus number will probably differ from the
  Pi's, so expect a config change.
- Camera: 22-pin, 0.5 mm pitch MIPI CSI connector.
- NPU must be enabled in `rsetup` overlays. Models are converted to RKNN on
  an x86 Linux PC with rknn-toolkit2. Ultralytics documents YOLO export for
  this board.

## Build phases

1. Order servos; print N0 test piece; adjust; print the rest.
2. Set servo IDs 1-14 one at a time, label them, centre before fitting horns.
3. Assemble: feet, legs, body, neck, head. Dry-fit PLA shells on the nylon
   frame before installing servos.
4. Wire power, the three servo chains and the IMU.
5. (Recommended) Walk on a Pi Zero 2 W with the repo's prebuilt image.
6. Port to the Radxa: Radxa Debian image, clone the repo, install its Python
   environment, fix IMU bus and serial device names, run the same policy on
   the CPU.
7. Add vision on the NPU.
8. Add mic/speaker and the cloud AI link; change the gamepad/Wi-Fi behaviour.

In parallel, once the Jetson arrives: flash it, run the tutorial's smoke
test, then weigh the printed parts, update the simulation model and retrain.

## Open issues and risks

- **Battery:** the README names no model, chemistry or capacity, only that
  it must handle peak current. Measure fully charged voltage against the
  XL330's limit before connecting. Capacity may need to grow for the Radxa,
  camera and audio.
- **Nobody has reported this replica walking on a Radxa Zero 3W.** The code
  is plain Python and should port; that is an expectation, not a tested fact.
- **NPU vision is slow:** forum users reported roughly 2-6 fps for object
  detection in simple setups.
- **Mounting** for camera, mic and speaker in the printed head has not been
  looked at.
- **Self-tapping screws into glass-filled nylon:** no reports found either way.
- **Club Wi-Fi** suitability not checked.
- **Orin Nano for training** is untested for this workflow (see the Jetson
  section). Two training codebases exist and one has to be chosen.

## Good first tasks for Claude Code

- Clone the repo and read the deployment code: find where the IMU bus, serial
  device and Wi-Fi/gamepad behaviour are set, and list what must change for
  the Radxa.
- Draft the robot-side "high-level command" interface for the cloud brain.
- Build the priced shopping sheet.
- Read jjjadand/microduck_rl and the Seeed tutorial; write the Jetson setup
  steps for the Orin Nano and note where its requirements differ from the
  verified Orin NX setup.
- Compare the robot model in jjjadand/microduck_rl with the one in the
  AI-FanGe repo and list what must change for the NylonG replica.

## Files on the Mac

`~/claudecode/ebmc/microduck/` holds this file, `microduck_split.zip` and
`microduck_print_plates.3mf`. The parent folder `~/claudecode/ebmc` is a git
repo of MatterControl source (the Pulse XE's slicer) with its own `EBMC.md`,
which was not read when this handoff was written. The microduck files are
untracked there.

## Sources

- https://github.com/AI-FanGe/Microduck-build-tutorial
- https://openelab.io/de/blogs/robotics/microduck-build-guide-openelab
- https://cnx-software.com/2026/08/28/microduck-a-duck-like-biped-robot-designed-for-physical-ai-experimentation-and-fun
- https://www.matterhackers.com/store/c/browse-by-filament-manufacturer/nylong
- https://www.matterhackers.com/articles/printing-with-nylon
- https://docs.radxa.com/en/zero/zero3/hardware-design/hardware-interface
- https://dl2.radxa.com/zero3/docs/hw/3w/radxa_zero_3w_product_brief.pdf
- https://docs.radxa.com/en/rock3/e25/app-development/rknn_toolkit_lite2_yolov5
- https://docs.ultralytics.com/integrations/rockchip-rknn/
- https://wiki.seeedstudio.com/ai_robotics_microduck_rl_on_jetson/
- https://github.com/jjjadand/microduck_rl
- https://physicalailab.io/en/jetson-orin-nano-super-developer-kit-price-whats-in-the-box/
- https://www.cnx-software.com/2026/01/15/raspberry-pi-ai-hat-2-targets-generative-ai-llm-vlm-with-hailo-10h-accelerator/
