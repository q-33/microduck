# Credits and licences

This repository builds on other people's work. Their licences apply to their parts.

- **Print files** (`microduck_print_plates.3mf`, `microduck_split.zip`, `NylonG_PulseXE/`, and the
  plates and STLs under `print/`) are the parts of the community Microduck replica by AI-FanGe,
  from https://github.com/AI-FanGe/Microduck-build-tutorial, re-arranged onto print plates with
  the geometry unchanged. That repository is MIT-licensed at its root; its `microduck/` folder
  states GPL-3.0 for software and CC BY-NC-SA for CAD. Treat the print files here as
  CC BY-NC-SA (credit the author, non-commercial, share alike) unless the author says otherwise.
- **Training code** is used, unmodified, from https://github.com/jjjadand/microduck_rl
  (Apache-2.0), included as a git submodule under `third_party/`.
- **The replica task** in `duck_train/` follows the observation layout and constants of the
  replica's on-robot code (GPL-3.0) so that trained policies match it.
- The Microduck robot itself is a design by Pollen Robotics.

No licence has been chosen yet for the original code and documents in this repository.
