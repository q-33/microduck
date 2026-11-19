# duck_mass

Puts weighed printed-part masses into the Microduck MuJoCo model. It writes a
**generated copy** of the model; the upstream export in `external/` is never
edited (a re-export would overwrite hand edits).

## How it works

Each body in the model has one lumped mass: printed parts + servos + bearings,
screws, PCBs. Nothing in the files says which printed part sits on which body,
so you supply that in a mapping file:

    new body mass = sum(printed parts on the body) + non_printed_g

- **Printed part grams** come from the weights CSV (`part_file,measured_g`).
  A part with a blank or missing weight uses an estimate column from
  `print/parts.csv` (default `petg_g_rough_author_settings`, change with
  `--fallback-column`). The report names every body that used an estimate.
- **non_printed_g** is a number you provide (weigh or look up), or `"derive"`,
  which means: current model body mass minus the `pla_g_solid` estimate of the
  listed parts (`--derive-column`). That is an estimate, and the report says so.
  It assumes the model's printed parts match the replica's, which may not hold
  (see below).
- A body with no `non_printed_g` is left exactly as it is.
- Errors stop the run: a part listed twice (in two bodies, or in a body and in
  `[unassigned]`), a part or body name that does not exist, parts with no
  `non_printed_g`, a weight of 0 or less, an unknown estimate column.

**Approximation:** for each changed body, the inertia tensor (`fullinertia` or
`diaginertia`) is multiplied by new mass / old mass, and the centre of mass
(`pos`) is kept. This assumes the mass distribution keeps the same shape and
only gets heavier or lighter. A part that is much denser in one place (such as a
glass-filled nylon bracket on one side of a body) breaks that assumption.

**Paths:** the copy sets `<compiler meshdir>` to the absolute path of the
upstream `assets/` folder, so it loads from `out/duck_mass/`. It is a
machine-specific file: regenerate it, don't copy it to another machine. Only the
robot file is generated; the upstream `scene_*.xml` files still include the
original `robot_walk.xml`.

## Model facts (read from `robot_walk.xml`)

- 15 bodies, each with an explicit `<inertial pos mass fullinertia>`; total
  737.243 g (MuJoCo 4.0.1 agrees). No `<include>`; `meshdir="assets"` (relative).
- `robot_walk_backlash.xml` and `robot_allcollisions*.xml` carry the same 15
  masses; the rollers variants have 19 inertials. Use `--model` for those.
- The template lists each body's meshes and how many `xl330` mesh instances it
  has. These come from the CAD export and are for orientation only.
- The printed parts come from a community replica (67 STLs with opaque names).
  They may not match the model's CAD parts one-to-one. One sign of that: the
  guessed sole + foot-housing for an ankle add up to 30.5 g (PLA solid), more
  than the model's whole 30.0 g ankle body, so `"derive"` fails there.

## Setup

`uv` was not installed and the repo `.venv` is Python 3.9 (no `tomllib`), so the
tool has its own git-ignored venv, made from Homebrew Python 3.13:

```
cd tools/duck_mass
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install mujoco pytest
```

The tool itself is standard library only (Python 3.11+); MuJoCo is used for the
load check when installed.

## Commands (all run from `tools/duck_mass/`)

```
.venv/bin/python duck_mass.py template
.venv/bin/python duck_mass.py apply --mapping mapping.toml --weights weights.csv
.venv/bin/python -m pytest -q
```

`template` writes `mapping.template.toml` (every body with its current mass,
every part under `[unassigned]`) and `weights.template.csv` (every part, blank
`measured_g`). The only proposed assignments (soles and foot-housings near the
ankle bodies) are comments marked GUESS; nothing is assigned.

`apply` writes to `out/duck_mass/` (change with `--out-dir`):
`robot_walk_measured.xml`, `report.txt`, `report.json`. It then loads the copy
in MuJoCo and prints the total mass MuJoCo computes.

## Workflow

1. `cp mapping.template.toml mapping.toml` and `cp weights.template.csv weights.csv`.
2. Weigh each printed part and fill `measured_g`.
3. In `mapping.toml`, move each part name from `[unassigned]` into its body's
   `parts`, and set `non_printed_g` for that body. Parts that are not on the
   robot (spare horns, the test piece) can stay in `[unassigned]`.
4. Run `apply` and read `report.txt`.

## Worked example (MADE-UP numbers)

`example/mapping.example.toml` and `example/weights.example.csv` hold invented
assignments and weights, only to show the format. They are not facts about the
robot.

```
.venv/bin/python duck_mass.py apply --mapping example/mapping.example.toml \
    --weights example/weights.example.csv --out-dir ../../out/duck_mass/example
```

Output (abridged):

```
body                   old g     new g   delta g  note
ankle_left            30.025    39.300    +9.275  all measured
ankle_right           30.025    36.230    +6.205  ESTIMATE for 1 of 2 parts
TOTAL                737.243   752.723   +15.480
bodies using estimates: ankle_right
MuJoCo 4.0.1 load: OK, 38 meshes, total mass 752.723 g
```

ankle_left = 18.4 + 6.9 (made-up weights) + 14.0 (made-up non-printed) = 39.3 g.
ankle_right = 18.1 (made-up weight) + 6.13 (PETG estimate, foot-housing not
weighed) + 12.0 = 36.23 g. Its inertia is scaled by 36.23 / 30.0251.

## Not verified

- No assignment of printed parts to bodies, and no servo or component mass.
  Those have to come from the owner.
- Whether the replica's parts correspond to the model's CAD parts.
- Training or simulation behaviour with the generated model; only loading and
  total mass were checked. mjlab loading the copy was not tried.
