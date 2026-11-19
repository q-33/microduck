# Microduck print files for Prusa printers

All 67 parts of the Microduck replica (source: AI-FanGe/Microduck-build-tutorial,
`microduck_print_plates.3mf`), arranged for two kinds of Prusa:

| Folder | Printer | Material | Plates |
|---|---|---|---|
| `prusa_xl_petg/` | Prusa XL 5-tool, 360 x 360 mm bed | PETG parts, PLA supports | 3 |
| `prusa_all_pla/` | Prusa MK3/MK4 size, 250 x 210 mm bed (fallback) | PLA (or PETG) | 7 |

Every part keeps the author's print orientation (checked against the original
Bambu project). The only changes are moves on the bed and, for some parts, a
90 degree turn about the vertical axis. Nothing was rotated off its print face.

**The 3MF files hold geometry and placement only.** They carry no printer,
filament, support or toolhead settings. Set those in the slicer.

## Folder contents

```
print/
  README.md            this file
  parts.csv            one row per part: size, volume, mass estimates, which plate
  prusa_xl_petg/       XL01..XL03 plates (3MF)
  prusa_all_pla/       01..07 plates (3MF)
  individual_STL/      all 67 parts as separate STLs, print orientation, fallback
  source_split/        the earlier split, unzipped from ../microduck_split.zip, untouched
  verification/
    source_report.json checks on the earlier split vs the original project
    build_summary.json plate list and mass totals from the build script
    layouts/*.png      top-down picture of every output plate
../tools/              the scripts (mdlib.py, verify_sources.py, build_plates.py, verify_outputs.py)
```

To rebuild or re-check (from the `microduck/` folder):

```
.venv/bin/python -I tools/verify_sources.py
.venv/bin/python -I tools/build_plates.py
.venv/bin/python -I tools/verify_outputs.py
```

## Prusa XL set (`prusa_xl_petg/`)

Usable area 340 x 340 mm (10 mm margin), at least 5 mm between part bounding
boxes. On every plate the **back-right corner, X 270-350 / Y 270-350 mm, is left
empty for the wipe tower**, and parts stay at least 5 mm away from it.

| Order | Plate | Parts | Footprint (X x Y, mm) | Tallest | PETG estimate |
|---|---|---|---|---|---|
| 1 | `XL01_test_piece.3mf` | 1 (servo bracket `frame_j`, bed centre) | 162.5-197.5 x 171.0-189.0 | 27.7 mm | 4 to 5 g |
| 2 | `XL02_structural.3mf` | 22 (frame parts, 2 leg links, 2 foot housings) | 10.0-343.4 x 10.0-274.0 | 39.5 mm | 110 to 140 g |
| 3 | `XL03_shells_bearings_horns.3mf` | 44 (shell, covers, beak, soles, ring, rings/horns/caps, 2 spare horns) | 10.0-348.3 x 10.0-245.9 | 116.1 mm | 147 to 165 g |

Print XL01 first and check it before printing XL02: does an XL330 servo fit,
do the M2 screws bite, are the holes the right size in PETG. The design was made
for PLA, and PETG shrinks differently. If it is undersized, scale all parts
the same amount in XY so the shells still line up.

### Toolheads: set these in PrusaSlicer

The 3MF files do not say which toolhead to use. After loading a plate:

1. Pick the **Original Prusa XL - 5T 0.4 nozzle** printer profile. It is in
   the PrusaSlicer 2.9.7 bundled profiles on this Mac.
2. Load PETG in toolhead 4 and PLA in toolhead 5, and pick matching filament
   profiles for those two extruders.
3. Assign **all parts to extruder 4** (PETG). PrusaSlicer has a per-object
   extruder choice in the object list. The exact menu wording was not checked
   here because the GUI was not opened.
4. In Print Settings, set **`support_material_extruder` = 5** and
   **`support_material_interface_extruder` = 5**, so supports print in PLA.
   Both setting names appear in the bundled `PrusaResearch.ini`.
   In that file, 0 means "use the current extruder".
5. Supports: the author printed with supports on (Bambu "tree(auto)", 30 degree
   threshold, not limited to the build plate). Turn supports on.
6. **Move the wipe tower into the reserved back-right corner.** The bundled
   profiles turn the wipe tower on (`wipe_tower = 1`). The XL print profile
   sets `wipe_tower_rotation_angle = 90` and `wipe_tower_brim_width = 3`. It
   does not override the position, so it inherits `wipe_tower_x = 170`,
   `wipe_tower_y = 140` and `wipe_tower_width = 60` from the common profile.
   That puts the tower in the middle of the bed, on top of parts. Drag the
   tower into the empty corner in the 3D view, or change `wipe_tower_x` /
   `wipe_tower_y`. Then check in the sliced preview that the tower, with its
   brim, stays inside the 80 x 80 mm corner. Its depth depends on purge
   volumes, so it was not measured here.
7. Toolhead numbering: PrusaSlicer numbers extruders from 1 in the GUI. The XL
   G-code numbers tools from 0 (T0..T4). Before printing, confirm that
   "toolhead 4" and "toolhead 5" mean the same tools on the printer and in the
   slicer.

Nothing in this folder was sliced, so none of these settings were tested.

## 250 x 210 mm set (`prusa_all_pla/`, fallback)

Usable area 240 x 200 mm (5 mm margin), at least 4 mm between part bounding boxes.
All seven plates of the earlier split already fit, so each is a **byte-for-byte
copy** of a source plate under a new name. The parts on them were not moved.
The NylonG plates had been laid out for a 250 x 220 mm bed, but every part
was already inside 240 x 200 mm.

| Order | Plate | Copy of | Parts | X range | Y range | Tallest | PLA estimate |
|---|---|---|---|---|---|---|---|
| 1 | `01_test_piece_N0.3mf` | N0_test_piece | 1 | 107.0-142.0 | 96.0-114.0 | 27.7 | 4 to 5 g |
| 2 | `02_frame_N1a.3mf` | N1_frame_1 | 6 | 38.6-211.4 | 41.8-178.2 | 18.8 | 31 to 33 g |
| 3 | `03_frame_N1b.3mf` | N1_frame_2 | 12 | 26.7-223.3 | 45.4-174.6 | 34.5 | 41 to 61 g |
| 4 | `04_legs_feet_N2.3mf` | N2_legs_feet_1 | 4 | 34.4-215.6 | 69.6-150.4 | 39.5 | 36 to 43 g |
| 5 | `05_bearings_horns_P3.3mf` | P3_bearings_horns_1 | 36 | 10.0-240.0 | 49.0-161.0 | 4.6 | 21 to 22 g |
| 6 | `06_shells_beak_P1.3mf` | P1_shells_beak_1 | 5 | 12.1-237.9 | 12.5-197.5 | 116.1 | 85 to 89 g |
| 7 | `07_soles_ring_P2.3mf` | P2_soles_ring_1 | 3 | 50.0-200.0 | 84.4-125.6 | 18.4 | 37 to 51 g |

Plate 5 has the smallest spacing: exactly 4.0 mm between part bounding boxes.
Plates 2, 4 and 7 are sparse. If you want fewer plates, they could be combined,
but that was left alone to avoid changing layouts that already pass.

## Author's slicer settings

These were read from `Metadata/project_settings.config` in the original Bambu
project: Bambu Lab P1S printer profile, "0.20mm Standard @BBL X1C" process,
PLA filament.

- 0.4 mm nozzle, 0.42 mm line width, 0.2 mm layers, 0.2 mm first layer
- 2 walls (`wall_loops` 2), 5 top layers, 3 bottom layers
- 15 % sparse infill, grid pattern
- supports on: tree(auto), 30 degree threshold, not limited to the build plate
- auto brim, 5 mm brim width

The gaps between parts (4 or 5 mm) are smaller than a 5 mm brim on each side.
If you add brims, check in the slicer that neighbouring brims don't merge.

## Mass estimates (`parts.csv`)

- **Solid**: the mesh volume times density (PLA 1.24 g/cm3, PETG 1.27 g/cm3).
  This is the upper bound for the part itself.
- **Rough estimate at the author's settings**: shell volume plus 15 % of
  the rest. Shell volume is each triangle's area times the shell thickness
  normal to it: 0.84 mm of walls (2 x 0.42 mm) on vertical faces, 1.0 mm of
  top skin (5 x 0.2 mm) and 0.6 mm of bottom skin (3 x 0.2 mm) on
  horizontal faces, blended for sloped faces, and capped at the solid volume.
  This is a rough approximation that no slicer has checked. It leaves out
  supports, brim, wipe tower and purge, so real filament use will be higher.

Totals for all 67 parts:

| | Solid | Rough at author settings |
|---|---|---|
| PLA | 303.5 g | 254.3 g |
| PETG | 310.8 g | 260.5 g |

Total part volume is 244.8 cm3. PrusaSlicer's `--info` volumes add up to the
same figure.

## Candidates for a later nylon reprint (shape reading only)

This list comes from looking at the shapes. It is **not engineering analysis**:
no loads, stresses or failure modes were calculated. These parts look like
they carry servo torque or the robot's weight:

- `plate3_frame_j_id28`, the test piece: U-shaped bracket with a servo-horn
  bolt pattern on both side walls.
- `plate3_frame_b_id14` and `_id60`: L-brackets, horn disc on one face and a
  servo-face mount on the other.
- `plate3_frame_dfbdfbbbbb1_id70`: bracket with two horn interfaces, likely at
  the hip.
- `plate3_frame_n_id36` and `_o_id38`: servo cradles with mounting bosses.
- `plate2_leg-link_7_id8` and `_8_id10`: long open shells with a horn disc at
  one end (leg links / servo housings).
- `plate1_foot-housing_grbrgn_id64` and `_thngthm_id66`: cylindrical foot or
  ankle housings with a horn opening and screw ears.
- `plate3_frame_k_id30` and `_l_id32`: link plates with horn-sized holes. These
  are less certain.

The rest look lightly loaded: thin plates, board mounts, shells, covers, soles,
and the rings, horns and caps.

## What was verified and what was not

Verified by script (`tools/verify_sources.py`, `tools/verify_outputs.py`):

- The original Bambu project parses. It has 67 objects: 65 on 5 plates
  (4/5/19/34/3) and 2 off-plate (the spare idle horns).
- Every earlier-split 3MF and STL parses. The plates hold 23 + 44 = 67 parts,
  each original object exactly once, with no missing or extra parts. Every
  mesh is watertight, a single body, with positive volume, and sits on the bed
  (min Z = 0). Each part has the same print orientation and volume as in the
  original. The individual STLs match the plate parts.
- The top-level `../NylonG_PulseXE/` files are byte-identical to the copies in
  the zip.
- Identical copies are intentional repeats in the original: 18 rings, 8 idle
  horns, 6 idle caps, and pairs of frame_b, frame_d/e, frame_n/o, the
  23 x 40 mm plates and the 14.6 mm rings.
- Output (re-parsed after writing): both sets have 67 parts, each original
  part exactly once. Every part is inside the usable area and below the height
  limit. No two bounding boxes are closer than 4 mm (small set) or 5 mm (XL).
  On the XL plates, nothing is within 5 mm of the tower corner. Meshes and
  orientations match the original. Reused plates are byte-identical to their
  source. `individual_STL/` has 67 files that match. `parts.csv` agrees with
  the plate files.
- PrusaSlicer 2.9.7 (`/Applications/Original Prusa Drivers/PrusaSlicer.app`)
  command line, `--info`: all 10 output plates load, with the expected object
  count and every object reported manifold. A `--dont-arrange --export-3mf`
  round trip of two plates, written to a temporary folder, kept every part's
  position to within 0.01 mm.
- The original zip, the original 3MF and `../NylonG_PulseXE/` were unchanged
  (SHA-256 checked before and after).

Not verified:

- No plate was sliced. Support amount, print time and real filament use are unknown.
- The PrusaSlicer GUI was not opened. It was not checked whether the GUI
  re-arranges parts when it loads them, or what the menu items are called.
- The wipe tower's real footprint, and whether it fits the 80 x 80 mm corner.
- Fit, strength or shrink of any part in PETG or PLA.
- The part labels (frame, leg-link, cover...) and the nylon-candidate list
  come from reading the geometry. They are not the author's names.
- The handoff note says the largest part is "about 123 x 92 x 46 mm". In print
  orientation no part has that size. The largest footprint is the beak tray,
  91.8 x 117.1 mm, and the tallest part is the big shell at 116.1 mm.
- Small rounding differences: the split stored vertices to 4 decimals. Volumes
  differ from the original by at most 0.05 mm3 per part.
