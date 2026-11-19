"""Build the two Prusa print sets from the verified source split.

Usage: .venv/bin/python -I tools/build_plates.py

  print/prusa_all_pla/   250 x 210 bed (MK3/MK4), 240 x 200 usable, >= 4 mm gaps.
                         Source plates that already fit are copied byte-for-byte.
  print/prusa_xl_petg/   360 x 360 bed (XL), 340 x 340 usable, >= 5 mm gaps,
                         80 x 80 mm wipe-tower corner kept clear (back-right).
  print/individual_STL/  all 67 source STLs, flat, byte-for-byte copies.
  print/parts.csv        per-part table (sizes, volume, mass estimates, plates).

Parts are never rotated off their print face: the only changes are XY
translation and rotation about Z (0 or 90 degrees).
"""
import csv
import glob
import json
import math
import os
import re
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdlib import read_3mf, write_3mf  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRINT = os.path.join(ROOT, "print")
SPLIT = os.path.join(PRINT, "source_split", "microduck_split")
OUT_MK = os.path.join(PRINT, "prusa_all_pla")
OUT_XL = os.path.join(PRINT, "prusa_xl_petg")
OUT_STL = os.path.join(PRINT, "individual_STL")

# --- small-Prusa set -------------------------------------------------------
MK_USABLE = (5.0, 5.0, 245.0, 205.0)
MK_MAX_Z = 210.0
MK_GAP = 4.0
# print order for the reused source plates
MK_PLAN = [
    ("01_test_piece_N0", "NylonG_PulseXE/N0_test_piece.3mf"),
    ("02_frame_N1a", "NylonG_PulseXE/N1_frame_1.3mf"),
    ("03_frame_N1b", "NylonG_PulseXE/N1_frame_2.3mf"),
    ("04_legs_feet_N2", "NylonG_PulseXE/N2_legs_feet_1.3mf"),
    ("05_bearings_horns_P3", "PLA_Prusa/P3_bearings_horns_1.3mf"),
    ("06_shells_beak_P1", "PLA_Prusa/P1_shells_beak_1.3mf"),
    ("07_soles_ring_P2", "PLA_Prusa/P2_soles_ring_1.3mf"),
]

# --- XL set ----------------------------------------------------------------
XL_USABLE = (10.0, 10.0, 350.0, 350.0)
XL_MAX_Z = 360.0
XL_GAP = 5.0
XL_TOWER = (270.0, 270.0, 350.0, 350.0)   # back-right corner of the usable area

# --- mass-estimate parameters (author's Bambu project settings) ------------
DENS = {"PLA": 1.24, "PETG": 1.27}
WALL_T = 2 * 0.42      # wall_loops 2 x line_width 0.42
TOP_T = 5 * 0.2        # top_shell_layers 5 x layer_height 0.2
BOT_T = 3 * 0.2        # bottom_shell_layers 3 x layer_height 0.2
INFILL = 0.15          # sparse_infill_density 15 %


def fits(box, usable, max_z, zmax):
    return (box[0] >= usable[0] - 1e-6 and box[1] >= usable[1] - 1e-6 and
            box[2] <= usable[2] + 1e-6 and box[3] <= usable[3] + 1e-6 and zmax <= max_z)


def bbox_gap(a, b):
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def rough_shell_volume(mesh):
    """Approximate printed shell volume: each face contributes area x the
    thickness of the shell normal to it (walls horizontally, top/bottom skins
    vertically). Capped at the solid volume."""
    nz = mesh.face_normals[:, 2]
    a = mesh.area_faces
    side = WALL_T * np.sqrt(np.clip(1 - nz ** 2, 0, 1))
    skin = np.where(nz > 0, TOP_T * nz, BOT_T * -nz)
    return float(min(mesh.volume, np.sum(a * np.maximum(side, skin))))


# ---------------------------------------------------------------------------
# MaxRects packer (best-short-side-fit) with a reserved obstacle and gaps.
# Every part rectangle is inflated by gap/2 on each side; the container is
# grown by gap/2 so the real part edge stays inside the usable area, and the
# obstacle is grown by gap/2 so parts keep >= gap from it.
# ---------------------------------------------------------------------------
class Bin:
    def __init__(self, usable, gap, obstacle=None):
        h = gap / 2
        x0, y0, x1, y1 = usable[0] - h, usable[1] - h, usable[2] + h, usable[3] + h
        self.free = [(x0, y0, x1, y1)]
        if obstacle:
            self._cut((obstacle[0] - h, obstacle[1] - h, obstacle[2] + h, obstacle[3] + h))
        self.placed = []

    def _cut(self, r):
        new = []
        for f in self.free:
            if r[0] >= f[2] or r[2] <= f[0] or r[1] >= f[3] or r[3] <= f[1]:
                new.append(f)
                continue
            if r[0] > f[0]: new.append((f[0], f[1], r[0], f[3]))
            if r[2] < f[2]: new.append((r[2], f[1], f[2], f[3]))
            if r[1] > f[1]: new.append((f[0], f[1], f[2], r[1]))
            if r[3] < f[3]: new.append((f[0], r[3], f[2], f[3]))
        # prune contained rectangles
        new = [a for i, a in enumerate(new) if a[2] - a[0] > 1e-6 and a[3] - a[1] > 1e-6]
        pruned = []
        for i, a in enumerate(new):
            if any(j != i and b[0] <= a[0] and b[1] <= a[1] and b[2] >= a[2] and b[3] >= a[3]
                   and (b != a or j < i) for j, b in enumerate(new)):
                continue
            pruned.append(a)
        self.free = pruned

    def find(self, w, h):
        best = None
        for f in self.free:
            fw, fh = f[2] - f[0], f[3] - f[1]
            for rot, (pw, ph) in ((0, (w, h)), (90, (h, w))):
                if pw <= fw + 1e-9 and ph <= fh + 1e-9:
                    score = (min(fw - pw, fh - ph), max(fw - pw, fh - ph))
                    if best is None or score < best[0]:
                        best = (score, f[0], f[1], pw, ph, rot)
        return best

    def place(self, x, y, w, h):
        self._cut((x, y, x + w, y + h))
        self.placed.append((x, y, x + w, y + h))


def rz(deg):
    t = math.radians(deg)
    m = np.eye(4)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = math.cos(t), -math.sin(t), math.sin(t), math.cos(t)
    return m


def tr(x, y, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def relocate(part, rot, x0, y0):
    """New build transform: rotate the placed part about Z then move its
    bbox min corner to (x0, y0). The object mesh itself is unchanged."""
    b = part.mesh.bounds
    c = (b[0] + b[1]) / 2
    d = rz(rot) @ tr(-c[0], -c[1])
    moved = part.mesh.copy()
    moved.apply_transform(d)
    nb = moved.bounds
    d = tr(x0 - nb[0][0], y0 - nb[0][1]) @ d
    return d @ part.transform


def main():
    for d in (OUT_MK, OUT_XL, OUT_STL):
        os.makedirs(d, exist_ok=True)

    # ---- load source parts -------------------------------------------------
    src = {}       # plate rel path -> [Part]
    parts = {}     # name -> dict
    for rel in sorted({p for _n, p in MK_PLAN}):
        ps = read_3mf(os.path.join(SPLIT, rel))
        src[rel] = ps
        for p in ps:
            m = re.match(r"plate(\w+?)_([a-z\-]+)_(.+)_id(\d+)$", p.name)
            parts[p.name] = {"part": p, "source_plate": rel, "orig_plate": m.group(1),
                             "label": m.group(2), "orig_name": m.group(3), "id": int(m.group(4))}
    assert len(parts) == 67, len(parts)

    # ---- small-Prusa set: reuse plates that pass, re-lay the rest ------------
    mk_plate_of = {}
    mk_summary = []
    for name, rel in MK_PLAN:
        ps = src[rel]
        boxes = [(p.mesh.bounds[0][0], p.mesh.bounds[0][1], p.mesh.bounds[1][0], p.mesh.bounds[1][1]) for p in ps]
        ok = all(fits(b, MK_USABLE, MK_MAX_Z, p.mesh.bounds[1][2]) for b, p in zip(boxes, ps))
        gaps = [bbox_gap(a, b) for i, a in enumerate(boxes) for b in boxes[i + 1:]]
        ok = ok and all(g >= MK_GAP - 1e-6 for g in gaps)
        if not ok:
            # Not reached with the current source files (all seven pass); kept
            # explicit so a future source change cannot silently slip through.
            raise SystemExit(f"{rel} does not fit 240x200 with {MK_GAP} mm gaps: re-layout needed")
        shutil.copyfile(os.path.join(SPLIT, rel), os.path.join(OUT_MK, name + ".3mf"))
        for p in ps:
            mk_plate_of[p.name] = name
        mk_summary.append({"plate": name, "source": rel, "parts": len(ps), "reused_unchanged": True})

    # ---- XL set --------------------------------------------------------------
    xl_plates = []
    test = parts["plate3_frame_j_id28"]["part"]
    b = test.mesh.bounds
    w, h = b[1][0] - b[0][0], b[1][1] - b[0][1]
    t = relocate(test, 0, 180 - w / 2, 180 - h / 2)   # centre of the bed
    xl_plates.append([("plate3_frame_j_id28", t)])

    # Remaining parts, grouped so each plate has a purpose where possible:
    # structural first (fit-check before the cosmetic shells), then the rest.
    def size(n):
        bb = parts[n]["part"].mesh.bounds
        return (bb[1][0] - bb[0][0]) * (bb[1][1] - bb[0][1])
    rest = [n for n in parts if n != "plate3_frame_j_id28"]
    structural = sorted([n for n in rest if parts[n]["source_plate"].startswith("NylonG")], key=size, reverse=True)
    other = sorted([n for n in rest if not parts[n]["source_plate"].startswith("NylonG")], key=size, reverse=True)
    bins = []
    for group in (structural, other):
        group_bins = []
        for n in group:
            p = parts[n]["part"]
            bb = p.mesh.bounds
            w, h = bb[1][0] - bb[0][0] + XL_GAP, bb[1][1] - bb[0][1] + XL_GAP
            # first fit over this group's own plates only, so a plate holds
            # one kind of part (structural vs shells/bearings/horns)
            placed = False
            for bn in group_bins:
                f = bn.find(w, h)
                if f:
                    _s, x, y, pw, ph, rot = f
                    bn.place(x, y, pw, ph)
                    bn.items.append((n, relocate(p, rot, x + XL_GAP / 2, y + XL_GAP / 2)))
                    placed = True
                    break
            if not placed:
                bn = Bin(XL_USABLE, XL_GAP, XL_TOWER)
                bn.items = []
                f = bn.find(w, h)
                if not f:
                    raise SystemExit(f"{n} does not fit an empty XL plate")
                _s, x, y, pw, ph, rot = f
                bn.place(x, y, pw, ph)
                bn.items.append((n, relocate(p, rot, x + XL_GAP / 2, y + XL_GAP / 2)))
                bins.append(bn)
                group_bins.append(bn)
    for bn in bins:
        xl_plates.append(bn.items)

    def xl_name(i, items):
        if i == 0:
            return "XL01_test_piece"
        kinds = {parts[n]["source_plate"].startswith("NylonG") for n, _t in items}
        tag = "structural" if kinds == {True} else ("shells_bearings_horns" if kinds == {False} else "mixed")
        return f"XL{i + 1:02d}_{tag}"

    xl_plate_of = {}
    xl_summary = []
    for i, items in enumerate(xl_plates):
        name = xl_name(i, items)
        write_3mf(os.path.join(OUT_XL, name + ".3mf"),
                  [(n, parts[n]["part"].local_mesh, t) for n, t in items])
        for n, _t in items:
            xl_plate_of[n] = name
        xl_summary.append({"plate": name, "parts": len(items)})

    # ---- individual STLs -----------------------------------------------------
    for f in glob.glob(os.path.join(SPLIT, "*", "individual_STL", "*.stl")):
        shutil.copyfile(f, os.path.join(OUT_STL, os.path.basename(f)))

    # ---- parts.csv -------------------------------------------------------------
    # The "label" column is a guess made from the source plates' object names, and
    # some guesses are wrong. Where a part was matched to an official MicroDuck mesh
    # (2026-11-18 fit study: aligned surface distance <= 0.07 mm), say what it is.
    # File names keep the old label: plates and the weight tool use them as keys.
    identified_as = {
        "plate2_leg-link_7_id8": "left_shell (left body shell; NOT a leg link)",
        "plate2_leg-link_8_id10": "right_shell (right body shell; NOT a leg link)",
        "plate3_frame_m_id34": "trunk_base",
        "plate3_frame_a_id12": "power_support (battery holder)",
        "plate3_frame_i_id26": "banana_pcb_locker",
        "plate3_frame_q_id42": "jaw_soft",
        "plate2_big-shell_5_id6": "top_head_shell",
        "plate1_beak-tray_3_id2": "bottom_head_shell",
        "plate1_beak-jaw_4_id4": "jaw",
        "plate2_cover_fgb_id72": "upper leg (NOT a cover; left/right not determined)",
        "plate2_cover_gbgb_id74": "upper leg (NOT a cover; left/right not determined)",
    }
    rows = []
    totals = {k: 0.0 for k in ("pla_solid", "pla_rough", "petg_solid", "petg_rough")}
    for n in sorted(parts, key=lambda n: (mk_plate_of[n], n)):
        p = parts[n]["part"].mesh
        vol = p.volume / 1000.0                      # cm3
        shell = rough_shell_volume(p) / 1000.0
        rough = shell + INFILL * (vol - shell)
        sz = p.bounds[1] - p.bounds[0]
        r = {
            "part_file": n + ".stl",
            "original_plate": parts[n]["orig_plate"],
            "label": parts[n]["label"],
            "identified_as": identified_as.get(n, ""),
            "original_name": parts[n]["orig_name"],
            "earlier_split_plate": parts[n]["source_plate"],
            "bbox_x_mm": round(sz[0], 2), "bbox_y_mm": round(sz[1], 2), "bbox_z_mm": round(sz[2], 2),
            "volume_cm3": round(vol, 3),
            "pla_g_solid": round(vol * DENS["PLA"], 2),
            "pla_g_rough_author_settings": round(rough * DENS["PLA"], 2),
            "petg_g_solid": round(vol * DENS["PETG"], 2),
            "petg_g_rough_author_settings": round(rough * DENS["PETG"], 2),
            "mk_plate_250x210": mk_plate_of[n],
            "xl_plate_360x360": xl_plate_of[n],
        }
        totals["pla_solid"] += r["pla_g_solid"]; totals["pla_rough"] += rough * DENS["PLA"]
        totals["petg_solid"] += r["petg_g_solid"]; totals["petg_rough"] += rough * DENS["PETG"]
        rows.append(r)
    with open(os.path.join(PRINT, "parts.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = {"mk": mk_summary, "xl": xl_summary, "totals_g": {k: round(v, 1) for k, v in totals.items()}}
    with open(os.path.join(PRINT, "verification", "build_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
