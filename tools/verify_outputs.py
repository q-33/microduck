"""Independently re-check everything build_plates.py wrote.

Usage: .venv/bin/python -I tools/verify_outputs.py
Re-parses every output 3MF and STL, compares each part with the ORIGINAL
Bambu project (same mesh, same print orientation), checks counts, bounds,
gaps, the XL tower corner, parts.csv consistency, and draws top-down layout
images to print/verification/layouts/.
"""
import csv
import filecmp
import glob
import os
import re
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdlib import read_3mf, geom_key, z_profile_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRINT = os.path.join(ROOT, "print")
ORIG = os.path.join(ROOT, "microduck_print_plates.3mf")
SPLIT = os.path.join(PRINT, "source_split", "microduck_split")
SETS = {
    "prusa_all_pla": {"bed": (250, 210), "usable": (5, 5, 245, 205), "max_z": 210, "gap": 4.0, "tower": None},
    "prusa_xl_petg": {"bed": (360, 360), "usable": (10, 10, 350, 350), "max_z": 360, "gap": 5.0,
                      "tower": (270, 270, 350, 350)},
}

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)
    return cond


def gap(a, b):
    return max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])


def main():
    orig = {}
    for p in read_3mf(ORIG):
        orig[int(p.object_id)] = p
    orig_keys = {k: (geom_key(p.local_mesh), z_profile_key(p.mesh), p.mesh.volume) for k, p in orig.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    os.makedirs(os.path.join(PRINT, "verification", "layouts"), exist_ok=True)

    plate_of = {}
    for set_name, cfg in SETS.items():
        files = sorted(glob.glob(os.path.join(PRINT, set_name, "*.3mf")))
        ids_seen = []
        print(f"== {set_name}: {len(files)} plates")
        for f in files:
            pname = os.path.splitext(os.path.basename(f))[0]
            parts = read_3mf(f)
            boxes = []
            fig, ax = plt.subplots(figsize=(6, 6 * cfg["bed"][1] / cfg["bed"][0]))
            ax.add_patch(Rectangle((0, 0), *cfg["bed"], fill=False, lw=1.5))
            u = cfg["usable"]
            ax.add_patch(Rectangle((u[0], u[1]), u[2] - u[0], u[3] - u[1], fill=False, ls="--", lw=0.8, ec="grey"))
            if cfg["tower"]:
                t = cfg["tower"]
                ax.add_patch(Rectangle((t[0], t[1]), t[2] - t[0], t[3] - t[1], fc="#f6c", alpha=0.3))
                ax.text((t[0] + t[2]) / 2, (t[1] + t[3]) / 2, "wipe\ntower", ha="center", va="center", fontsize=7)
            for p in parts:
                oid = int(re.search(r"_id(\d+)$", p.name).group(1))
                ids_seen.append(oid)
                plate_of.setdefault(set_name, {})[p.name] = pname
                b = p.mesh.bounds
                box = (b[0][0], b[0][1], b[1][0], b[1][1])
                boxes.append((p.name, box))
                check(p.mesh.is_watertight, f"{pname}:{p.name} not watertight")
                check(p.mesh.volume > 0, f"{pname}:{p.name} non-positive volume")
                check(abs(b[0][2]) < 0.05, f"{pname}:{p.name} not on the bed (min z {b[0][2]:.3f})")
                check(b[1][2] <= cfg["max_z"], f"{pname}:{p.name} too tall")
                check(box[0] >= u[0] - 1e-6 and box[1] >= u[1] - 1e-6 and box[2] <= u[2] + 1e-6 and box[3] <= u[3] + 1e-6,
                      f"{pname}:{p.name} outside usable area {box}")
                if cfg["tower"]:
                    check(gap(box, cfg["tower"]) >= cfg["gap"] - 1e-6, f"{pname}:{p.name} too close to tower corner")
                # same mesh and same print orientation as the original Bambu object
                gk, zp, vol = orig_keys[oid]
                # split files store vertices to 4 decimals, so allow max(0.05 mm3, 1e-5 relative) volume difference
                check(geom_key(p.local_mesh) == gk or abs(p.mesh.volume - vol) <= max(0.05, 1e-5 * vol),
                      f"{pname}:{p.name} mesh differs from original id {oid}")
                zs = z_profile_key(p.mesh)
                check(len(zs) == len(zp) and np.allclose(zs, zp, atol=0.05),
                      f"{pname}:{p.name} print orientation differs from original id {oid}")
                ax.add_patch(Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fc="#9cf", ec="k", lw=0.5))
                v = p.mesh.vertices
                ax.plot(*np.array([v[:, 0], v[:, 1]])[:, ::max(1, len(v) // 1500)], ",", color="#036", ms=0.5)
                ax.text((box[0] + box[2]) / 2, (box[1] + box[3]) / 2, str(oid), fontsize=5, ha="center", va="center")
            mg = min((gap(a[1], b[1]) for i, a in enumerate(boxes) for b in boxes[i + 1:]), default=float("inf"))
            check(mg >= cfg["gap"] - 1e-6, f"{pname}: min bbox gap {mg:.2f} < {cfg['gap']}")
            ext = np.array([bx for _n, bx in boxes])
            zmax = max(p.mesh.bounds[1][2] for p in parts)
            print(f"  {pname:30s} parts={len(parts):2d}  X {ext[:,0].min():6.1f}-{ext[:,2].max():6.1f}  "
                  f"Y {ext[:,1].min():6.1f}-{ext[:,3].max():6.1f}  Zmax {zmax:6.1f}  min gap {mg if mg != float('inf') else '-'}")
            ax.set_xlim(-5, cfg["bed"][0] + 5); ax.set_ylim(-5, cfg["bed"][1] + 5); ax.set_aspect("equal")
            ax.set_title(f"{set_name}/{pname}  ({len(parts)} parts)", fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(PRINT, "verification", "layouts", f"{set_name}__{pname}.png"), dpi=110)
            plt.close(fig)
        check(len(ids_seen) == 67, f"{set_name}: {len(ids_seen)} parts, expected 67")
        check(len(set(ids_seen)) == len(ids_seen), f"{set_name}: duplicated parts")
        check(set(ids_seen) == set(orig), f"{set_name}: missing {sorted(set(orig) - set(ids_seen))}")
        print(f"  total parts {len(ids_seen)}, unique {len(set(ids_seen))}, match original ids: {set(ids_seen) == set(orig)}")

    # reused small-Prusa plates are byte-identical to the source split
    for f in sorted(glob.glob(os.path.join(PRINT, "prusa_all_pla", "*.3mf"))):
        tag = os.path.splitext(os.path.basename(f))[0].rsplit("_", 1)[1]
        src = {"N0": "NylonG_PulseXE/N0_test_piece.3mf", "N1a": "NylonG_PulseXE/N1_frame_1.3mf",
               "N1b": "NylonG_PulseXE/N1_frame_2.3mf", "N2": "NylonG_PulseXE/N2_legs_feet_1.3mf",
               "P1": "PLA_Prusa/P1_shells_beak_1.3mf", "P2": "PLA_Prusa/P2_soles_ring_1.3mf",
               "P3": "PLA_Prusa/P3_bearings_horns_1.3mf"}[tag]
        check(filecmp.cmp(f, os.path.join(SPLIT, src), shallow=False), f"{f} differs from {src}")

    # individual STLs
    stls = sorted(glob.glob(os.path.join(PRINT, "individual_STL", "*.stl")))
    check(len(stls) == 67, f"{len(stls)} STLs")
    for f in stls:
        m = trimesh.load(f, force="mesh", process=False)
        m.merge_vertices()
        oid = int(re.search(r"_id(\d+)\.stl$", f).group(1))
        check(m.is_watertight, f"{f} not watertight")
        check(abs(m.volume - orig_keys[oid][2]) < 0.5, f"{f} volume differs from original")
        check(np.allclose(z_profile_key(m), orig_keys[oid][1], atol=0.05) if len(z_profile_key(m)) == len(orig_keys[oid][1]) else False,
              f"{f} orientation differs from original")
    print(f"individual_STL: {len(stls)} files checked")

    # parts.csv consistent with the files
    rows = list(csv.DictReader(open(os.path.join(PRINT, "parts.csv"))))
    check(len(rows) == 67 and len({r["part_file"] for r in rows}) == 67, "parts.csv row count")
    for r in rows:
        n = r["part_file"][:-4]
        check(plate_of["prusa_all_pla"].get(n) == r["mk_plate_250x210"], f"csv mk plate wrong for {n}")
        check(plate_of["prusa_xl_petg"].get(n) == r["xl_plate_360x360"], f"csv xl plate wrong for {n}")
        oid = int(re.search(r"_id(\d+)$", n).group(1))
        check(abs(float(r["volume_cm3"]) - orig_keys[oid][2] / 1000) < 0.002, f"csv volume wrong for {n}")
    print("parts.csv: 67 rows checked")

    # originals untouched (hashes recorded before work started are compared by the caller)
    print(f"\nFAILURES: {len(fail)}")
    for x in fail:
        print("  -", x)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
