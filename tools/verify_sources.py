"""Verify the earlier print split against the original Bambu project.

Usage: .venv/bin/python -I tools/verify_sources.py
Writes print/verification/source_report.json and prints a summary.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdlib import read_3mf, mesh_stats, geom_key, z_profile_key  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT = os.path.join(ROOT, "print", "source_split", "microduck_split")
ORIG = os.path.join(ROOT, "microduck_print_plates.3mf")
OUT = os.path.join(ROOT, "print", "verification")
USABLE = (5.0, 5.0, 245.0, 205.0)   # 250 x 210 bed less 5 mm margin
MAX_Z = 210.0


def bbox_gap(a, b):
    """Smallest axis gap between two XY boxes (negative = overlap)."""
    dx = max(b[0] - a[2], a[0] - b[2])
    dy = max(b[1] - a[3], a[1] - b[3])
    return max(dx, dy)


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"original": {}, "plates": {}, "stls": {}, "issues": []}
    issues = report["issues"]

    # ---- original Bambu project -------------------------------------------
    orig_parts = read_3mf(ORIG)
    import zipfile
    import defusedxml.ElementTree as ET
    ms = ET.fromstring(zipfile.ZipFile(ORIG).read("Metadata/model_settings.config"))
    plate_of = {}
    for pl in ms.findall("plate"):
        pid = pl.find("metadata[@key='plater_id']").get("value")
        for mi in pl.findall("model_instance"):
            plate_of[mi.find("metadata[@key='object_id']").get("value")] = pid
    name_of = {o.get("id"): o.find("metadata[@key='name']").get("value") for o in ms.findall("object")}
    orig = {}
    for p in orig_parts:
        st = mesh_stats(p.mesh)
        orig[p.object_id] = {"name": name_of.get(p.object_id, p.name),
                             "plate": plate_of.get(p.object_id, "none"),
                             "stats": st, "zprof": z_profile_key(p.mesh),
                             "geom": geom_key(p.local_mesh)}
    report["original"] = {
        "parts": len(orig_parts),
        "on_plates": sum(1 for v in orig.values() if v["plate"] != "none"),
        "per_plate": {k: sum(1 for v in orig.values() if v["plate"] == k)
                      for k in sorted({v["plate"] for v in orig.values()})},
        "objects": {k: {"name": v["name"], "plate": v["plate"], **{x: v["stats"][x] for x in
                    ("size", "watertight", "volume_mm3", "bodies")}} for k, v in orig.items()},
    }

    # ---- split plates -----------------------------------------------------
    seen_ids = {}
    split_parts = {}
    for f in sorted(glob.glob(os.path.join(SPLIT, "*", "*.3mf"))):
        rel = os.path.relpath(f, SPLIT)
        try:
            parts = read_3mf(f)
        except Exception as e:  # noqa: BLE001
            report["plates"][rel] = {"parse_error": repr(e)}
            issues.append(f"{rel}: parse error {e!r}")
            continue
        boxes, entries = [], []
        for p in parts:
            st = mesh_stats(p.mesh)
            m = re.search(r"_id(\d+)$", p.name)
            oid = m.group(1) if m else None
            seen_ids.setdefault(oid, []).append(rel)
            split_parts[p.name] = (rel, p)
            box = (st["min"][0], st["min"][1], st["max"][0], st["max"][1])
            boxes.append((p.name, box))
            e = {"name": p.name, **st}
            # same print orientation as the original? compare z profiles
            if oid in orig:
                zo, zs = orig[oid]["zprof"], z_profile_key(p.mesh)
                e["same_orientation_as_original"] = bool(len(zo) == len(zs) and np.allclose(zo, zs, atol=0.05))
                e["volume_matches_original"] = bool(abs(st["volume_mm3"] - orig[oid]["stats"]["volume_mm3"]) < 0.5)
                e["original_plate"] = orig[oid]["plate"]
                if not e["same_orientation_as_original"]:
                    issues.append(f"{rel}:{p.name} orientation differs from original")
            else:
                issues.append(f"{rel}:{p.name} has no matching original object id")
            if not st["watertight"]:
                issues.append(f"{rel}:{p.name} not watertight")
            if st["volume_mm3"] <= 1e-6:
                issues.append(f"{rel}:{p.name} zero/negative volume")
            if abs(st["min"][2]) > 0.05:
                issues.append(f"{rel}:{p.name} min z = {st['min'][2]} (not on bed)")
            entries.append(e)
        allmin = np.min([e["min"] for e in entries], axis=0)
        allmax = np.max([e["max"] for e in entries], axis=0)
        fits = bool(allmin[0] >= USABLE[0] - 1e-6 and allmin[1] >= USABLE[1] - 1e-6 and
                    allmax[0] <= USABLE[2] + 1e-6 and allmax[1] <= USABLE[3] + 1e-6 and allmax[2] <= MAX_Z)
        min_gap = min((bbox_gap(a[1], b[1]) for i, a in enumerate(boxes) for b in boxes[i + 1:]), default=None)
        report["plates"][rel] = {
            "parts": len(parts), "extent_min": allmin.round(2).tolist(), "extent_max": allmax.round(2).tolist(),
            "fits_240x200_usable": fits, "min_bbox_gap_mm": None if min_gap is None else round(min_gap, 2),
            "entries": entries,
        }

    # ---- individual STLs --------------------------------------------------
    for f in sorted(glob.glob(os.path.join(SPLIT, "*", "individual_STL", "*.stl"))):
        rel = os.path.relpath(f, SPLIT)
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            m = trimesh.load(f, force="mesh", process=False)
            m.merge_vertices()
        except Exception as e:  # noqa: BLE001
            report["stls"][rel] = {"parse_error": repr(e)}
            issues.append(f"{rel}: parse error")
            continue
        st = mesh_stats(m)
        e = {**st}
        if name in split_parts:
            prel, p = split_parts[name]
            e["in_plate"] = prel
            e["same_as_plate_part"] = bool(geom_key(m) == geom_key(p.mesh) or (
                np.allclose(z_profile_key(m), z_profile_key(p.mesh), atol=0.05) and abs(m.volume - p.mesh.volume) < 0.5))
            if not e["same_as_plate_part"]:
                issues.append(f"{rel}: geometry differs from {prel}")
            if os.path.basename(os.path.dirname(os.path.dirname(f))) != os.path.dirname(prel):
                issues.append(f"{rel}: STL folder does not match plate folder {prel}")
        else:
            issues.append(f"{rel}: no plate contains this part")
        if not st["watertight"]:
            issues.append(f"{rel}: not watertight")
        report["stls"][rel] = e
    stl_names = {os.path.splitext(os.path.basename(k))[0] for k in report["stls"]}
    for n in split_parts:
        if n not in stl_names:
            issues.append(f"{n}: in a plate but no individual STL")

    # ---- totals / duplicates ---------------------------------------------
    dup = {k: v for k, v in seen_ids.items() if len(v) > 1}
    missing = sorted(set(orig) - set(seen_ids), key=int)
    extra = sorted(set(seen_ids) - set(orig), key=lambda x: str(x))
    nyl = sum(v["parts"] for k, v in report["plates"].items() if k.startswith("NylonG") and "parts" in v)
    pla = sum(v["parts"] for k, v in report["plates"].items() if k.startswith("PLA") and "parts" in v)
    groups = {}
    for n, (_r, p) in split_parts.items():
        groups.setdefault(geom_key(p.local_mesh), []).append(n)
    report["totals"] = {"nylon_parts": nyl, "pla_parts": pla, "sum": nyl + pla, "original": len(orig_parts),
                        "duplicate_ids": dup, "missing_ids": missing, "extra_ids": extra,
                        "stl_count": len(report["stls"]),
                        "identical_copy_groups": {g[0]: len(g) for g in groups.values() if len(g) > 1}}
    if dup:
        issues.append(f"duplicate ids across plates: {dup}")
    if missing:
        issues.append(f"original ids missing from split: {missing}")

    # top-level NylonG_PulseXE copies identical to the zip's?
    import filecmp
    for f in sorted(glob.glob(os.path.join(ROOT, "NylonG_PulseXE", "*"))):
        b = os.path.basename(f)
        cand = glob.glob(os.path.join(SPLIT, "NylonG_PulseXE", "**", b), recursive=True)
        report.setdefault("toplevel_copies", {})[b] = bool(cand and filecmp.cmp(f, cand[0], shallow=False))

    with open(os.path.join(OUT, "source_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    # ---- summary ----------------------------------------------------------
    print("Original:", report["original"]["parts"], "objects;", report["original"]["on_plates"], "on plates;",
          report["original"]["per_plate"])
    for k, v in report["plates"].items():
        if "parse_error" in v:
            print(k, "PARSE ERROR", v["parse_error"]); continue
        print(f"{k:42s} parts={v['parts']:2d} min={v['extent_min']} max={v['extent_max']} "
              f"fits240x200={v['fits_240x200_usable']} minGap={v['min_bbox_gap_mm']}")
    print("Totals:", {k: v for k, v in report["totals"].items() if k != "identical_copy_groups"})
    print("Identical-copy groups:", report["totals"]["identical_copy_groups"])
    print("Top-level copies identical:", report.get("toplevel_copies"))
    nonwt = [k for k, v in report["original"]["objects"].items() if not v["watertight"]]
    print("Original non-watertight objects:", nonwt)
    print("Multi-body parts:", [e["name"] for v in report["plates"].values() for e in v.get("entries", []) if e["bodies"] != 1])
    print(f"Issues ({len(issues)}):")
    for i in issues:
        print("  -", i)


if __name__ == "__main__":
    main()
