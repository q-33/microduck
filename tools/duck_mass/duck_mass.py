#!/usr/bin/env python3
"""duck_mass: put weighed 3D-printed part masses into the Microduck MuJoCo model.

Two commands:

  template  write a mapping template (one table per model body, every printed
            part listed as unassigned) and a weights template CSV.
  apply     read the filled-in mapping + weights, write a GENERATED copy of the
            model with new body masses, plus report.txt and report.json.

The upstream model is never modified. Body mass = sum(printed parts) +
non_printed_g. The inertia tensor is scaled by new_mass/old_mass and the
centre of mass is kept (assumes the mass distribution keeps its shape).

Standard library only; MuJoCo is used for the optional load check if installed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import sys
import tomllib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # microduck/
DEFAULT_MODEL = ROOT / "external/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml"
DEFAULT_PARTS = ROOT / "print/parts.csv"
DEFAULT_OUT = ROOT / "out/duck_mass"
DEFAULT_FALLBACK = "petg_g_rough_author_settings"
DEFAULT_DERIVE = "pla_g_solid"

# Proposed assignments, written to the template as COMMENTS ONLY.
# Basis: these model bodies contain meshes named sole_* / foot_*, and parts.csv
# has two parts labelled "sole" and two labelled "foot-housing". The labels are
# themselves guesses from geometry, and left/right is unknown.
GUESSES = {
    "ankle_left": ("sole_left, foot_left",
                   ["plate5_sole_111_id44.stl or plate5_sole_222_id46.stl",
                    "plate1_foot-housing_grbrgn_id64.stl or plate1_foot-housing_thngthm_id66.stl"]),
    "ankle_right": ("sole_right, foot_right",
                    ["the other sole", "the other foot-housing"]),
}


class MassError(Exception):
    """Bad input. The message says what to fix."""


# ---------------------------------------------------------------- inputs

def parse_xml(path: Path) -> ET.ElementTree:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.parse(path, parser)


@dataclass
class Body:
    name: str
    elem: ET.Element
    inertial: ET.Element | None
    meshes: list[str] = field(default_factory=list)

    @property
    def mass_kg(self) -> float | None:
        return None if self.inertial is None else float(self.inertial.get("mass"))


def model_bodies(tree: ET.ElementTree) -> list[Body]:
    out = []
    for b in tree.getroot().iter("body"):
        name = b.get("name")
        if not name:
            continue
        meshes = []
        for g in b.findall("geom"):
            m = g.get("mesh")
            if m and m not in meshes:
                meshes.append(m)
        out.append(Body(name, b, b.find("inertial"), meshes))
    return out


def read_parts(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    parts: dict[str, dict[str, str]] = {}
    for r in rows:
        name = r["part_file"].strip()
        if name in parts:
            raise MassError(f"{path}: part {name} appears twice")
        parts[name] = r
    return parts


def estimate_g(parts: dict, part: str, column: str) -> float:
    row = parts[part]
    if column not in row:
        raise MassError(f"column {column!r} is not in parts.csv "
                        f"(have: {', '.join(k for k in row if k.endswith('_g') or '_g_' in k)})")
    return float(row[column])


def read_weights(path: Path | None, parts: dict) -> dict[str, float]:
    if path is None:
        return {}
    weights: dict[str, float] = {}
    with open(path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f), start=2):
            name = (r.get("part_file") or "").strip()
            val = (r.get("measured_g") or "").strip()
            if not name:
                continue
            if name not in parts:
                raise MassError(f"{path} line {i}: part {name} is not in parts.csv")
            if name in weights:
                raise MassError(f"{path} line {i}: part {name} weighed twice")
            if val == "":
                continue
            g = float(val)
            if g <= 0:
                raise MassError(f"{path} line {i}: measured_g must be > 0, got {val}")
            weights[name] = g
    return weights


def load_mapping(path: Path, parts: dict, bodies: dict[str, Body]) -> dict[str, dict]:
    data = tomllib.loads(Path(path).read_text())
    seen: dict[str, str] = {}

    def claim(part: str, where: str):
        if part not in parts:
            raise MassError(f"mapping {where}: part {part} is not in parts.csv")
        if part in seen:
            raise MassError(f"mapping: part {part} is listed more than once "
                            f"({seen[part]} and {where})")
        seen[part] = where

    for part in data.get("unassigned", {}).get("parts", []):
        claim(part, "[unassigned]")

    mapping = {}
    for name, entry in data.get("bodies", {}).items():
        if name not in bodies:
            raise MassError(f"mapping: body {name} is not in the model")
        plist = entry.get("parts", [])
        for p in plist:
            claim(p, f"[bodies.{name}]")
        npm = entry.get("non_printed_g")
        if npm is None:
            if plist:
                raise MassError(f"mapping [bodies.{name}]: parts are listed but non_printed_g "
                                "is not set (grams, or \"derive\")")
            continue  # body left as in the model
        if isinstance(npm, str):
            if npm != "derive":
                raise MassError(f"mapping [bodies.{name}]: non_printed_g must be a number "
                                f"or \"derive\", got {npm!r}")
        elif not isinstance(npm, (int, float)) or npm < 0:
            raise MassError(f"mapping [bodies.{name}]: non_printed_g must be >= 0")
        if bodies[name].inertial is None:
            raise MassError(f"body {name} has no <inertial> in the model; cannot set its mass")
        mapping[name] = {"parts": list(plist), "non_printed_g": npm}
    return mapping


# ---------------------------------------------------------------- core

def compute(bodies: list[Body], parts: dict, mapping: dict, weights: dict,
            fallback_column: str, derive_column: str) -> list[dict]:
    if parts:  # validate column names up front, even if unused
        first = next(iter(parts))
        estimate_g(parts, first, fallback_column)
        estimate_g(parts, first, derive_column)
    rows = []
    for b in bodies:
        old = b.mass_kg
        row = {"name": b.name, "old_mass_kg": old, "new_mass_kg": old, "delta_kg": 0.0,
               "mapped": False, "parts": [], "non_printed_g": None,
               "non_printed_source": None, "used_estimates": False}
        if b.name in mapping:
            m = mapping[b.name]
            printed = 0.0
            for p in m["parts"]:
                if p in weights:
                    g, src = weights[p], "measured"
                else:
                    g, src = estimate_g(parts, p, fallback_column), f"estimate:{fallback_column}"
                    row["used_estimates"] = True
                row["parts"].append({"part_file": p, "grams": g, "source": src})
                printed += g
            if m["non_printed_g"] == "derive":
                npm = old * 1000 - sum(estimate_g(parts, p, derive_column) for p in m["parts"])
                if npm <= 0:
                    raise MassError(f"body {b.name}: derived non-printed mass is {npm:.3f} g "
                                    "(model mass minus estimate of its parts); set non_printed_g")
                row["non_printed_source"] = f"derived-estimate:{derive_column}"
                row["used_estimates"] = True
            else:
                npm = float(m["non_printed_g"])
                row["non_printed_source"] = "given"
            new = (printed + npm) / 1000.0
            if new <= 0:
                raise MassError(f"body {b.name}: new mass would be {new} kg")
            row.update(mapped=True, non_printed_g=npm, new_mass_kg=new, delta_kg=new - old)
        rows.append(row)
    return rows


def _fmt(x: float) -> str:
    return f"{x:.10g}"


def rewrite_model(tree: ET.ElementTree, bodies: list[Body], rows: list[dict],
                  model_path: Path) -> None:
    """Set masses, scale inertia, and make asset paths absolute. Modifies tree."""
    by_name = {b.name: b for b in bodies}
    for r in rows:
        if not r["mapped"]:
            continue
        inertial = by_name[r["name"]].inertial
        ratio = r["new_mass_kg"] / r["old_mass_kg"]
        inertial.set("mass", _fmt(r["new_mass_kg"]))
        for attr in ("fullinertia", "diaginertia"):
            if inertial.get(attr) is not None:
                inertial.set(attr, " ".join(_fmt(float(v) * ratio)
                                            for v in inertial.get(attr).split()))
    root = tree.getroot()
    src_dir = model_path.resolve().parent
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    for attr in ("meshdir", "texturedir", "assetdir"):
        val = compiler.get(attr)
        if val is not None:
            compiler.set(attr, str((src_dir / val).resolve()))
    if compiler.get("meshdir") is None and compiler.get("assetdir") is None:
        compiler.set("meshdir", str(src_dir))
    for inc in root.iter("include"):
        inc.set("file", str((src_dir / inc.get("file")).resolve()))


def mujoco_check(xml_path: Path) -> dict:
    try:
        import mujoco  # type: ignore
    except ImportError:
        return {"loaded": None, "note": "mujoco not installed; load not checked"}
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    return {"loaded": True, "mujoco_version": mujoco.__version__, "nbody": int(m.nbody),
            "nmesh": int(m.nmesh), "total_mass_kg": float(m.body_mass.sum())}


def build_report(rows, model_path, mapping_path, weights_path, parts, mapping,
                 fallback_column, derive_column, xml_path, check) -> tuple[dict, str]:
    assigned = {p for m in mapping.values() for p in m["parts"]}
    total_old = sum(r["old_mass_kg"] or 0 for r in rows)
    total_new = sum(r["new_mass_kg"] or 0 for r in rows)
    report = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "mapping": str(mapping_path), "weights": str(weights_path) if weights_path else None,
        "fallback_column": fallback_column, "derive_column": derive_column,
        "output_model": str(xml_path),
        "bodies": rows,
        "total_old_kg": total_old, "total_new_kg": total_new, "total_delta_kg": total_new - total_old,
        "unmapped_bodies": [r["name"] for r in rows if not r["mapped"]],
        "bodies_using_estimates": [r["name"] for r in rows if r["used_estimates"]],
        "parts_not_in_any_body": sorted(set(parts) - assigned),
        "mujoco_check": check,
    }
    L = [f"duck_mass report  ({report['generated_at']})",
         f"model:   {model_path}", f"output:  {xml_path}",
         f"weights: {weights_path or '(none: every part uses the estimate)'}",
         f"estimate column for unweighed parts: {fallback_column}", "",
         f"{'body':<18}{'old g':>10}{'new g':>10}{'delta g':>10}  note"]
    for r in rows:
        old = r["old_mass_kg"]
        if old is None:
            L.append(f"{r['name']:<18}{'-':>10}{'-':>10}{'-':>10}  no <inertial>")
            continue
        if not r["mapped"]:
            note = "unmapped (unchanged)"
        else:
            est = [p["part_file"] for p in r["parts"] if p["source"] != "measured"]
            notes = []
            if est:
                notes.append(f"ESTIMATE for {len(est)} of {len(r['parts'])} parts")
            if r["non_printed_source"] != "given":
                notes.append("ESTIMATE non-printed mass (derived)")
            note = "; ".join(notes) or "all measured"
        L.append(f"{r['name']:<18}{old*1000:>10.3f}{r['new_mass_kg']*1000:>10.3f}"
                 f"{r['delta_kg']*1000:>+10.3f}  {note}")
    L += ["", f"{'TOTAL':<18}{total_old*1000:>10.3f}{total_new*1000:>10.3f}"
              f"{(total_new-total_old)*1000:>+10.3f}", ""]
    L.append("bodies using estimates: " + (", ".join(report["bodies_using_estimates"]) or "none"))
    L.append("unmapped bodies: " + (", ".join(report["unmapped_bodies"]) or "none"))
    L.append(f"printed parts not assigned to any body: {len(report['parts_not_in_any_body'])}"
             f" of {len(parts)}")
    if check.get("loaded"):
        L.append(f"MuJoCo {check['mujoco_version']} load: OK, {check['nmesh']} meshes, "
                 f"total mass {check['total_mass_kg']*1000:.3f} g")
    else:
        L.append(f"MuJoCo load: {check.get('note')}")
    L.append("Inertia scaled by new/old mass per body; centre of mass unchanged (approximation).")
    return report, "\n".join(L) + "\n"


def apply(model_path: Path, parts_path: Path, mapping_path: Path, weights_path: Path | None,
          out_dir: Path, fallback_column: str = DEFAULT_FALLBACK,
          derive_column: str = DEFAULT_DERIVE) -> dict:
    model_path = Path(model_path).resolve()
    out_dir = Path(out_dir).resolve()
    if out_dir == model_path.parent:
        raise MassError("refusing to write next to the upstream model; choose another --out-dir")
    tree = parse_xml(model_path)
    bodies = model_bodies(tree)
    parts = read_parts(parts_path)
    mapping = load_mapping(mapping_path, parts, {b.name: b for b in bodies})
    weights = read_weights(weights_path, parts)
    rows = compute(bodies, parts, mapping, weights, fallback_column, derive_column)
    rewrite_model(tree, bodies, rows, model_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{model_path.stem}_measured.xml"
    root = tree.getroot()
    root.insert(0, ET.Comment(
        f" GENERATED by tools/duck_mass from {model_path} - do not edit; re-run duck_mass. "))
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=True)

    check = mujoco_check(xml_path)
    report, text = build_report(rows, model_path, mapping_path, weights_path, parts, mapping,
                                fallback_column, derive_column, xml_path, check)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "report.txt").write_text(text)
    return {"xml_path": xml_path, "report": report, "text": text}


# ---------------------------------------------------------------- template

def make_template(model_path: Path, parts_path: Path, out_path: Path,
                  weights_out: Path | None = None) -> None:
    bodies = model_bodies(parse_xml(model_path))
    parts = read_parts(parts_path)
    total = sum(b.mass_kg or 0 for b in bodies)
    L = [
        "# duck_mass mapping TEMPLATE (generated; copy to mapping.toml and edit the copy)",
        f"# model: {model_path}",
        f"# parts: {parts_path}",
        f"# model total mass: {total*1000:.3f} g over {len(bodies)} bodies",
        "#",
        "# For each body you want to update:",
        "#   parts         = printed part files (from parts.csv) bolted to this body.",
        "#                   Move each name out of [unassigned] at the bottom; a part may",
        "#                   appear only once in the whole file.",
        "#   non_printed_g = grams of everything else on the body (servos, bearings,",
        "#                   screws, PCBs, cables). Weigh it or look it up. Or write",
        "#                   \"derive\" to use: current body mass - PLA estimate of the",
        "#                   listed parts. That is an ESTIMATE and the report says so.",
        "# A body with no non_printed_g is left exactly as in the model.",
        "# The 'meshes' comment lists what the model's CAD puts in each body. The",
        "# replica's printed parts may not match the model's CAD one-to-one.",
        "",
    ]
    for b in bodies:
        L.append(f"[bodies.{b.name}]")
        if b.mass_kg is None:
            L.append("# no <inertial> in the model: this body cannot be updated")
        else:
            L.append(f"# current mass in model: {b.mass_kg*1000:.3f} g")
        L.append(f"# meshes: {', '.join(b.meshes) or '(none)'}")
        n_servo = sum(1 for g in b.elem.findall("geom") if g.get("mesh") == "xl330")
        L.append(f"# xl330 servo mesh instances in this body: {n_servo}")
        if b.name in GUESSES:
            meshes, cands = GUESSES[b.name]
            L.append(f"# GUESS, NOT FACT (body has meshes {meshes}; part labels are guesses too):")
            for c in cands:
                L.append(f"#   maybe {c}")
        L += ["parts = []", "# non_printed_g = ", ""]
    L.append("[unassigned]")
    L.append("# every printed part starts here; move names into a body's parts list.")
    L.append("parts = [")
    for name, r in parts.items():
        L.append(f'  "{name}",  # {r.get("label", "")}, PETG est '
                 f'{r.get("petg_g_rough_author_settings", "?")} g')
    L.append("]")
    Path(out_path).write_text("\n".join(L) + "\n")

    if weights_out is not None:
        with open(weights_out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["part_file", "label", "measured_g", "petg_g_rough_author_settings"])
            for name, r in parts.items():
                w.writerow([name, r.get("label", ""), "", r.get("petg_g_rough_author_settings", "")])


# ---------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    here = Path(__file__).resolve().parent

    t = sub.add_parser("template", help="write mapping and weights templates")
    t.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    t.add_argument("--parts", type=Path, default=DEFAULT_PARTS)
    t.add_argument("--out", type=Path, default=here / "mapping.template.toml")
    t.add_argument("--weights-out", type=Path, default=here / "weights.template.csv")

    a = sub.add_parser("apply", help="write the generated model and reports")
    a.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    a.add_argument("--parts", type=Path, default=DEFAULT_PARTS)
    a.add_argument("--mapping", type=Path, required=True)
    a.add_argument("--weights", type=Path, default=None,
                   help="CSV with part_file,measured_g (blank = not weighed)")
    a.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    a.add_argument("--fallback-column", default=DEFAULT_FALLBACK,
                   help="parts.csv column used for parts with no measurement")
    a.add_argument("--derive-column", default=DEFAULT_DERIVE,
                   help="parts.csv column used for non_printed_g = \"derive\"")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "template":
            make_template(args.model, args.parts, args.out, args.weights_out)
            print(f"wrote {args.out}\nwrote {args.weights_out}")
        else:
            res = apply(args.model, args.parts, args.mapping, args.weights, args.out_dir,
                        args.fallback_column, args.derive_column)
            print(res["text"], end="")
            print(f"wrote {res['xml_path']}\nwrote {args.out_dir / 'report.txt'}, report.json")
    except MassError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
