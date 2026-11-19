"""Tests for duck_mass. Run from tools/duck_mass:  .venv/bin/python -m pytest -q"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duck_mass as dm  # noqa: E402

REAL_MODEL = dm.DEFAULT_MODEL
REAL_PARTS = dm.DEFAULT_PARTS
have_real = REAL_MODEL.exists() and REAL_PARTS.exists()

# ---------------------------------------------------------------- fixtures

TINY_XML = """<?xml version="1.0" ?>
<!-- tiny test model -->
<mujoco model="tiny">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <freejoint/>
      <inertial pos="0.01 0 0" mass="0.2" fullinertia="2e-4 2.5e-4 3e-4 1e-6 -2e-6 3e-6"/>
      <geom type="mesh" mesh="cube"/>
      <body name="arm" pos="0.05 0 0">
        <joint name="j1" axis="0 0 1"/>
        <inertial pos="0 0.02 0" mass="0.05" diaginertia="2e-5 2.5e-5 3e-5"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
      <body name="tip" pos="0 0.05 0">
        <joint name="j2" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="0.03" fullinertia="4e-6 5e-6 6e-6 0 0 0"/>
        <geom type="box" size="0.01 0.01 0.01"/>
      </body>
    </body>
  </worldbody>
  <asset>
    <mesh file="cube.stl"/>
  </asset>
</mujoco>
"""

# Binary STL of a small cube (12 triangles), so MuJoCo has a real mesh to load.
def _cube_stl(s: float = 0.01) -> bytes:
    import struct
    v = [(x, y, z) for x in (0, s) for y in (0, s) for z in (0, s)]
    faces = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    out = bytearray(b"\0" * 80) + struct.pack("<I", len(faces))
    for f in faces:
        out += struct.pack("<3f", 0, 0, 0)
        for i in f:
            out += struct.pack("<3f", *v[i])
        out += struct.pack("<H", 0)
    return bytes(out)


PARTS_CSV = (
    "part_file,label,pla_g_solid,petg_g_solid,petg_g_rough_author_settings\n"
    "p_a.stl,frame,10.0,10.2,8.0\n"
    "p_b.stl,frame,5.0,5.1,4.0\n"
    "p_c.stl,shell,20.0,20.5,18.0\n"
    "p_d.stl,ring,1.0,1.02,0.9\n"
)


@pytest.fixture
def tiny(tmp_path: Path) -> dict[str, Path]:
    src = tmp_path / "upstream"
    (src / "assets").mkdir(parents=True)
    (src / "assets" / "cube.stl").write_bytes(_cube_stl())
    model = src / "tiny.xml"
    model.write_text(TINY_XML)
    parts = tmp_path / "parts.csv"
    parts.write_text(PARTS_CSV)
    return {"model": model, "parts": parts, "out": tmp_path / "out", "dir": tmp_path}


def write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def run_apply(t, mapping_text: str, weights_text: str | None = None, **kw):
    mapping = write(t["dir"] / "mapping.toml", mapping_text)
    weights = None
    if weights_text is not None:
        weights = write(t["dir"] / "weights.csv", weights_text)
    return dm.apply(
        model_path=t["model"], parts_path=t["parts"], mapping_path=mapping,
        weights_path=weights, out_dir=t["out"],
        fallback_column=kw.get("fallback", "petg_g_rough_author_settings"),
        derive_column=kw.get("derive", "pla_g_solid"),
    )


def masses_in(path: Path) -> dict[str, float]:
    tree = dm.parse_xml(path)
    return {b.get("name"): float(b.find("inertial").get("mass"))
            for b in tree.iter("body") if b.find("inertial") is not None}


def inertial_attr(path: Path, body: str, attr: str) -> list[float]:
    tree = dm.parse_xml(path)
    for b in tree.iter("body"):
        if b.get("name") == body:
            return [float(x) for x in b.find("inertial").get(attr).split()]
    raise KeyError(body)


# ---------------------------------------------------------------- arithmetic

def test_mass_replacement_arithmetic(tiny):
    res = run_apply(
        tiny,
        '[bodies.base]\nparts = ["p_a.stl", "p_c.stl"]\nnon_printed_g = 150.0\n',
        "part_file,measured_g\np_a.stl,12.5\np_c.stl,22.0\n",
    )
    m = masses_in(res["xml_path"])
    assert m["base"] == pytest.approx((150.0 + 12.5 + 22.0) / 1000, abs=1e-12)
    body = next(b for b in res["report"]["bodies"] if b["name"] == "base")
    assert body["old_mass_kg"] == pytest.approx(0.2)
    assert body["delta_kg"] == pytest.approx(0.1845 - 0.2)
    assert body["used_estimates"] is False


def test_non_printed_only_body(tiny):
    res = run_apply(tiny, "[bodies.tip]\nparts = []\nnon_printed_g = 40\n")
    assert masses_in(res["xml_path"])["tip"] == pytest.approx(0.040)


def test_inertia_scaled_by_mass_ratio_and_com_unchanged(tiny):
    res = run_apply(
        tiny,
        '[bodies.base]\nparts = ["p_a.stl"]\nnon_printed_g = 290\n'
        '[bodies.arm]\nparts = ["p_b.stl"]\nnon_printed_g = 20\n',
        "part_file,measured_g\np_a.stl,10\np_b.stl,5\n",
    )
    out = res["xml_path"]
    r_base = 0.300 / 0.2
    assert inertial_attr(out, "base", "fullinertia") == pytest.approx(
        [x * r_base for x in (2e-4, 2.5e-4, 3e-4, 1e-6, -2e-6, 3e-6)], rel=1e-9)
    r_arm = 0.025 / 0.05
    assert inertial_attr(out, "arm", "diaginertia") == pytest.approx(
        [x * r_arm for x in (2e-5, 2.5e-5, 3e-5)], rel=1e-9)
    assert inertial_attr(out, "base", "pos") == pytest.approx([0.01, 0, 0])
    assert inertial_attr(out, "arm", "pos") == pytest.approx([0, 0.02, 0])


def test_unmapped_bodies_untouched(tiny):
    res = run_apply(tiny, "[bodies.arm]\nparts = []\nnon_printed_g = 60\n")
    out = res["xml_path"]
    src_tree, out_tree = dm.parse_xml(tiny["model"]), dm.parse_xml(out)
    for name in ("base", "tip"):
        a = next(b for b in src_tree.iter("body") if b.get("name") == name).find("inertial")
        b = next(b for b in out_tree.iter("body") if b.get("name") == name).find("inertial")
        assert a.attrib == b.attrib
    assert res["report"]["unmapped_bodies"] == ["base", "tip"]


def test_fallback_to_estimate_is_reported(tiny):
    res = run_apply(
        tiny,
        '[bodies.base]\nparts = ["p_a.stl", "p_b.stl"]\nnon_printed_g = 100\n'
        '[bodies.arm]\nparts = ["p_c.stl"]\nnon_printed_g = 10\n',
        "part_file,measured_g\np_a.stl,11\np_b.stl,\np_c.stl,19\n",
    )
    rep = res["report"]
    base = next(b for b in rep["bodies"] if b["name"] == "base")
    assert base["used_estimates"] is True
    pb = next(p for p in base["parts"] if p["part_file"] == "p_b.stl")
    assert pb["source"] == "estimate:petg_g_rough_author_settings"
    assert pb["grams"] == pytest.approx(4.0)
    assert rep["bodies_using_estimates"] == ["base"]
    assert masses_in(res["xml_path"])["base"] == pytest.approx(0.115)
    assert "base" in res["text"] and "ESTIMATE" in res["text"]


def test_no_weights_file_means_all_estimates(tiny):
    res = run_apply(tiny, '[bodies.arm]\nparts = ["p_c.stl"]\nnon_printed_g = 10\n',
                    fallback="petg_g_solid")
    assert masses_in(res["xml_path"])["arm"] == pytest.approx(0.0305)
    assert res["report"]["bodies_using_estimates"] == ["arm"]


def test_derived_non_printed_is_labelled_estimate(tiny):
    # base: 200 g in model, p_a PLA solid 10 g -> derived non-printed 190 g.
    res = run_apply(tiny, '[bodies.base]\nparts = ["p_a.stl"]\nnon_printed_g = "derive"\n',
                    "part_file,measured_g\np_a.stl,12\n")
    base = next(b for b in res["report"]["bodies"] if b["name"] == "base")
    assert base["non_printed_g"] == pytest.approx(190.0)
    assert base["non_printed_source"] == "derived-estimate:pla_g_solid"
    assert base["used_estimates"] is True
    assert masses_in(res["xml_path"])["base"] == pytest.approx(0.202)


# ---------------------------------------------------------------- validation errors

def test_part_assigned_to_two_bodies_is_error(tiny):
    with pytest.raises(dm.MassError, match="p_a.stl.*more than once"):
        run_apply(tiny, '[bodies.base]\nparts = ["p_a.stl"]\nnon_printed_g = 1\n'
                        '[bodies.arm]\nparts = ["p_a.stl"]\nnon_printed_g = 1\n')


def test_part_in_body_and_unassigned_is_error(tiny):
    with pytest.raises(dm.MassError, match="more than once"):
        run_apply(tiny, '[bodies.base]\nparts = ["p_a.stl"]\nnon_printed_g = 1\n'
                        '[unassigned]\nparts = ["p_a.stl", "p_b.stl"]\n')


def test_unknown_part_in_mapping_is_error(tiny):
    with pytest.raises(dm.MassError, match="not in parts.csv"):
        run_apply(tiny, '[bodies.base]\nparts = ["nope.stl"]\nnon_printed_g = 1\n')


def test_unknown_part_in_weights_is_error(tiny):
    with pytest.raises(dm.MassError, match="not in parts.csv"):
        run_apply(tiny, "[bodies.tip]\nparts = []\nnon_printed_g = 1\n",
                  "part_file,measured_g\nghost.stl,3\n")


def test_unknown_body_is_error(tiny):
    with pytest.raises(dm.MassError, match="not in the model"):
        run_apply(tiny, "[bodies.wing]\nparts = []\nnon_printed_g = 1\n")


def test_parts_without_non_printed_is_error(tiny):
    with pytest.raises(dm.MassError, match="non_printed_g"):
        run_apply(tiny, '[bodies.base]\nparts = ["p_a.stl"]\n')


def test_bad_fallback_column_is_error(tiny):
    with pytest.raises(dm.MassError, match="column"):
        run_apply(tiny, "[bodies.tip]\nparts = []\nnon_printed_g = 1\n", fallback="abs_g")


def test_body_without_parts_or_mass_is_left_unmapped(tiny):
    res = run_apply(tiny, "[bodies.base]\nparts = []\n")
    assert "base" in res["report"]["unmapped_bodies"]


# ---------------------------------------------------------------- output files and MuJoCo

def test_reports_written(tiny):
    res = run_apply(tiny, "[bodies.tip]\nparts = []\nnon_printed_g = 45\n")
    data = json.loads((tiny["out"] / "report.json").read_text())
    assert data["total_new_kg"] == pytest.approx(0.2 + 0.05 + 0.045)
    assert data["total_old_kg"] == pytest.approx(0.28)
    assert (tiny["out"] / "report.txt").read_text() == res["text"]


def test_generated_tiny_model_loads_with_meshes_and_total_matches(tiny):
    mujoco = pytest.importorskip("mujoco")
    res = run_apply(tiny, '[bodies.base]\nparts = ["p_a.stl"]\nnon_printed_g = 90\n',
                    "part_file,measured_g\np_a.stl,10\n")
    assert res["xml_path"].parent == tiny["out"]
    m = mujoco.MjModel.from_xml_path(str(res["xml_path"]))
    assert m.nmesh == 1
    assert float(m.body_mass.sum()) == pytest.approx(0.28 - 0.2 + 0.1, abs=1e-6)


@pytest.mark.skipif(not have_real, reason="real model or parts.csv missing")
def test_real_model_template_lists_every_body_and_part(tmp_path):
    out = tmp_path / "mapping.template.toml"
    dm.make_template(REAL_MODEL, REAL_PARTS, out, tmp_path / "weights.template.csv")
    data = tomllib.loads(out.read_text())
    model_bodies = [b.name for b in dm.model_bodies(dm.parse_xml(REAL_MODEL))]
    assert sorted(data["bodies"]) == sorted(model_bodies)
    assert len(model_bodies) == 15
    parts = dm.read_parts(REAL_PARTS)
    assert sorted(data["unassigned"]["parts"]) == sorted(parts)
    assert len(parts) == 67
    # Guesses must be comments only: nothing is assigned, no mass is set.
    for body in data["bodies"].values():
        assert body["parts"] == []
        assert "non_printed_g" not in body


@pytest.mark.skipif(not have_real, reason="real model or parts.csv missing")
def test_real_model_end_to_end_total_mass_in_mujoco(tmp_path):
    mujoco = pytest.importorskip("mujoco")
    old = mujoco.MjModel.from_xml_path(str(REAL_MODEL))
    old_total = float(old.body_mass.sum())
    assert old_total == pytest.approx(0.737, abs=0.001)
    # MADE-UP numbers, only to exercise the code path.
    mapping = write(tmp_path / "m.toml",
                    '[bodies.ankle_left]\nparts = ["plate5_sole_111_id44.stl"]\nnon_printed_g = 12.0\n'
                    '[bodies.jaw_soft]\nparts = ["plate2_big-shell_5_id6.stl", "plate1_beak-jaw_4_id4.stl"]\n'
                    'non_printed_g = "derive"\n')
    weights = write(tmp_path / "w.csv", "part_file,measured_g\nplate5_sole_111_id44.stl,19.3\n")
    res = dm.apply(model_path=REAL_MODEL, parts_path=REAL_PARTS, mapping_path=mapping,
                   weights_path=weights, out_dir=tmp_path / "out",
                   fallback_column="petg_g_rough_author_settings", derive_column="pla_g_solid")
    new = mujoco.MjModel.from_xml_path(str(res["xml_path"]))
    assert new.nmesh == old.nmesh and new.nbody == old.nbody
    rep = res["report"]
    mapped = [b for b in rep["bodies"] if b["mapped"]]
    expected = old_total - sum(b["old_mass_kg"] for b in mapped) + sum(b["new_mass_kg"] for b in mapped)
    assert float(new.body_mass.sum()) == pytest.approx(expected, abs=1e-6)
    assert rep["mujoco_check"]["total_mass_kg"] == pytest.approx(expected, abs=1e-6)
    assert rep["bodies_using_estimates"] == ["jaw_soft"]
    # Upstream file is untouched by the run.
    assert "GENERATED" not in REAL_MODEL.read_text()
