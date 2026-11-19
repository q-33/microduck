"""Shared helpers for the Microduck print-file tools.

Minimal 3MF reader/writer (core spec + Bambu production-extension
components) and mesh statistics. Uses numpy + trimesh from .venv/.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
import xml.etree.ElementTree as _ET  # noqa: F401 (types only)
import defusedxml.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import trimesh

NS_CORE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
NS_PROD = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
NS = {"c": NS_CORE}
PATH_ATTR = "{%s}path" % NS_PROD

# PLA density used for mass estimates (g/cm3)
PLA_DENSITY = 1.24


def parse_transform(s: str | None) -> np.ndarray:
    """3MF transform: 12 numbers, row-major 4x3 (m00 m01 m02 m10 ... m30 m31 m32).
    Point p (row vector) maps to [p 1] @ M. Returns a 4x4 matrix for column vectors."""
    m = np.eye(4)
    if not s:
        return m
    v = [float(x) for x in s.split()]
    if len(v) != 12:
        raise ValueError(f"bad transform {s!r}")
    a = np.array(v).reshape(4, 3)
    m[:3, :3] = a[:3, :].T
    m[:3, 3] = a[3, :]
    return m


def format_transform(m: np.ndarray) -> str:
    a = np.vstack([m[:3, :3].T, m[:3, 3]])
    return " ".join(f"{x:.6f}".rstrip("0").rstrip(".") if "." in f"{x:.6f}" else f"{x:.6f}" for x in a.flatten())


@dataclass
class Part:
    name: str
    object_id: str
    mesh: trimesh.Trimesh          # in plate (world) coordinates
    local_mesh: trimesh.Trimesh    # in object coordinates
    transform: np.ndarray          # build item transform (4x4)
    printable: bool = True
    extra: dict = field(default_factory=dict)


def _read_mesh_elem(mesh_el) -> trimesh.Trimesh:
    verts = mesh_el.find("c:vertices", NS)
    tris = mesh_el.find("c:triangles", NS)
    v = np.array([[float(e.get("x")), float(e.get("y")), float(e.get("z"))]
                  for e in verts.findall("c:vertex", NS)], dtype=np.float64)
    f = np.array([[int(e.get("v1")), int(e.get("v2")), int(e.get("v3"))]
                  for e in tris.findall("c:triangle", NS)], dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


def read_3mf(path: str) -> list[Part]:
    """Return every build item of a 3MF as a Part with its mesh placed in plate
    coordinates. Handles meshes, components and production-extension p:path
    references to other model files in the package."""
    zf = zipfile.ZipFile(path)
    names = set(zf.namelist())
    rels = zf.read("_rels/.rels").decode("utf-8")
    m = re.search(r'Target="([^"]+)"[^>]*Type="[^"]*3dmodel"', rels) or \
        re.search(r'Type="[^"]*3dmodel"[^>]*Target="([^"]+)"', rels)
    root_path = m.group(1).lstrip("/") if m else "3D/3dmodel.model"
    cache: dict[str, ET.Element] = {}

    def model(p):
        p = p.lstrip("/")
        if p not in cache:
            if p not in names:
                raise FileNotFoundError(f"{p} missing from {path}")
            cache[p] = ET.fromstring(zf.read(p))
        return cache[p]

    def resolve(file_path, oid, depth=0) -> trimesh.Trimesh:
        if depth > 8:
            raise RuntimeError("component recursion too deep")
        root = model(file_path)
        for obj in root.find("c:resources", NS).findall("c:object", NS):
            if obj.get("id") == oid:
                mesh_el = obj.find("c:mesh", NS)
                if mesh_el is not None:
                    return _read_mesh_elem(mesh_el)
                comps = obj.find("c:components", NS)
                parts = []
                for c in comps.findall("c:component", NS):
                    sub_path = c.get(PATH_ATTR) or file_path
                    sub = resolve(sub_path, c.get("objectid"), depth + 1).copy()
                    sub.apply_transform(parse_transform(c.get("transform")))
                    parts.append(sub)
                return trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
        raise KeyError(f"object {oid} not in {file_path}")

    root = model(root_path)
    obj_names = {o.get("id"): o.get("name") for o in
                 root.find("c:resources", NS).findall("c:object", NS)}
    out = []
    for item in root.find("c:build", NS).findall("c:item", NS):
        oid = item.get("objectid")
        local = resolve(root_path, oid)
        t = parse_transform(item.get("transform"))
        world = local.copy()
        world.apply_transform(t)
        out.append(Part(name=obj_names.get(oid) or f"object_{oid}", object_id=oid,
                        mesh=world, local_mesh=local, transform=t,
                        printable=item.get("printable", "1") != "0"))
    return out


def write_3mf(path: str, parts: list[tuple[str, trimesh.Trimesh, np.ndarray]]) -> None:
    """Write a plain core-spec 3MF: one object per part (mesh in object coords)
    and one build item per part carrying its placement transform."""
    buf = io.StringIO()
    buf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    buf.write(f'<model unit="millimeter" xml:lang="en-US" xmlns="{NS_CORE}">')
    buf.write('<metadata name="Application">microduck tools/mdlib.py</metadata><resources>')
    for i, (name, mesh, _t) in enumerate(parts, start=1):
        buf.write(f'<object id="{i}" type="model" name="{name}"><mesh><vertices>')
        for x, y, z in mesh.vertices:
            buf.write(f'<vertex x="{x:.5f}" y="{y:.5f}" z="{z:.5f}"/>')
        buf.write("</vertices><triangles>")
        for a, b, c in mesh.faces:
            buf.write(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>')
        buf.write("</triangles></mesh></object>")
    buf.write("</resources><build>")
    for i, (_n, _m, t) in enumerate(parts, start=1):
        buf.write(f'<item objectid="{i}" transform="{format_transform(t)}"/>')
    buf.write("</build></model>")
    ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
          '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
          '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", ct)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", buf.getvalue())


def mesh_stats(mesh: trimesh.Trimesh) -> dict:
    b = mesh.bounds
    ext = b[1] - b[0]
    try:
        bodies = len(mesh.split(only_watertight=False))
    except Exception:
        bodies = -1
    return {
        "verts": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "min": b[0].round(3).tolist(),
        "max": b[1].round(3).tolist(),
        "size": ext.round(3).tolist(),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "volume_mm3": float(mesh.volume),
        "area_mm2": float(mesh.area),
        "bodies": bodies,
        "degenerate_faces": int((mesh.area_faces < 1e-12).sum()),
    }


def geom_key(mesh: trimesh.Trimesh) -> str:
    """Placement-independent fingerprint: hash of rounded, centred, sorted
    vertex coordinates. Identical copies of a part give the same key."""
    v = np.unique(np.round(mesh.vertices, 3), axis=0)
    v = np.round(v - v.mean(axis=0), 2)
    v = v[np.lexsort(v.T[::-1])]
    return hashlib.sha1(v.tobytes()).hexdigest()[:16] + f"_{len(mesh.faces)}"


def z_profile_key(mesh: trimesh.Trimesh) -> np.ndarray:
    """Sorted vertex z (relative to min z). Invariant to XY translation and
    rotation about Z, so it tests 'same print orientation'."""
    v = np.unique(np.round(mesh.vertices, 3), axis=0)
    z = v[:, 2] - v[:, 2].min()
    return np.sort(z)
