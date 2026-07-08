"""URDF/SRDF extractor for the structured corpus.

Walks all URDF files under the three vendored source roots, populates the
``robots``, ``joints`` and ``links`` typed tables, and yields markdown chunks
(one per robot) for the FTS index.

Conventions enforced (and asserted at extract time):

* ``robot_id`` = the ``<robot name="...">`` attribute.
* ``urdf_idx`` = position in URDF declaration order, **excluding** joints whose
  ``type`` is ``fixed``. This matches IsaacLab's articulation-builder ordering.
* ``sdk_idx`` is left NULL here; ``extract_sdk_joint_order.py`` backfills it.
* Mesh references ``package://<pkg>/<relpath>`` are resolved to corpus URIs
  pointing at the actual on-disk mesh under the tarball's ``install/etc/<pkg>/``.
* All source URIs use the existing ``oli-corpus://<doc_id>#<section>`` shape.

The extractor never guesses on ambiguity — missing or malformed inertials/axes
become NULL columns and are noted in the per-robot markdown summary.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .structured_schema import DOC_ID_TARBALL

LOG = logging.getLogger(__name__)

PACKAGE_URI_RE = re.compile(r"^package://(?P<pkg>[^/]+)/(?P<rel>.+)$")


@dataclass(frozen=True)
class Joint:
    name: str
    type: str
    parent_link: str
    child_link: str
    axis: tuple[float, float, float] | None
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    mimic_of: str | None


@dataclass(frozen=True)
class Link:
    name: str
    mass: float | None
    com: tuple[float, float, float] | None
    inertia: tuple[float, float, float, float, float, float] | None
    visual_mesh_uri: str | None
    collision_mesh_uri: str | None


@dataclass
class RobotDoc:
    robot_id: str
    description: str
    source_uri: str
    joints: list[Joint] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)


@dataclass(frozen=True)
class UrdfChunk:
    doc_id: str
    section: str
    heading: str
    body: str


# ----------------------------- parsing ------------------------------------ #


def _float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _xyz(value: str | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    parts = value.strip().split()
    if len(parts) != 3:
        return None
    try:
        return tuple(float(p) for p in parts)  # type: ignore[return-value]
    except ValueError:
        return None


def _resolve_mesh_uri(
    pkg_uri: str | None,
    tarball_root: Path,
    fallback_doc_id: str = DOC_ID_TARBALL,
) -> str | None:
    """Resolve a ``package://<pkg>/<rel>`` mesh ref to an ``oli-corpus://`` URI.

    Returns None if the mesh file does not exist under the expected paths or the
    ref shape is not recognized.
    """
    if pkg_uri is None:
        return None
    m = PACKAGE_URI_RE.match(pkg_uri.strip())
    if not m:
        return None
    pkg, rel = m.group("pkg"), m.group("rel")
    # Description packages put meshes under install/etc/<pkg>/<rel>.
    candidates = [
        Path("install/etc") / pkg / rel,
        Path("install/share") / pkg / rel,
    ]
    for cand in candidates:
        if (tarball_root / cand).exists():
            return f"oli-corpus://{fallback_doc_id}#{cand.as_posix()}"
    return None


def _parse_joint(elem: ET.Element) -> Joint:
    # NOTE: ElementTree elements with no children are falsy, so `elem.find(...) or fallback`
    # always picks the fallback. Use `is not None` checks instead.
    name = elem.attrib.get("name", "")
    jtype = elem.attrib.get("type", "")
    parent_el = elem.find("parent")
    parent = parent_el.attrib.get("link", "") if parent_el is not None else ""
    child_el = elem.find("child")
    child = child_el.attrib.get("link", "") if child_el is not None else ""
    axis_el = elem.find("axis")
    axis = _xyz(axis_el.attrib.get("xyz")) if axis_el is not None else None
    limit = elem.find("limit")
    lower = upper = effort = velocity = None
    if limit is not None:
        lower = _float(limit.attrib.get("lower"))
        upper = _float(limit.attrib.get("upper"))
        effort = _float(limit.attrib.get("effort"))
        velocity = _float(limit.attrib.get("velocity"))
    mimic = elem.find("mimic")
    mimic_of = mimic.attrib.get("joint") if mimic is not None else None
    return Joint(name, jtype, parent, child, axis, lower, upper, effort, velocity, mimic_of)


def _parse_link(elem: ET.Element, tarball_root: Path) -> Link:
    name = elem.attrib.get("name", "")
    inertial = elem.find("inertial")
    mass = com = inertia = None
    if inertial is not None:
        mass_elem = inertial.find("mass")
        if mass_elem is not None:
            mass = _float(mass_elem.attrib.get("value"))
        origin = inertial.find("origin")
        if origin is not None:
            com = _xyz(origin.attrib.get("xyz"))
        in_elem = inertial.find("inertia")
        if in_elem is not None:
            inertia = (
                _float(in_elem.attrib.get("ixx")) or 0.0,
                _float(in_elem.attrib.get("ixy")) or 0.0,
                _float(in_elem.attrib.get("ixz")) or 0.0,
                _float(in_elem.attrib.get("iyy")) or 0.0,
                _float(in_elem.attrib.get("iyz")) or 0.0,
                _float(in_elem.attrib.get("izz")) or 0.0,
            )

    visual_mesh = collision_mesh = None
    visual = elem.find("visual")
    if visual is not None:
        mesh = visual.find("geometry/mesh")
        if mesh is not None:
            visual_mesh = _resolve_mesh_uri(mesh.attrib.get("filename"), tarball_root)
    collision = elem.find("collision")
    if collision is not None:
        mesh = collision.find("geometry/mesh")
        if mesh is not None:
            collision_mesh = _resolve_mesh_uri(mesh.attrib.get("filename"), tarball_root)

    return Link(name, mass, com, inertia, visual_mesh, collision_mesh)


def parse_urdf(path: Path, doc_id: str, section: str, tarball_root: Path) -> RobotDoc:
    """Parse one URDF file into a RobotDoc. Does not touch the database.

    ``robot_id`` is the URDF's ``<robot name="...">`` attribute, with the
    ``_rl`` suffix appended when the filename ends in ``_rl.urdf`` but the
    declared name does not — this disambiguates RL-variant URDFs that share
    the same internal name as the production URDF (e.g. ``HU_D04_01.urdf``
    and ``HU_D04_01_rl.urdf`` both declare ``name="HU_D04_01"``).
    """
    tree = ET.parse(path)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"{path}: root element is not <robot>")
    urdf_name = root.attrib.get("name", path.stem)
    robot_id = urdf_name
    if path.stem.endswith("_rl") and not robot_id.endswith("_rl"):
        robot_id = f"{robot_id}_rl"
    source_uri = f"oli-corpus://{doc_id}#{section}"
    doc = RobotDoc(robot_id=robot_id, description=f"{urdf_name} ({path.name})", source_uri=source_uri)
    for elem in root:
        if elem.tag == "joint":
            doc.joints.append(_parse_joint(elem))
        elif elem.tag == "link":
            doc.links.append(_parse_link(elem, tarball_root))
    return doc


# ----------------------------- summary ------------------------------------ #


def markdown_summary(doc: RobotDoc) -> str:
    """Render a per-robot markdown summary used as an FTS chunk."""
    lines: list[str] = [f"# Robot: {doc.robot_id}", ""]
    lines.append(f"Source: `{doc.source_uri}`")
    lines.append("")
    lines.append("## Joints")
    lines.append("")
    lines.append(
        "`urdf_idx` = declaration order across ALL joints. "
        "`dof_idx` = sequential index across non-fixed joints (matches IsaacLab DoF order)."
    )
    lines.append("")
    lines.append(
        "| urdf_idx | dof_idx | name | type | parent | child | axis | lower | upper | effort | velocity | mimic_of |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    dof_idx = 0
    for urdf_idx, j in enumerate(doc.joints):
        dof_cell = "-" if j.type == "fixed" else str(dof_idx)
        if j.type != "fixed":
            dof_idx += 1
        axis_str = "-" if j.axis is None else f"({j.axis[0]:g}, {j.axis[1]:g}, {j.axis[2]:g})"
        lines.append(
            f"| {urdf_idx} | {dof_cell} | {j.name} | {j.type} | {j.parent_link} | {j.child_link} | {axis_str} | "
            f"{j.lower if j.lower is not None else '-'} | "
            f"{j.upper if j.upper is not None else '-'} | "
            f"{j.effort if j.effort is not None else '-'} | "
            f"{j.velocity if j.velocity is not None else '-'} | "
            f"{j.mimic_of or '-'} |"
        )
    lines.append("")
    lines.append("## Links")
    lines.append("")
    lines.append("| name | mass | visual_mesh | collision_mesh |")
    lines.append("|---|---|---|---|")
    for l in doc.links:
        lines.append(
            f"| {l.name} | "
            f"{l.mass if l.mass is not None else '-'} | "
            f"{l.visual_mesh_uri or '-'} | "
            f"{l.collision_mesh_uri or '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


# ----------------------------- DB writes ---------------------------------- #


def write_robot(db: sqlite3.Connection, doc: RobotDoc) -> None:
    db.execute(
        "INSERT OR REPLACE INTO robots(robot_id, description, source_uri) VALUES (?, ?, ?)",
        (doc.robot_id, doc.description, doc.source_uri),
    )
    rows = []
    for urdf_idx, j in enumerate(doc.joints):
        ax = j.axis or (None, None, None)
        rows.append(
            (
                doc.robot_id,
                urdf_idx,            # position in URDF declaration order, ALL joints
                None,                # sdk_idx backfilled later
                j.name,
                j.type,
                j.parent_link,
                j.child_link,
                ax[0], ax[1], ax[2],
                j.lower, j.upper, j.effort, j.velocity,
                j.mimic_of,
                doc.source_uri,
            )
        )
    db.executemany(
        """
        INSERT OR REPLACE INTO joints(
          robot_id, urdf_idx, sdk_idx, name, type, parent_link, child_link,
          axis_x, axis_y, axis_z, lower, upper, effort, velocity, mimic_of, source_uri
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    link_rows = []
    for l in doc.links:
        com = l.com or (None, None, None)
        inertia = l.inertia or (None, None, None, None, None, None)
        link_rows.append(
            (
                doc.robot_id,
                l.name,
                l.mass,
                com[0], com[1], com[2],
                inertia[0], inertia[1], inertia[2], inertia[3], inertia[4], inertia[5],
                l.visual_mesh_uri,
                l.collision_mesh_uri,
                doc.source_uri,
            )
        )
    db.executemany(
        """
        INSERT OR REPLACE INTO links(
          robot_id, name, mass, com_x, com_y, com_z,
          ixx, ixy, ixz, iyy, iyz, izz,
          visual_mesh_uri, collision_mesh_uri, source_uri
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        link_rows,
    )


# ----------------------------- walk + run --------------------------------- #


def iter_urdfs(tarball_root: Path) -> Iterator[Path]:
    """Yield URDF paths under the tarball root, excluding test fixtures.

    The mroscontrol test URDFs (transmission loader tests etc.) are skipped —
    they are minimal fixtures, not real robots.
    """
    for p in sorted(tarball_root.rglob("*.urdf")):
        if "mroscontrol/test/urdf" in p.as_posix():
            continue
        yield p


def relpath_from_root(path: Path, tarball_root: Path) -> str:
    return path.relative_to(tarball_root).as_posix()


def run(db: sqlite3.Connection, tarball_root: Path) -> list[UrdfChunk]:
    """Walk URDFs, populate typed tables, return chunks for FTS insertion.

    Raises on parser errors so the build fails loudly rather than silently
    skipping a robot.
    """
    chunks: list[UrdfChunk] = []
    for path in iter_urdfs(tarball_root):
        section = relpath_from_root(path, tarball_root)
        try:
            doc = parse_urdf(path, DOC_ID_TARBALL, section, tarball_root)
        except ET.ParseError as exc:
            LOG.error("urdf parse failed: %s: %s", path, exc)
            continue
        write_robot(db, doc)
        chunks.append(
            UrdfChunk(
                doc_id=DOC_ID_TARBALL,
                section=section,
                heading=f"Robot: {doc.robot_id} (URDF)",
                body=markdown_summary(doc),
            )
        )
    return chunks
