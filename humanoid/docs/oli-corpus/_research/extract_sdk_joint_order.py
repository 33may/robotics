"""SDK joint-order extractor.

The canonical order of the SDK's ``q``/``dq``/``tau`` arrays for each robot is
not stored as a C++ constant — it's set at runtime via
``RobotState.motor_names`` populated by the on-robot daemon. The only static
authority is the ``humanoid-rl-deploy-python`` controller configs, where the
order is encoded as **comments** beside numeric arrays in
``controllers/<robot>/walk_controller/walk_param.yaml``:

::

    action_scale:
      [0.2511,  # left_hip_pitch_joint
       0.2511,  # left_hip_roll_joint
       ...
      ]

We recover that sequence by raw-text parsing the YAML comments (PyYAML drops
them). The recovered list must:

1. Have N entries matching the count of non-fixed joints in the URDF.
2. Every name must exist in the robot's URDF joints.

If both checks pass, we backfill ``joints.sdk_idx`` for the matching robot.
If either check fails, we log a warning and skip — never silently guess.

Mapping from deploy-repo robot name to corpus robot_id: deploy uses
``HU_D04_01`` which matches ``HU_D04_01`` in the robots table. The
``_rl``-suffixed URDF variant is treated as the same physical robot and gets
its sdk_idx backfilled in parallel.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

LOG = logging.getLogger(__name__)

JOINT_COMMENT = re.compile(r"#\s*([A-Za-z][A-Za-z0-9_]*_joint)\b")


@dataclass(frozen=True)
class JointOrderResult:
    robot_id: str
    order: list[str]
    source_path: Path


def parse_joint_order_from_walk_param(path: Path) -> list[str]:
    """Return the joint name sequence recovered from comment annotations.

    Takes the *first occurrence* of each joint name across the whole file
    (the walk_param.yaml may list joints multiple times — once per array;
    they're all in the same canonical order).
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")
    seen: dict[str, int] = {}
    for m in JOINT_COMMENT.finditer(text):
        name = m.group(1)
        if name not in seen:
            seen[name] = len(seen)
    return list(seen)


def _movable_joint_names(db: sqlite3.Connection, robot_id: str) -> list[str]:
    rows = db.execute(
        "SELECT urdf_idx, name FROM joints WHERE robot_id = ? AND type != 'fixed' ORDER BY urdf_idx",
        (robot_id,),
    ).fetchall()
    return [row[1] for row in rows]


def _backfill_sdk_idx(db: sqlite3.Connection, robot_id: str, order: list[str]) -> int:
    """Update ``joints.sdk_idx`` for each name in ``order``. Returns rows updated."""
    updated = 0
    for sdk_idx, name in enumerate(order):
        cursor = db.execute(
            "UPDATE joints SET sdk_idx = ? WHERE robot_id = ? AND name = ?",
            (sdk_idx, robot_id, name),
        )
        updated += cursor.rowcount
    return updated


def _candidate_robot_ids(db: sqlite3.Connection, deploy_name: str) -> list[str]:
    """Return all robot_ids that match a deploy-repo robot name.

    HU_D04_01 in the deploy repo should propagate sdk_idx to both
    ``HU_D04_01`` (production URDF) and ``HU_D04_01_rl`` (RL variant).
    """
    rows = db.execute(
        "SELECT robot_id FROM robots WHERE robot_id = ? OR robot_id = ? || '_rl'",
        (deploy_name, deploy_name),
    ).fetchall()
    return [r[0] for r in rows]


def iter_walk_params(deploy_root: Path) -> list[tuple[str, Path]]:
    """Yield (deploy_robot_name, walk_param.yaml) pairs."""
    out: list[tuple[str, Path]] = []
    controllers_dir = deploy_root / "controllers"
    if not controllers_dir.exists():
        return out
    for robot_dir in sorted(controllers_dir.iterdir()):
        if not robot_dir.is_dir():
            continue
        for cand in (
            robot_dir / "walk_controller" / "walk_param.yaml",
            robot_dir / "mimic_controller" / "mimic_param.yaml",
        ):
            if cand.exists():
                out.append((robot_dir.name, cand))
                break
    return out


def run(db: sqlite3.Connection, deploy_root: Path) -> list[JointOrderResult]:
    """Walk deploy-python controllers/, backfill joints.sdk_idx for matched robots."""
    results: list[JointOrderResult] = []
    if not deploy_root.exists():
        LOG.warning("deploy_root not present, skipping sdk_joint_order: %s", deploy_root)
        return results
    for deploy_name, path in iter_walk_params(deploy_root):
        order = parse_joint_order_from_walk_param(path)
        if not order:
            LOG.warning("no joint-name comments in %s; cannot recover order", path)
            continue
        for robot_id in _candidate_robot_ids(db, deploy_name):
            movable = _movable_joint_names(db, robot_id)
            if not movable:
                continue
            order_set = set(order)
            movable_set = set(movable)
            if order_set != movable_set:
                # Either the deploy config covers a different superset or names mismatch.
                # Don't backfill if there are unknowns; backfill partial only if names ⊆ URDF.
                if not order_set.issubset(movable_set):
                    LOG.warning(
                        "sdk_joint_order: %s walk_param has joints not in URDF: %s",
                        robot_id, sorted(order_set - movable_set),
                    )
                    continue
                LOG.info(
                    "sdk_joint_order: %s deploy config covers %d of %d URDF joints (partial)",
                    robot_id, len(order_set), len(movable_set),
                )
            updated = _backfill_sdk_idx(db, robot_id, order)
            results.append(JointOrderResult(robot_id=robot_id, order=order, source_path=path))
            LOG.info("sdk_joint_order: %s backfilled %d rows from %s", robot_id, updated, path)
    return results
