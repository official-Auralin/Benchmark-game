"""Helpers for slow CLI workflow regression tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SPLIT_ORDER = (
    "public_dev",
    "public_val",
    "private_eval",
)


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "gf01", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def clone_tree(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def remap_freeze_to_official_splits(
    freeze_dir: Path,
    *,
    split_order: tuple[str, ...] = OFFICIAL_SPLIT_ORDER,
) -> None:
    bundle_path = freeze_dir / "instance_bundle_v1.json"
    manifest_path = freeze_dir / "split_manifest_v1.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    instances = bundle.get("instances", [])
    if not isinstance(instances, list) or len(instances) != len(split_order):
        raise AssertionError("expected deterministic instance count for split remap")

    for instance, split_id in zip(instances, split_order):
        if not isinstance(instance, dict):
            raise AssertionError("instance row is not an object")
        instance["split_id"] = split_id

    manifest_rows = []
    for instance in instances:
        manifest_rows.append(
            {
                "instance_id": str(instance.get("instance_id", "")),
                "split_id": str(instance.get("split_id", "")),
                "mode": str(instance.get("mode", "normal")),
                "seed": int(instance.get("seed", 0)),
                "t_star": int(instance.get("t_star", 0)),
                "window_size": int(instance.get("window_size", 0)),
                "budget_timestep": int(instance.get("budget_timestep", 0)),
                "budget_atoms": int(instance.get("budget_atoms", 0)),
            }
        )
    manifest["instance_count"] = len(manifest_rows)
    manifest["instances"] = manifest_rows

    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def q033_manifest_paths(out_dir: Path) -> list[Path]:
    index_path = out_dir / "q033_manifest_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    paths = payload.get("manifest_paths", [])
    return [out_dir / Path(str(path)).name for path in paths]
