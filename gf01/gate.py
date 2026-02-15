"""
One-shot CI-style regression gate for the GF-01 harness.

This module runs a fixed sequence of commands and checks, producing a single
machine-readable summary. It is intended to provide a reproducible "go/no-go"
signal for core harness correctness before publishing benchmark results.
"""
# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Bobby Veihman"
__copyright__ = "Academic Commons"
__license__ = "License Name"
__version__ = "1.0.0"
__maintainer__ = "Bobby Veihman"
__email__ = "bv2340@columbia.edu"
__status__ = "Development"

import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _run_subprocess(cmd: list[str], cwd: Path) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    end = time.perf_counter()
    stdout_lines = proc.stdout.splitlines()
    stderr_lines = proc.stderr.splitlines()
    return {
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "returncode": int(proc.returncode),
        "duration_ms": (end - start) * 1000.0,
        "stdout_preview": stdout_lines[-20:],
        "stderr_preview": stderr_lines[-20:],
        "stdout_text": proc.stdout,
    }


def _parse_json(stdout_text: str) -> tuple[bool, dict[str, Any]]:
    try:
        payload = json.loads(stdout_text)
        if isinstance(payload, dict):
            return True, payload
    except Exception:
        pass
    return False, {}


def run_regression_gate(
    root: Path,
    fixture_root: Path,
    seed_checks: int,
    seed_profile: int,
    public_count: int,
    private_count: int,
    fail_fast: bool = False,
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []

    def add_step(name: str, result: dict[str, Any], passed: bool, notes: dict[str, Any] | None = None) -> bool:
        step = {
            "name": name,
            "passed": bool(passed),
            "cmd": result.get("cmd", ""),
            "returncode": int(result.get("returncode", 1)),
            "duration_ms": float(result.get("duration_ms", 0.0)),
            "stdout_preview": result.get("stdout_preview", []),
            "stderr_preview": result.get("stderr_preview", []),
        }
        if notes:
            step["notes"] = notes
        steps.append(step)
        return passed

    # Step 1: compile
    r_compile = _run_subprocess([sys.executable, "-m", "compileall", "gf01", "tests"], cwd=root)
    ok = add_step("compileall", r_compile, r_compile["returncode"] == 0)
    if fail_fast and not ok:
        return _finalize(steps)

    # Step 2: unit tests
    r_tests = _run_subprocess(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"],
        cwd=root,
    )
    ok = add_step("unittest", r_tests, r_tests["returncode"] == 0)
    if fail_fast and not ok:
        return _finalize(steps)

    # Step 3: benchmark checks
    r_checks = _run_subprocess([sys.executable, "-m", "gf01", "checks", "--seed", str(seed_checks)], cwd=root)
    checks_ok = r_checks["returncode"] == 0
    checks_payload_ok, checks_payload = _parse_json(str(r_checks.get("stdout_text", "")))
    failed_checks: list[str] = []
    if checks_payload_ok:
        for key, value in checks_payload.items():
            if isinstance(value, dict) and not bool(value.get("passed", False)):
                failed_checks.append(str(key))
    else:
        checks_ok = False
    checks_ok = checks_ok and len(failed_checks) == 0
    ok = add_step(
        "checks",
        r_checks,
        checks_ok,
        notes={"failed_checks": failed_checks, "json_parsed": checks_payload_ok},
    )
    if fail_fast and not ok:
        return _finalize(steps)

    # Step 4: profiling gates
    r_profile = _run_subprocess(
        [
            sys.executable,
            "-m",
            "gf01",
            "profile",
            "--seed",
            str(seed_profile),
            "--public-count",
            str(public_count),
            "--private-count",
            str(private_count),
        ],
        cwd=root,
    )
    profile_ok = r_profile["returncode"] == 0
    profile_payload_ok, profile_payload = _parse_json(str(r_profile.get("stdout_text", "")))
    passed_all = False
    failed_gates: list[str] = []
    if profile_payload_ok:
        gates = profile_payload.get("gates", {})
        if isinstance(gates, dict):
            passed_all = bool(gates.get("passed_all", False))
            failed_gates = [str(x) for x in gates.get("failed", [])]
    else:
        profile_ok = False
    profile_ok = profile_ok and passed_all
    ok = add_step(
        "profile",
        r_profile,
        profile_ok,
        notes={"passed_all": passed_all, "failed_gates": failed_gates, "json_parsed": profile_payload_ok},
    )
    if fail_fast and not ok:
        return _finalize(steps)

    runs_path = fixture_root / "runs_v1_valid.jsonl"
    manifest_path = fixture_root / "split_manifest_v1.json"
    fixture_exists = runs_path.exists() and manifest_path.exists()

    # Step 5: strict official validate on fixture
    if not fixture_exists:
        r_validate = {
            "cmd": "",
            "returncode": 1,
            "duration_ms": 0.0,
            "stdout_preview": [],
            "stderr_preview": [],
            "stdout_text": "",
        }
        ok = add_step(
            "validate_official_fixture",
            r_validate,
            False,
            notes={"fixture_exists": False, "runs_path": str(runs_path), "manifest_path": str(manifest_path)},
        )
        if fail_fast and not ok:
            return _finalize(steps)
    else:
        r_validate = _run_subprocess(
            [
                sys.executable,
                "-m",
                "gf01",
                "validate",
                "--runs",
                str(runs_path),
                "--manifest",
                str(manifest_path),
                "--official",
            ],
            cwd=root,
        )
        ok = add_step(
            "validate_official_fixture",
            r_validate,
            r_validate["returncode"] == 0,
            notes={"fixture_exists": True},
        )
        if fail_fast and not ok:
            return _finalize(steps)

    # Step 6: strict official report on fixture
    if not fixture_exists:
        r_report = {
            "cmd": "",
            "returncode": 1,
            "duration_ms": 0.0,
            "stdout_preview": [],
            "stderr_preview": [],
            "stdout_text": "",
        }
        add_step(
            "report_official_fixture",
            r_report,
            False,
            notes={"fixture_exists": False, "runs_path": str(runs_path), "manifest_path": str(manifest_path)},
        )
    else:
        r_report = _run_subprocess(
            [
                sys.executable,
                "-m",
                "gf01",
                "report",
                "--runs",
                str(runs_path),
                "--manifest",
                str(manifest_path),
                "--official",
            ],
            cwd=root,
        )
        add_step(
            "report_official_fixture",
            r_report,
            r_report["returncode"] == 0,
            notes={"fixture_exists": True},
        )

    return _finalize(steps)


def _finalize(steps: list[dict[str, Any]]) -> dict[str, Any]:
    failed_steps = [s["name"] for s in steps if not bool(s.get("passed", False))]
    return {
        "status": "ok" if not failed_steps else "error",
        "passed": len(failed_steps) == 0,
        "failed_steps": failed_steps,
        "steps": steps,
    }
