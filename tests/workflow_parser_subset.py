"""
Dependency-free parser for the small workflow-YAML subset used in CI tests.

Supported subset (GitHub Actions workflow shape):
- top-level ``jobs:``
- job objects keyed at 2-space indentation
- job scalar fields (e.g., ``runs-on``, ``timeout-minutes``)
- job ``needs`` in inline or list form
- job nested maps (e.g., ``env``)
- ``steps`` list with step scalar fields
- step nested maps (e.g., ``env``, ``with``)
- step ``run`` block scalars (``|`` and ``>``) captured as raw text

This parser is intentionally narrow. It validates indentation/layout patterns
used by the supported subset and raises ``WorkflowSubsetParseError`` on shapes
it does not understand, so test failures are explicit rather than silently
mis-parsed.
"""

from __future__ import annotations

from typing import TypeAlias


WorkflowStep: TypeAlias = dict[str, object]
WorkflowJob: TypeAlias = dict[str, object]
WorkflowJobs: TypeAlias = dict[str, WorkflowJob]


JOB_NAME_INDENT = 2
JOB_FIELD_INDENT = 4
JOB_CHILD_INDENT = 6
STEP_ITEM_INDENT = 6
STEP_FIELD_INDENT = 8
STEP_CHILD_INDENT = 10


class WorkflowSubsetParseError(ValueError):
    """Raised when the workflow text falls outside the supported subset."""


def parse_workflow_jobs_subset(text: str) -> WorkflowJobs:
    """Parse the subset of workflow YAML needed by CI-policy regression tests."""

    def fail(i: int, message: str, raw: str) -> "WorkflowSubsetParseError":
        return WorkflowSubsetParseError(
            f"line {i + 1}: {message}; got: {raw!r}"
        )

    def indent_of(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    def parse_value(raw: str) -> object:
        value = raw.strip()
        if not value:
            return ""
        if value.startswith("[") and value.endswith("]"):
            body = value[1:-1].strip()
            if not body:
                return []
            return [item.strip().strip("'\"") for item in body.split(",")]
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            return value[1:-1]
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return value

    jobs: WorkflowJobs = {}
    lines = text.splitlines()

    in_jobs = False
    current_job: WorkflowJob | None = None
    current_step: WorkflowStep | None = None
    current_job_nested_map_key: str | None = None
    current_nested_map_key: str | None = None
    collect_job_needs = False
    block_target_map: dict[str, object] | None = None
    block_target_key: str | None = None
    block_indent: int | None = None
    block_lines: list[str] = []

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        indent = indent_of(raw)

        if block_target_key is not None and block_target_map is not None:
            if stripped == "":
                block_lines.append("")
                i += 1
                continue
            assert block_indent is not None
            if indent >= block_indent:
                block_lines.append(raw[block_indent:])
                i += 1
                continue
            block_target_map[block_target_key] = "\n".join(block_lines)
            block_target_map = None
            block_target_key = None
            block_indent = None
            block_lines = []
            continue  # Reprocess current line outside block mode.

        if stripped == "" or stripped.startswith("#"):
            i += 1
            continue

        if "\t" in raw[:indent]:
            raise fail(i, "tabs are not supported for indentation", raw)
        if indent % 2 != 0:
            raise fail(i, "indentation must use multiples of 2 spaces", raw)

        if not in_jobs:
            if indent == 0 and stripped == "jobs:":
                in_jobs = True
            i += 1
            continue

        # Stop parsing once jobs section ends (another top-level key begins).
        if indent == 0 and stripped.endswith(":") and stripped != "jobs:":
            break

        if current_nested_map_key is not None and indent < STEP_CHILD_INDENT:
            current_nested_map_key = None
        if current_job_nested_map_key is not None and indent < JOB_CHILD_INDENT:
            current_job_nested_map_key = None

        if (
            indent == JOB_NAME_INDENT
            and stripped.endswith(":")
            and not stripped.startswith("- ")
        ):
            job_name = stripped[:-1]
            current_job = {"steps": [], "needs": []}
            jobs[job_name] = current_job
            current_step = None
            current_job_nested_map_key = None
            current_nested_map_key = None
            collect_job_needs = False
            i += 1
            continue

        if current_job is None:
            raise fail(i, "encountered job content before a job key", raw)

        if collect_job_needs:
            if indent == JOB_CHILD_INDENT and stripped.startswith("- "):
                needs = current_job.setdefault("needs", [])
                if not isinstance(needs, list):
                    raise fail(i, "job needs container is not a list", raw)
                needs.append(stripped[2:].strip())
                i += 1
                continue
            if indent >= JOB_CHILD_INDENT:
                raise fail(i, "expected list item under job needs", raw)
            collect_job_needs = False

        if (
            current_job_nested_map_key is not None
            and indent >= JOB_CHILD_INDENT
        ):
            if ":" not in stripped or stripped.startswith("- "):
                raise fail(i, "expected key: value under job nested map", raw)
            key, value_part = stripped.split(":", 1)
            nested = current_job.setdefault(current_job_nested_map_key, {})
            if not isinstance(nested, dict):
                raise fail(i, "job nested map container is not a dict", raw)
            nested_key = key.strip()
            value = value_part.strip()
            if value in {"|", ">"}:
                block_target_map = nested
                block_target_key = nested_key
                block_indent = JOB_CHILD_INDENT + 2
                block_lines = []
                i += 1
                continue
            nested[nested_key] = parse_value(value)
            i += 1
            continue

        if indent == JOB_FIELD_INDENT and ":" in stripped and not stripped.startswith("- "):
            key, value_part = stripped.split(":", 1)
            key = key.strip()
            value_part = value_part.strip()
            if key == "needs":
                if value_part:
                    parsed_needs = parse_value(value_part)
                    if isinstance(parsed_needs, list):
                        current_job["needs"] = [str(x) for x in parsed_needs]
                    else:
                        current_job["needs"] = [str(parsed_needs)]
                else:
                    current_job["needs"] = []
                    collect_job_needs = True
                i += 1
                continue
            if key == "steps":
                i += 1
                continue
            if value_part == "":
                current_job[key] = {}
                current_job_nested_map_key = key
                i += 1
                continue
            current_job[key] = parse_value(value_part)
            i += 1
            continue

        if indent == STEP_ITEM_INDENT and stripped.startswith("- "):
            current_step = {}
            current_nested_map_key = None
            item_body = stripped[2:].strip()
            if item_body:
                if ":" not in item_body:
                    raise fail(i, "unsupported step list item shape", raw)
                key, value_part = item_body.split(":", 1)
                current_step[key.strip()] = parse_value(value_part.strip())
            steps = current_job.setdefault("steps", [])
            if not isinstance(steps, list):
                raise fail(i, "job steps container is not a list", raw)
            steps.append(current_step)
            i += 1
            continue

        if current_step is None:
            raise fail(i, "encountered step content before a step item", raw)

        if current_nested_map_key is not None and indent >= STEP_CHILD_INDENT:
            if ":" not in stripped or stripped.startswith("- "):
                raise fail(i, "expected key: value under step nested map", raw)
            key, value_part = stripped.split(":", 1)
            nested = current_step.setdefault(current_nested_map_key, {})
            if not isinstance(nested, dict):
                raise fail(i, "step nested map container is not a dict", raw)
            nested_key = key.strip()
            value = value_part.strip()
            if value in {"|", ">"}:
                block_target_map = nested
                block_target_key = nested_key
                block_indent = STEP_CHILD_INDENT + 2
                block_lines = []
                i += 1
                continue
            nested[nested_key] = parse_value(value)
            i += 1
            continue

        if indent == STEP_FIELD_INDENT and ":" in stripped:
            key, value_part = stripped.split(":", 1)
            key = key.strip()
            value_part = value_part.strip()

            if value_part in {"|", ">"}:
                block_target_map = current_step
                block_target_key = key
                block_indent = STEP_CHILD_INDENT
                block_lines = []
                i += 1
                continue

            if value_part == "":
                current_step[key] = {}
                current_nested_map_key = key
                i += 1
                continue

            current_step[key] = parse_value(value_part)
            current_nested_map_key = None
            i += 1
            continue

        raise fail(i, "unsupported indentation or syntax in jobs subset", raw)

    if block_target_key is not None and block_target_map is not None:
        block_target_map[block_target_key] = "\n".join(block_lines)

    return jobs


def job_steps(job: WorkflowJob) -> list[WorkflowStep]:
    """Return the parsed steps list with explicit structure validation."""
    steps = job.get("steps", [])
    if not isinstance(steps, list):
        raise WorkflowSubsetParseError("job.steps is not a list")
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise WorkflowSubsetParseError(f"job.steps[{idx}] is not a mapping")
    return steps  # type: ignore[return-value]


def job_needs(job: WorkflowJob) -> list[str]:
    """Return the parsed needs list with explicit structure validation."""
    needs = job.get("needs", [])
    if not isinstance(needs, list):
        raise WorkflowSubsetParseError("job.needs is not a list")
    for idx, item in enumerate(needs):
        if not isinstance(item, str):
            raise WorkflowSubsetParseError(f"job.needs[{idx}] is not a string")
    return needs  # type: ignore[return-value]


def step_env(step: WorkflowStep) -> dict[str, object]:
    """Return step env mapping or an empty mapping, validating structure."""
    env = step.get("env", {})
    if not isinstance(env, dict):
        raise WorkflowSubsetParseError("step.env is not a mapping")
    return env


def steps_by_name(job: WorkflowJob) -> dict[str, WorkflowStep]:
    """Index steps by ``name`` for tests that assert specific workflow steps."""
    result: dict[str, WorkflowStep] = {}
    for step in job_steps(job):
        name = step.get("name")
        if isinstance(name, str):
            result[name] = step
    return result
