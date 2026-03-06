"""Playable episode command implementations."""

from __future__ import annotations

import argparse
import json

from ..baselines import make_agent
from ..generator import generate_instance
from ..io import load_instance_bundle, write_json
from ..meta import (
    ADAPTATION_POLICY_VERSION,
    ALLOWED_EVAL_TRACKS,
    RENDERER_POLICY_VERSION,
    renderer_profile_for_track,
)
from ..models import GeneratorConfig
from ..play import (
    EpisodeAborted,
    baseline_policy,
    human_policy,
    parse_action_script,
    run_episode,
    scripted_policy,
)
from .shared import (
    adaptation_policy_message,
    renderer_policy_message,
    track_tool_policy_message,
)


def cmd_play(args: argparse.Namespace) -> int:
    if args.instances:
        instances, _ = load_instance_bundle(args.instances)
        if args.instance_index < 0 or args.instance_index >= len(instances):
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "instance_index_out_of_range",
                        "instance_count": len(instances),
                        "instance_index": int(args.instance_index),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        instance = instances[args.instance_index]
    else:
        cfg = GeneratorConfig()
        instance, _ = generate_instance(seed=args.seed, cfg=cfg, split_id=args.split)

    if args.agent and args.script:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "mutually_exclusive_inputs",
                    "message": "choose either --agent or --script, not both",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    eval_track = str(args.eval_track).strip()
    renderer_track = str(args.renderer_track).strip()
    visual_backend = str(args.visual_backend).strip()
    renderer_profile_id = renderer_profile_for_track(renderer_track)
    tool_allowlist_id = str(args.tool_allowlist_id).strip()
    tool_log_hash = str(args.tool_log_hash).strip()
    adaptation_condition = str(args.adaptation_condition).strip()
    adaptation_budget_tokens = int(args.adaptation_budget_tokens)
    adaptation_data_scope = str(args.adaptation_data_scope).strip()
    adaptation_protocol_id = str(args.adaptation_protocol_id).strip()

    if eval_track not in ALLOWED_EVAL_TRACKS:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": f"unsupported eval_track {eval_track}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    renderer_msg = renderer_policy_message(
        renderer_track=renderer_track,
        renderer_profile_id=renderer_profile_id,
    )
    if renderer_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "renderer_policy_violation",
                    "message": renderer_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if renderer_track != "visual" and visual_backend != "text":
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "renderer_backend_policy_violation",
                    "message": (
                        "--visual-backend is only valid for --renderer-track visual; "
                        "use --visual-backend text for non-visual tracks"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    policy_msg = track_tool_policy_message(
        eval_track=eval_track,
        tool_allowlist_id=tool_allowlist_id,
        tool_log_hash=tool_log_hash,
    )
    if policy_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "track_policy_violation",
                    "message": policy_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    adaptation_msg = adaptation_policy_message(
        adaptation_condition=adaptation_condition,
        adaptation_budget_tokens=adaptation_budget_tokens,
        adaptation_data_scope=adaptation_data_scope,
        adaptation_protocol_id=adaptation_protocol_id,
    )
    if adaptation_msg is not None:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "adaptation_policy_violation",
                    "message": adaptation_msg,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    if args.agent:
        agent = make_agent(args.agent)
        agent_id = str(args.agent).strip().lower()
        if agent_id in {"tool", "bl-03", "bl-03-toolplanner"} and eval_track == "EVAL-CB":
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "track_policy_violation",
                        "message": "tool planner agent is not allowed in EVAL-CB",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        if agent_id in {"oracle", "bl-04", "bl-04-exactoracle"} and eval_track != "EVAL-OC":
            print(
                json.dumps(
                    {
                        "status": "error",
                        "error_type": "track_policy_violation",
                        "message": "exact oracle agent is restricted to EVAL-OC",
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
        policy = baseline_policy(agent, instance)
        actor = agent.name
    elif args.script:
        actions_by_t = parse_action_script(args.script)
        policy = scripted_policy(actions_by_t)
        actor = "scripted-policy"
    else:
        policy = human_policy(
            renderer_track=renderer_track,
            visual_backend=visual_backend,
        )
        actor = "human-interactive"

    try:
        result = run_episode(instance, policy, renderer_track=renderer_track)
    except (ValueError, EpisodeAborted) as exc:
        error_type = (
            "episode_aborted" if isinstance(exc, EpisodeAborted) else "episode_execution_error"
        )
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": error_type,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2

    payload = {
        "status": "ok",
        "actor": actor,
        "run_contract": {
            "eval_track": eval_track,
            "renderer_track": renderer_track,
            "renderer_policy_version": RENDERER_POLICY_VERSION,
            "renderer_profile_id": renderer_profile_id,
            "visual_backend": visual_backend,
            "tool_allowlist_id": tool_allowlist_id or "none",
            "tool_log_hash": tool_log_hash,
            "play_protocol": "commit_only",
            "scored_commit_episode": True,
            "adaptation_policy_version": ADAPTATION_POLICY_VERSION,
            "adaptation_condition": adaptation_condition,
            "adaptation_budget_tokens": adaptation_budget_tokens,
            "adaptation_data_scope": adaptation_data_scope,
            "adaptation_protocol_id": adaptation_protocol_id or "none",
        },
        "instance": instance.to_canonical_dict(),
        "episode": result,
    }
    if args.out:
        write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
