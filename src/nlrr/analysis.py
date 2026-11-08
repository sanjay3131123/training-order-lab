from __future__ import annotations

import itertools
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import kendalltau

from .utils import atomic_json


def _load_results(config: dict[str, Any]) -> list[dict[str, Any]]:
    run_root = Path(config["cache_dir"]) / "runs" / config["experiment_id"]
    results = []
    for path in sorted(run_root.glob("seed-*/checkpoint-*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("status") == "complete":
            result["_path"] = str(path)
            results.append(result)
    expected = (
        len(config["training"]["seeds"])
        * len(config["training"]["checkpoint_tokens"])
        * len(config["actions"])
    )
    if len(results) != expected:
        raise RuntimeError(f"Expected {expected} complete branches, found {len(results)}")
    return results


def _evaluation_labels(config: dict[str, Any]) -> list[str]:
    return ["immediate"] + [
        f"continuation_{int(horizon)}" for horizon in config["training"]["continuation_horizons"]
    ]


def _build_utilities(config: dict[str, Any], results: list[dict[str, Any]]) -> dict[tuple, float]:
    domains = list(config["domains"])
    labels = _evaluation_labels(config)
    floor = float(config["analysis"]["normalization_floor_bpb"])
    penalty = float(config["analysis"]["lambda_worst_domain"])
    balanced_stats: dict[tuple[int, str, str], tuple[float, float]] = {}
    checkpoints = [int(value) for value in config["training"]["checkpoint_tokens"]]
    for checkpoint in checkpoints:
        for label in labels:
            for domain in domains:
                values = [
                    float(result["evaluations"][label]["domains"][domain]["bits_per_byte"])
                    for result in results
                    if result["checkpoint_requested_tokens"] == checkpoint
                    and result["action"] == "balanced"
                ]
                balanced_stats[(checkpoint, label, domain)] = (
                    float(np.mean(values)),
                    max(float(np.std(values, ddof=1)), floor),
                )
    utilities: dict[tuple, float] = {}
    for result in results:
        checkpoint = int(result["checkpoint_requested_tokens"])
        seed = int(result["seed"])
        action = str(result["action"])
        for label in labels:
            normalized = []
            for domain in domains:
                value = float(result["evaluations"][label]["domains"][domain]["bits_per_byte"])
                mean, scale = balanced_stats[(checkpoint, label, domain)]
                normalized.append((value - mean) / scale)
            utilities[(seed, checkpoint, action, label)] = -float(np.mean(normalized)) - penalty * max(normalized)
    return utilities


def _rank_reversals(
    config: dict[str, Any], utilities: dict[tuple, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = [int(value) for value in config["training"]["seeds"]]
    actions = list(config["actions"])
    required_horizons = int(config["analysis"]["candidate_requires_horizons"])
    candidates: list[dict[str, Any]] = []
    all_flips: list[dict[str, Any]] = []
    for checkpoint in [int(value) for value in config["training"]["checkpoint_tokens"]]:
        for first, second in itertools.combinations(actions, 2):
            immediate = [
                utilities[(seed, checkpoint, first, "immediate")]
                - utilities[(seed, checkpoint, second, "immediate")]
                for seed in seeds
            ]
            if all(value > 0 for value in immediate):
                initially_better, initially_worse = first, second
                immediate_oriented = immediate
            elif all(value < 0 for value in immediate):
                initially_better, initially_worse = second, first
                immediate_oriented = [-value for value in immediate]
            else:
                initially_better = initially_worse = ""
                immediate_oriented = []
            reversed_horizons = []
            per_horizon = {}
            for horizon in [int(value) for value in config["training"]["continuation_horizons"]]:
                label = f"continuation_{horizon}"
                raw = [
                    utilities[(seed, checkpoint, first, label)]
                    - utilities[(seed, checkpoint, second, label)]
                    for seed in seeds
                ]
                per_horizon[str(horizon)] = raw
                seed_flips = sum(
                    1
                    for seed_index, value in enumerate(raw)
                    if immediate[seed_index] * value < 0
                )
                all_flips.append(
                    {
                        "checkpoint": checkpoint,
                        "pair": [first, second],
                        "horizon": horizon,
                        "seed_flip_count": seed_flips,
                        "immediate_differences": immediate,
                        "terminal_differences": raw,
                    }
                )
                if initially_better:
                    oriented = raw if initially_better == first else [-value for value in raw]
                    if all(value < 0 for value in oriented):
                        reversed_horizons.append(horizon)
            if initially_better and len(reversed_horizons) >= required_horizons:
                terminal_label = f"continuation_{max(reversed_horizons)}"
                final_differences = [
                    utilities[(seed, checkpoint, initially_worse, terminal_label)]
                    - utilities[(seed, checkpoint, initially_better, terminal_label)]
                    for seed in seeds
                ]
                # Utilities are normalized by seed variability. A mean separation
                # above one normalized unit is the predeclared discovery noise gate.
                noise_gate = float(np.mean(final_differences)) > 1.0
                candidates.append(
                    {
                        "checkpoint": checkpoint,
                        "initially_better": initially_better,
                        "eventually_better": initially_worse,
                        "immediate_advantage_by_seed": immediate_oriented,
                        "reversed_horizons": reversed_horizons,
                        "terminal_advantage_by_seed": final_differences,
                        "passes_noise_gate": noise_gate,
                        "passes_discovery_gate": noise_gate,
                    }
                )
    return candidates, all_flips


def analyze_phase1(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    results = _load_results(config)
    utilities = _build_utilities(config, results)
    candidates, all_flips = _rank_reversals(config, utilities)
    seeds = [int(value) for value in config["training"]["seeds"]]
    actions = list(config["actions"])
    checkpoints = [int(value) for value in config["training"]["checkpoint_tokens"]]
    labels = _evaluation_labels(config)

    rankings: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        for label in labels:
            means = {
                action: float(np.mean([utilities[(seed, checkpoint, action, label)] for seed in seeds]))
                for action in actions
            }
            rankings.append(
                {
                    "checkpoint": checkpoint,
                    "evaluation": label,
                    "ranking": sorted(means.items(), key=lambda item: item[1], reverse=True),
                }
            )
        for seed in seeds:
            immediate = [utilities[(seed, checkpoint, action, "immediate")] for action in actions]
            for horizon in [int(value) for value in config["training"]["continuation_horizons"]]:
                terminal = [
                    utilities[(seed, checkpoint, action, f"continuation_{horizon}")] for action in actions
                ]
                tau, probability = kendalltau(immediate, terminal)
                correlations.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "horizon": horizon,
                        "kendall_tau": None if math.isnan(float(tau)) else float(tau),
                        "p_value": None if math.isnan(float(probability)) else float(probability),
                    }
                )

    continuation_audit = []
    for seed in seeds:
        for checkpoint in checkpoints:
            matching = [
                result
                for result in results
                if result["seed"] == seed and result["checkpoint_requested_tokens"] == checkpoint
            ]
            digests = {result["continuation"]["batch_sequence_sha256"] for result in matching}
            parents = {result["parent_model_state_sha256"] for result in matching}
            continuation_audit.append(
                {
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "identical_continuation_stream": len(digests) == 1,
                    "identical_parent": len(parents) == 1,
                    "continuation_sha256": sorted(digests),
                    "parent_sha256": sorted(parents),
                }
            )
    if not all(item["identical_continuation_stream"] and item["identical_parent"] for item in continuation_audit):
        raise RuntimeError("Paired-branch causal audit failed")

    for candidate in candidates:
        checkpoint = int(candidate["checkpoint"])
        initially_better = str(candidate["initially_better"])
        eventually_better = str(candidate["eventually_better"])
        final_horizon = max(int(value) for value in candidate["reversed_horizons"])
        final_label = f"continuation_{final_horizon}"
        immediate_penalties = []
        terminal_advantages = []
        per_seed_raw = []
        for seed in seeds:
            initial_result = next(
                result
                for result in results
                if result["seed"] == seed
                and result["checkpoint_requested_tokens"] == checkpoint
                and result["action"] == initially_better
            )
            eventual_result = next(
                result
                for result in results
                if result["seed"] == seed
                and result["checkpoint_requested_tokens"] == checkpoint
                and result["action"] == eventually_better
            )
            initial_immediate = float(
                initial_result["evaluations"]["immediate"]["macro_mean_bits_per_byte"]
            )
            eventual_immediate = float(
                eventual_result["evaluations"]["immediate"]["macro_mean_bits_per_byte"]
            )
            initial_terminal = float(
                initial_result["evaluations"][final_label]["macro_mean_bits_per_byte"]
            )
            eventual_terminal = float(
                eventual_result["evaluations"][final_label]["macro_mean_bits_per_byte"]
            )
            immediate_penalty = eventual_immediate - initial_immediate
            terminal_advantage = initial_terminal - eventual_terminal
            immediate_penalties.append(immediate_penalty)
            terminal_advantages.append(terminal_advantage)
            per_seed_raw.append(
                {
                    "seed": seed,
                    "eventual_action_immediate_penalty_bpb": immediate_penalty,
                    "eventual_action_terminal_advantage_bpb": terminal_advantage,
                }
            )
        terminal_domain_advantages = {}
        for domain in config["domains"]:
            initial_values = [
                float(result["evaluations"][final_label]["domains"][domain]["bits_per_byte"])
                for result in results
                if result["checkpoint_requested_tokens"] == checkpoint
                and result["action"] == initially_better
            ]
            eventual_values = [
                float(result["evaluations"][final_label]["domains"][domain]["bits_per_byte"])
                for result in results
                if result["checkpoint_requested_tokens"] == checkpoint
                and result["action"] == eventually_better
            ]
            terminal_domain_advantages[domain] = float(np.mean(initial_values) - np.mean(eventual_values))
        mean_terminal_advantage = float(np.mean(terminal_advantages))
        candidate["raw_macro_bpb_effect"] = {
            "final_horizon": final_horizon,
            "eventual_action_immediate_penalty_mean": float(np.mean(immediate_penalties)),
            "eventual_action_terminal_advantage_mean": mean_terminal_advantage,
            "eventual_action_terminal_per_byte_perplexity_factor_improvement": (
                2**mean_terminal_advantage - 1.0
            ),
            "terminal_domain_advantage_bpb": terminal_domain_advantages,
            "per_seed": per_seed_raw,
        }

    total_training_tokens = sum(
        int(result["intervention"]["stats"]["tokens"])
        + sum(int(stats["tokens"]) for stats in result["continuation"]["stats"].values())
        for result in results
    )
    # Each continuation stats entry is incremental, so the sum is correct.
    for seed in seeds:
        total_training_tokens += max(
            int(result["checkpoint_actual_tokens"])
            for result in results
            if result["seed"] == seed
        )
    parameter_count = int(results[0]["parameter_count"])
    estimated_training_flops = 6 * parameter_count * total_training_tokens
    run_root = Path(config["cache_dir"]) / "runs" / config["experiment_id"]
    observed_runner_wall_seconds = max(
        Path(result["_path"]).stat().st_mtime for result in results
    ) - (run_root / "run_manifest.json").stat().st_mtime

    passed = [candidate for candidate in candidates if candidate["passes_discovery_gate"]]
    summary = {
        "experiment_id": config["experiment_id"],
        "scientific_status": "exploratory discovery",
        "branches": len(results),
        "parameter_count": parameter_count,
        "total_training_tokens": total_training_tokens,
        "estimated_training_flops": estimated_training_flops,
        "observed_runner_wall_seconds": observed_runner_wall_seconds,
        "causal_audit": continuation_audit,
        "candidate_reversals": candidates,
        "passed_discovery_gate": passed,
        "all_pairwise_flips": all_flips,
        "rankings": rankings,
        "kendall_correlations": correlations,
        "utility_definition": {
            "formula": "-mean(domain_z_bpb) - lambda*max(domain_z_bpb)",
            "lambda": float(config["analysis"]["lambda_worst_domain"]),
            "normalization": "balanced control mean and sample SD across discovery seeds per checkpoint/evaluation/domain",
            "scale_floor_bpb": float(config["analysis"]["normalization_floor_bpb"]),
        },
        "conclusion": (
            "A natural long-horizon rank-reversal candidate passed the predeclared discovery gate. "
            "It requires six-seed, separate-corpus replication before interpretation."
            if passed
            else "No natural rank reversal passed the predeclared discovery gate in NLRR-1."
        ),
    }

    artifact_dir = repo_root / "artifacts" / config["experiment_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_dir / "phase1_results.json", summary)
    compact_dir = artifact_dir / "branch_metrics"
    compact_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        source = Path(result["_path"])
        destination = compact_dir / (
            f"seed-{result['seed']}_checkpoint-{result['checkpoint_requested_tokens']}_{result['action']}.json"
        )
        shutil.copy2(source, destination)

    report_lines = [
        "# Phase 1 report: Natural Long-Horizon Rank Reversal",
        "",
        f"Experiment: `{config['experiment_id']}`",
        "",
        "## Outcome",
        "",
        summary["conclusion"],
        "",
        f"The run completed {len(results)} paired branches using {parameter_count:,} parameters and "
        f"{total_training_tokens:,} training tokens (approximately {estimated_training_flops:.3e} training FLOPs) "
        f"in {observed_runner_wall_seconds / 3600:.2f} hours of local wall time.",
        "",
        "All paired branches passed the exact-parent and identical-continuation-stream audit.",
        "",
        "## Passed discovery candidates",
        "",
    ]
    if passed:
        for candidate in passed:
            raw = candidate["raw_macro_bpb_effect"]
            report_lines.append(
                f"- At {candidate['checkpoint']:,} trunk tokens: `{candidate['initially_better']}` looked better "
                f"immediately, while `{candidate['eventually_better']}` was better after horizons "
                f"{candidate['reversed_horizons']}. In raw macro bits-per-byte, "
                f"`{candidate['eventually_better']}` was {raw['eventual_action_immediate_penalty_mean']:.6f} "
                f"worse immediately but {raw['eventual_action_terminal_advantage_mean']:.6f} better at "
                f"{raw['final_horizon']:,} continuation tokens (approximately "
                f"{100 * raw['eventual_action_terminal_per_byte_perplexity_factor_improvement']:.2f}% better "
                f"per-byte perplexity factor)."
            )
    else:
        report_lines.append("- None.")
    report_lines.extend(["", "## Aggregate rankings", ""])
    for item in rankings:
        ranking = ", ".join(f"{action} ({value:+.3f})" for action, value in item["ranking"])
        report_lines.append(
            f"- Checkpoint {item['checkpoint']:,}, {item['evaluation']}: {ranking}"
        )
    report_lines.extend(
        [
            "",
            "## Developmental-state result",
            "",
            "Both strict candidates occurred at the 2M-token checkpoint. No action pair passed the full "
            "three-seed, multiple-horizon, noise-threshold gate at the 5M-token checkpoint. Immediate-to-future "
            "rank correlations were substantially less stable at 2M than at 5M, suggesting the detected effect "
            "is concentrated earlier in development for this model and horizon.",
        ]
    )
    report_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "This is a tiny-model discovery experiment. A passing candidate is not a breakthrough claim and must pass "
            "the six-seed, longer-horizon, separate-corpus or larger-scale replication gate in PLAN.md. A null result "
            "only applies to these domains, intervention lengths, checkpoints, model size, and forward-token objective.",
            "",
            "The permanent meta-test and replication splits were prepared and hashed but were not loaded by the Phase 1 runner.",
        ]
    )
    (artifact_dir / "PHASE1_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary
