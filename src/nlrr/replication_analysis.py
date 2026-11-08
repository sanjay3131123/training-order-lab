from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

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
        raise RuntimeError(f"Expected {expected} results, found {len(results)}")
    cells = {
        (int(result["seed"]), int(result["checkpoint_requested_tokens"]), str(result["action"]))
        for result in results
    }
    if len(cells) != expected:
        raise RuntimeError("Replication grid contains duplicate or missing cells")
    return results


def _find(results: list[dict[str, Any]], seed: int, checkpoint: int, action: str) -> dict[str, Any]:
    return next(
        result
        for result in results
        if int(result["seed"]) == seed
        and int(result["checkpoint_requested_tokens"]) == checkpoint
        and result["action"] == action
    )


def analyze_replication_screen(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    results = _load_results(config)
    seeds = [int(value) for value in config["training"]["seeds"]]
    screen = config["screen"]
    checkpoint = int(screen["primary_checkpoint"])
    primary_horizon = int(screen["primary_horizon"])
    supporting_horizon = int(screen["supporting_horizon"])
    primary_label = f"continuation_{primary_horizon}"
    supporting_label = f"continuation_{supporting_horizon}"

    audits = []
    for seed in seeds:
        paired = [_find(results, seed, checkpoint, action) for action in config["actions"]]
        parents = {result["parent_model_state_sha256"] for result in paired}
        continuations = {result["continuation"]["batch_sequence_sha256"] for result in paired}
        audits.append(
            {
                "seed": seed,
                "identical_parent": len(parents) == 1,
                "identical_continuation_stream": len(continuations) == 1,
                "parent_sha256": sorted(parents),
                "continuation_sha256": sorted(continuations),
            }
        )
    causal_audit_pass = all(
        row["identical_parent"] and row["identical_continuation_stream"] for row in audits
    )
    if not causal_audit_pass:
        raise RuntimeError("Replication causal audit failed")

    seed_rows = []
    primary_advantages = []
    supporting_advantages = []
    immediate_penalties = []
    for seed in seeds:
        science = _find(results, seed, checkpoint, "science_math")
        balanced = _find(results, seed, checkpoint, "balanced")
        science_immediate = float(science["evaluations"]["immediate"]["macro_mean_bits_per_byte"])
        balanced_immediate = float(balanced["evaluations"]["immediate"]["macro_mean_bits_per_byte"])
        science_supporting = float(science["evaluations"][supporting_label]["macro_mean_bits_per_byte"])
        balanced_supporting = float(balanced["evaluations"][supporting_label]["macro_mean_bits_per_byte"])
        science_primary = float(science["evaluations"][primary_label]["macro_mean_bits_per_byte"])
        balanced_primary = float(balanced["evaluations"][primary_label]["macro_mean_bits_per_byte"])
        immediate_penalty = science_immediate - balanced_immediate
        supporting_advantage = balanced_supporting - science_supporting
        primary_advantage = balanced_primary - science_primary
        immediate_penalties.append(immediate_penalty)
        supporting_advantages.append(supporting_advantage)
        primary_advantages.append(primary_advantage)
        seed_rows.append(
            {
                "seed": seed,
                "science_immediate_penalty_bpb": immediate_penalty,
                "science_advantage_at_supporting_horizon_bpb": supporting_advantage,
                "science_advantage_at_primary_horizon_bpb": primary_advantage,
                "rank_reversal_at_both_horizons": (
                    immediate_penalty > 0 and supporting_advantage > 0 and primary_advantage > 0
                ),
            }
        )

    domain_advantages = {}
    for domain in config["domains"]:
        values = []
        for seed in seeds:
            science = _find(results, seed, checkpoint, "science_math")
            balanced = _find(results, seed, checkpoint, "balanced")
            values.append(
                float(balanced["evaluations"][primary_label]["domains"][domain]["bits_per_byte"])
                - float(science["evaluations"][primary_label]["domains"][domain]["bits_per_byte"])
            )
        domain_advantages[domain] = {"mean_bpb": float(np.mean(values)), "by_seed": values}

    mean_primary = float(np.mean(primary_advantages))
    standard_error = float(np.std(primary_advantages, ddof=1) / math.sqrt(len(primary_advantages)))
    critical = float(t.ppf(0.975, df=len(primary_advantages) - 1))
    confidence_interval = [mean_primary - critical * standard_error, mean_primary + critical * standard_error]

    checks = {
        "causal_audit": causal_audit_pass,
        "all_seeds_reverse": all(row["rank_reversal_at_both_horizons"] for row in seed_rows),
        "all_seeds_better_at_supporting_horizon": all(value > 0 for value in supporting_advantages),
        "all_seeds_better_at_primary_horizon": all(value > 0 for value in primary_advantages),
        "minimum_mean_terminal_advantage": mean_primary
        >= float(screen["minimum_terminal_advantage_bpb"]),
        "no_large_domain_regression": min(
            value["mean_bpb"] for value in domain_advantages.values()
        )
        >= -float(screen["maximum_domain_regression_bpb"]),
    }
    status = "pass" if all(checks.values()) else "fail"

    tokens_per_step = int(results[0]["tokens_per_step"])
    total_training_tokens = sum(
        int(result["intervention"]["stats"]["tokens"])
        + sum(int(stats["tokens"]) for stats in result["continuation"]["stats"].values())
        for result in results
    )
    for seed in seeds:
        total_training_tokens += max(
            int(result["checkpoint_actual_tokens"])
            for result in results
            if int(result["seed"]) == seed
        )
    run_root = Path(config["cache_dir"]) / "runs" / config["experiment_id"]
    wall_seconds = max(Path(result["_path"]).stat().st_mtime for result in results) - (
        run_root / "run_manifest.json"
    ).stat().st_mtime
    parameter_count = int(results[0]["parameter_count"])

    summary = {
        "experiment_id": config["experiment_id"],
        "scientific_status": "rapid falsification screen; not final replication",
        "screen_status": status,
        "decision": (
            "Advance to the frozen six-seed R2 confirmation."
            if status == "pass"
            else "Do not advance to R2; downgrade the long-horizon data-selection arm."
        ),
        "checks": checks,
        "seed_results": seed_rows,
        "mean_science_immediate_penalty_bpb": float(np.mean(immediate_penalties)),
        "mean_science_supporting_advantage_bpb": float(np.mean(supporting_advantages)),
        "mean_science_primary_advantage_bpb": mean_primary,
        "primary_advantage_95_percent_t_interval": confidence_interval,
        "primary_per_byte_perplexity_factor_improvement": 2**mean_primary - 1.0,
        "domain_advantages_at_primary_horizon": domain_advantages,
        "causal_audit": audits,
        "parameter_count": parameter_count,
        "tokens_per_step": tokens_per_step,
        "total_training_tokens": total_training_tokens,
        "estimated_training_flops": 6 * parameter_count * total_training_tokens,
        "observed_runner_wall_seconds": wall_seconds,
        "sealed_splits_used": {"meta_test": False, "replication": False},
    }

    artifact_dir = repo_root / "artifacts" / config["experiment_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_dir / "replication_screen_results.json", summary)
    compact_dir = artifact_dir / "branch_metrics"
    compact_dir.mkdir(exist_ok=True)
    for result in results:
        destination = compact_dir / f"seed-{result['seed']}_{result['action']}.json"
        shutil.copy2(result["_path"], destination)

    lines = [
        "# NLRR-1 rapid replication screen",
        "",
        f"Status: **{status.upper()}**",
        "",
        summary["decision"],
        "",
        "## Primary result",
        "",
        f"Science/math's mean immediate penalty was {summary['mean_science_immediate_penalty_bpb']:.6f} bits/byte. "
        f"Its mean advantage after {primary_horizon:,} identical continuation tokens was "
        f"{mean_primary:.6f} bits/byte, corresponding to approximately "
        f"{100 * summary['primary_per_byte_perplexity_factor_improvement']:.2f}% better per-byte perplexity factor.",
        "",
        f"Descriptive paired 95% t interval: [{confidence_interval[0]:.6f}, {confidence_interval[1]:.6f}] bits/byte. "
        "This interval is reported but is not the predeclared R1 decision rule.",
        "",
        "## Per-seed results",
        "",
        "| Seed | Immediate penalty | Advantage at 1M | Advantage at 2M | Reversed |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['science_immediate_penalty_bpb']:+.6f} | "
            f"{row['science_advantage_at_supporting_horizon_bpb']:+.6f} | "
            f"{row['science_advantage_at_primary_horizon_bpb']:+.6f} | "
            f"{'yes' if row['rank_reversal_at_both_horizons'] else 'no'} |"
        )
    lines.extend(["", "## Gate checks", ""])
    for name, value in checks.items():
        lines.append(f"- {name}: {'pass' if value else 'fail'}")
    lines.extend(["", "## Held-out domain effects at 2M", ""])
    for domain, value in domain_advantages.items():
        lines.append(f"- {domain}: {value['mean_bpb']:+.6f} bits/byte")
    lines.extend(
        [
            "",
            "Positive values favor science/math. The meta-test and final-replication splits remained sealed.",
            "",
            "## Scope",
            "",
            f"The screen used {parameter_count:,} parameters and {total_training_tokens:,} training tokens in "
            f"{wall_seconds / 3600:.2f} local hours. A pass permits R2; it does not establish transfer or an "
            "architectural breakthrough.",
        ]
    )
    (artifact_dir / "REPLICATION_SCREEN_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return summary
