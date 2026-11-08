"""Verify published numerical claims offline using only the standard library."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path = ROOT) -> dict:
    all_seeds: set[int] = set()
    total_tokens = total_branches = 0
    screen_records = []
    for config_name, summary_name in (
        ("phase1", "phase1_results"),
        ("replication_screen", "replication_screen_results"),
    ):
        config = read_json(root / "configs" / f"{config_name}.json")
        directory = root / "artifacts" / config["experiment_id"]
        summary = read_json(directory / f"{summary_name}.json")
        records = [read_json(path) for path in sorted((directory / "branch_metrics").glob("*.json"))]
        expected = {
            (seed, checkpoint, action)
            for seed in config["training"]["seeds"]
            for checkpoint in config["training"]["checkpoint_tokens"]
            for action in config["actions"]
        }
        actual = {(r["seed"], r["checkpoint_requested_tokens"], r["action"]) for r in records}
        require(actual == expected and len(records) == len(expected), "Branch grid mismatch")
        grouped: dict[tuple, list] = defaultdict(list)
        trunks: dict[int, int] = {}
        tokens = 0
        for record in records:
            require(record["status"] == "complete", "Incomplete branch")
            require(record["parameter_count"] == summary["parameter_count"] == 2525600, "Model size mismatch")
            require(record["experiment_id"] == config["experiment_id"], "Experiment mismatch")
            seed = record["seed"]
            all_seeds.add(seed)
            trunks[seed] = max(trunks.get(seed, 0), record["checkpoint_actual_tokens"])
            tokens += record["intervention"]["stats"]["tokens"]
            tokens += sum(item["tokens"] for item in record["continuation"]["stats"].values())
            grouped[(seed, record["checkpoint_requested_tokens"])].append(record)
            for evaluation in record["evaluations"].values():
                domain_mean = mean(value["bits_per_byte"] for value in evaluation["domains"].values())
                require(math.isclose(domain_mean, evaluation["macro_mean_bits_per_byte"], abs_tol=1e-12), "Domain mean mismatch")
        for group in grouped.values():
            require(len({r["parent_model_state_sha256"] for r in group}) == 1, "Parent mismatch")
            require(len({r["continuation"]["batch_sequence_sha256"] for r in group}) == 1, "Continuation mismatch")
        tokens += sum(trunks.values())
        require(tokens == summary["total_training_tokens"], "Token accounting mismatch")
        total_tokens += tokens
        total_branches += len(records)
        if config_name == "replication_screen":
            screen_records = records
            screen_config, screen_summary = config, summary

    contrasts = []
    supporting = []
    penalties = []
    domains: dict[str, list[float]] = defaultdict(list)
    primary = f"continuation_{screen_config['screen']['primary_horizon']}"
    support = f"continuation_{screen_config['screen']['supporting_horizon']}"
    for seed in screen_config["training"]["seeds"]:
        pair = {r["action"]: r["evaluations"] for r in screen_records if r["seed"] == seed}
        science, balanced = pair["science_math"], pair["balanced"]
        penalties.append(science["immediate"]["macro_mean_bits_per_byte"] - balanced["immediate"]["macro_mean_bits_per_byte"])
        contrasts.append(balanced[primary]["macro_mean_bits_per_byte"] - science[primary]["macro_mean_bits_per_byte"])
        supporting.append(balanced[support]["macro_mean_bits_per_byte"] - science[support]["macro_mean_bits_per_byte"])
        row = next(r for r in screen_summary["seed_results"] if r["seed"] == seed)
        require(math.isclose(contrasts[-1], row["science_advantage_at_primary_horizon_bpb"], abs_tol=1e-12), "Seed contrast mismatch")
        for domain in screen_config["domains"]:
            domains[domain].append(balanced[primary]["domains"][domain]["bits_per_byte"] - science[primary]["domains"][domain]["bits_per_byte"])
    advantage = mean(contrasts)
    require(math.isclose(advantage, screen_summary["mean_science_primary_advantage_bpb"], abs_tol=1e-12), "Summary effect mismatch")
    gates = {
        "causal_audit": True,
        "all_seeds_reverse": all(p > 0 and s > 0 and c > 0 for p, s, c in zip(penalties, supporting, contrasts)),
        "all_seeds_better_at_supporting_horizon": all(v > 0 for v in supporting),
        "all_seeds_better_at_primary_horizon": all(v > 0 for v in contrasts),
        "minimum_mean_terminal_advantage": advantage >= screen_config["screen"]["minimum_terminal_advantage_bpb"],
        "no_large_domain_regression": min(mean(v) for v in domains.values()) >= -screen_config["screen"]["maximum_domain_regression_bpb"],
    }
    require(gates == screen_summary["checks"], "Screen gate mismatch")
    status = "pass" if all(gates.values()) else "fail"
    require(status == screen_summary["screen_status"], "Screen status mismatch")
    require((total_branches, len(all_seeds), total_tokens) == (36, 6, 72124416), "Portfolio totals mismatch")
    return {
        "branches": total_branches,
        "seeds": len(all_seeds),
        "training_tokens": total_tokens,
        "replication_advantage_bits_per_byte": advantage,
        "replication_practical_effect_gate": status,
        "per_byte_perplexity_reduction_percent": 100 * (1 - 2 ** (-advantage)),
        "verification": "pass",
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
