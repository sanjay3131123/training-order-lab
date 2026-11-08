from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import Corpus, DomainSampler, fixed_eval_batches
from .model import ModelConfig, TinyGPT
from .utils import atomic_json, environment_manifest, rounded_steps, seed_everything, stable_seed


def build_model_and_optimizer(config: dict[str, Any]) -> tuple[TinyGPT, torch.optim.Optimizer]:
    model = TinyGPT(ModelConfig(**config["model"]))
    decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.dim() >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.dim() < 2]
    training = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(training["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(training["learning_rate"]),
        betas=(float(training["beta1"]), float(training["beta2"])),
        fused=False,
    )
    return model, optimizer


def learning_rate(config: dict[str, Any], global_tokens: int) -> float:
    training = config["training"]
    maximum = float(training["learning_rate"])
    minimum = float(training["minimum_learning_rate"])
    warmup = int(training["warmup_tokens"])
    schedule = int(training["schedule_tokens"])
    if global_tokens < warmup:
        return maximum * max(global_tokens, 1) / warmup
    if global_tokens >= schedule:
        return minimum
    ratio = (global_tokens - warmup) / max(1, schedule - warmup)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return minimum + coefficient * (maximum - minimum)


def train_steps(
    model: TinyGPT,
    optimizer: torch.optim.Optimizer,
    sampler: DomainSampler,
    steps: int,
    global_tokens: int,
    config: dict[str, Any],
    log_prefix: str = "",
) -> tuple[int, dict[str, float]]:
    model.train()
    tokens_per_step = int(config["training"]["batch_size"]) * int(config["model"]["block_size"])
    log_every = int(config["training"]["log_every_steps"])
    gradient_clip = float(config["training"]["gradient_clip"])
    losses: list[float] = []
    started = time.perf_counter()
    for local_step in range(steps):
        rate = learning_rate(config, global_tokens)
        for group in optimizer.param_groups:
            group["lr"] = rate
        inputs, targets = sampler.next_batch()
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(inputs, targets)
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        global_tokens += tokens_per_step
        losses.append(float(loss.detach()))
        if log_every and (local_step + 1) % log_every == 0:
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate_tps = (local_step + 1) * tokens_per_step / elapsed
            print(
                f"{log_prefix} step {local_step + 1}/{steps} "
                f"loss={sum(losses[-log_every:]) / min(log_every, len(losses)):.4f} "
                f"tokens/s={rate_tps:,.0f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    return global_tokens, {
        "steps": steps,
        "tokens": steps * tokens_per_step,
        "mean_loss_nats": sum(losses) / max(1, len(losses)),
        "elapsed_seconds": elapsed,
        "tokens_per_second": steps * tokens_per_step / max(elapsed, 1e-9),
    }


@torch.inference_mode()
def evaluate(
    model: TinyGPT,
    eval_batches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
) -> dict[str, Any]:
    model.eval()
    domain_metrics: dict[str, dict[str, float | int]] = {}
    total_nats = 0.0
    total_tokens = 0
    for domain, batches in eval_batches.items():
        domain_nats = 0.0
        domain_tokens = 0
        for inputs, targets in batches:
            _, loss = model(inputs, targets)
            assert loss is not None
            count = targets.numel()
            domain_nats += float(loss) * count
            domain_tokens += count
        average_nats = domain_nats / domain_tokens
        domain_metrics[domain] = {
            "loss_nats": average_nats,
            "bits_per_byte": average_nats / math.log(2.0),
            "tokens": domain_tokens,
        }
        total_nats += domain_nats
        total_tokens += domain_tokens
    aggregate_nats = total_nats / total_tokens
    return {
        "domains": domain_metrics,
        "macro_mean_bits_per_byte": sum(
            float(metrics["bits_per_byte"]) for metrics in domain_metrics.values()
        )
        / len(domain_metrics),
        "micro_loss_nats": aggregate_nats,
        "micro_bits_per_byte": aggregate_nats / math.log(2.0),
        "tokens": total_tokens,
    }


def _state_hash(model: TinyGPT) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _parent_checkpoints(
    config: dict[str, Any], corpus: Corpus, seed: int, run_root: Path
) -> dict[int, Path]:
    training = config["training"]
    model, optimizer = build_model_and_optimizer(config)
    sampler_seed = stable_seed(config["experiment_id"], seed, "trunk", "balanced")
    sampler = DomainSampler(
        corpus,
        "train",
        "balanced",
        int(config["model"]["block_size"]),
        int(training["batch_size"]),
        sampler_seed,
    )
    parent_dir = run_root / f"seed-{seed}" / "parents"
    parent_dir.mkdir(parents=True, exist_ok=True)
    targets = sorted(int(value) for value in training["checkpoint_tokens"])
    tokens_per_step = int(training["batch_size"]) * int(config["model"]["block_size"])
    checkpoints = {target: parent_dir / f"tokens-{target}.pt" for target in targets}
    global_tokens = 0

    existing = [target for target in targets if checkpoints[target].exists()]
    if existing:
        latest = max(existing)
        payload = load_checkpoint(checkpoints[latest], model, optimizer, restore_rng=True)
        sampler.load_state_dict(payload["sampler"])
        global_tokens = int(payload["global_tokens"])

    for target in targets:
        if checkpoints[target].exists():
            continue
        target_steps = rounded_steps(target, tokens_per_step)
        completed_steps = global_tokens // tokens_per_step
        steps = target_steps - completed_steps
        print(f"seed={seed} trunk -> {target:,} tokens ({steps} steps)", flush=True)
        global_tokens, stats = train_steps(
            model, optimizer, sampler, steps, global_tokens, config, f"seed={seed} trunk"
        )
        save_checkpoint(
            checkpoints[target],
            model,
            optimizer,
            global_tokens,
            sampler.state_dict(),
            {
                "kind": "parent",
                "seed": seed,
                "requested_tokens": target,
                "actual_tokens": global_tokens,
                "training_stats": stats,
                "model_state_sha256": _state_hash(model),
            },
        )
    return checkpoints


def _run_branch(
    config: dict[str, Any],
    corpus: Corpus,
    seed: int,
    checkpoint_target: int,
    parent_path: Path,
    action: str,
    run_root: Path,
    eval_batches: dict[str, list[tuple[torch.Tensor, torch.Tensor]]],
) -> Path:
    branch_dir = run_root / f"seed-{seed}" / f"checkpoint-{checkpoint_target}" / action
    result_path = branch_dir / "result.json"
    if result_path.exists():
        print(f"skip complete seed={seed} checkpoint={checkpoint_target} action={action}", flush=True)
        return result_path
    branch_dir.mkdir(parents=True, exist_ok=True)
    model, optimizer = build_model_and_optimizer(config)
    payload = load_checkpoint(parent_path, model, optimizer, restore_rng=True)
    parent_hash = payload["metadata"]["model_state_sha256"]
    global_tokens = int(payload["global_tokens"])
    training = config["training"]
    tokens_per_step = int(training["batch_size"]) * int(config["model"]["block_size"])

    intervention_sampler = DomainSampler(
        corpus,
        "train",
        action,
        int(config["model"]["block_size"]),
        int(training["batch_size"]),
        stable_seed(config["experiment_id"], seed, checkpoint_target, action, "intervention"),
    )
    intervention_steps = rounded_steps(int(training["intervention_tokens"]), tokens_per_step)
    print(
        f"seed={seed} checkpoint={checkpoint_target:,} action={action} intervention",
        flush=True,
    )
    global_tokens, intervention_stats = train_steps(
        model,
        optimizer,
        intervention_sampler,
        intervention_steps,
        global_tokens,
        config,
        f"seed={seed} cp={checkpoint_target} {action} intervention",
    )
    evaluations = {"immediate": evaluate(model, eval_batches)}

    continuation_sampler = DomainSampler(
        corpus,
        "train",
        "balanced",
        int(config["model"]["block_size"]),
        int(training["batch_size"]),
        stable_seed(config["experiment_id"], seed, checkpoint_target, "identical-continuation"),
    )
    continuation_stats: dict[str, Any] = {}
    completed_steps = 0
    for horizon in sorted(int(value) for value in training["continuation_horizons"]):
        target_steps = rounded_steps(horizon, tokens_per_step)
        steps = target_steps - completed_steps
        print(
            f"seed={seed} checkpoint={checkpoint_target:,} action={action} "
            f"continuation -> {horizon:,}",
            flush=True,
        )
        global_tokens, stats = train_steps(
            model,
            optimizer,
            continuation_sampler,
            steps,
            global_tokens,
            config,
            f"seed={seed} cp={checkpoint_target} {action} cont={horizon}",
        )
        completed_steps = target_steps
        continuation_stats[str(horizon)] = stats
        evaluations[f"continuation_{horizon}"] = evaluate(model, eval_batches)

    terminal_model_hash = _state_hash(model)
    terminal_checkpoint = None
    if bool(training.get("save_terminal_checkpoint", False)):
        terminal_checkpoint_path = branch_dir / "terminal.pt"
        save_checkpoint(
            terminal_checkpoint_path,
            model,
            optimizer,
            global_tokens,
            continuation_sampler.state_dict(),
            {
                "kind": "terminal_branch",
                "seed": seed,
                "checkpoint_requested_tokens": checkpoint_target,
                "action": action,
                "terminal_model_state_sha256": terminal_model_hash,
            },
        )
        terminal_checkpoint = str(terminal_checkpoint_path)

    result = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "seed": seed,
        "checkpoint_requested_tokens": checkpoint_target,
        "checkpoint_actual_tokens": int(payload["global_tokens"]),
        "action": action,
        "parent_checkpoint": str(parent_path),
        "parent_model_state_sha256": parent_hash,
        "parameter_count": model.parameter_count(),
        "tokens_per_step": tokens_per_step,
        "intervention": {
            "requested_tokens": int(training["intervention_tokens"]),
            "stats": intervention_stats,
            "batch_sequence_sha256": intervention_sampler.sequence_digest.hexdigest(),
        },
        "continuation": {
            "requested_horizons": [int(value) for value in training["continuation_horizons"]],
            "stats": continuation_stats,
            "batch_sequence_sha256": continuation_sampler.sequence_digest.hexdigest(),
        },
        "evaluations": evaluations,
        "terminal_model_state_sha256": terminal_model_hash,
        "terminal_checkpoint": terminal_checkpoint,
        "final_global_tokens": global_tokens,
    }
    atomic_json(result_path, result)
    return result_path


def run_phase1(config: dict[str, Any], repo_root: Path) -> list[Path]:
    seed_everything(int(config["training"]["seeds"][0]), int(config["training"]["threads"]))
    cache = Path(config["cache_dir"])
    prepared_dir = cache / "data" / "prepared" / config["experiment_id"]
    run_root = cache / "runs" / config["experiment_id"]
    run_root.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(prepared_dir, list(config["domains"]))
    eval_batches = {
        domain: fixed_eval_batches(
            corpus,
            domain,
            int(config["model"]["block_size"]),
            int(config["training"]["batch_size"]),
            int(config["training"]["eval_batches_per_domain"]),
            config["experiment_id"],
        )
        for domain in config["domains"]
    }
    model_probe, _ = build_model_and_optimizer(config)
    run_manifest = {
        "experiment_id": config["experiment_id"],
        "environment": environment_manifest(),
        "parameter_count": model_probe.parameter_count(),
        "config": config,
        "prepared_manifest": corpus.manifest,
        "design_status": "exploratory discovery; permanent meta-test sealed",
    }
    atomic_json(run_root / "run_manifest.json", run_manifest)
    artifact_dir = repo_root / "artifacts" / config["experiment_id"]
    atomic_json(artifact_dir / "run_manifest.json", run_manifest)

    results: list[Path] = []
    for seed in [int(value) for value in config["training"]["seeds"]]:
        seed_everything(seed, int(config["training"]["threads"]))
        parents = _parent_checkpoints(config, corpus, seed, run_root)
        for checkpoint_target in sorted(parents):
            for action in config["actions"]:
                results.append(
                    _run_branch(
                        config,
                        corpus,
                        seed,
                        checkpoint_target,
                        parents[checkpoint_target],
                        action,
                        run_root,
                        eval_batches,
                    )
                )
    return results
