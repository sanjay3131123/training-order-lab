from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .data import Corpus, DomainSampler
from .experiment import build_model_and_optimizer, train_steps
from .utils import environment_manifest, seed_everything


def synthetic_world(prerequisite_matters: bool) -> dict[str, Any]:
    """A unit-test world with an inserted delayed prerequisite effect."""
    states = {
        "immediate_skill": {"A": 1.0, "B": 0.0},
        "prerequisite": {"A": 0.0, "B": 1.0},
    }
    immediate = {action: states["immediate_skill"][action] for action in ("A", "B")}
    terminal = {}
    for action in ("A", "B"):
        downstream_gain = 2.0 * states["prerequisite"][action] if prerequisite_matters else 0.5
        terminal[action] = immediate[action] + downstream_gain
    reversal = immediate["A"] > immediate["B"] and terminal["A"] < terminal["B"]
    return {"immediate": immediate, "terminal": terminal, "reversal": reversal}


def _toy_config(cache_dir: Path) -> dict[str, Any]:
    return {
        "experiment_id": "stage-minus-one",
        "cache_dir": str(cache_dir),
        "domains": ["alpha", "beta"],
        "actions": ["alpha", "beta", "balanced"],
        "model": {
            "vocab_size": 258,
            "block_size": 8,
            "n_layer": 1,
            "n_head": 2,
            "n_embd": 32,
            "dropout": 0.0,
            "bias": False,
        },
        "training": {
            "batch_size": 2,
            "threads": 1,
            "learning_rate": 0.001,
            "minimum_learning_rate": 0.0001,
            "warmup_tokens": 16,
            "schedule_tokens": 1000,
            "weight_decay": 0.01,
            "beta1": 0.9,
            "beta2": 0.95,
            "gradient_clip": 1.0,
            "log_every_steps": 0,
        },
    }


def _make_toy_corpus(root: Path) -> Corpus:
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps({"kind": "stage-minus-one"}), encoding="utf-8")
    for domain_index, domain in enumerate(("alpha", "beta")):
        directory = root / domain
        directory.mkdir()
        base = np.asarray(
            [(index * (domain_index + 3) + 17) % 256 for index in range(4096)], dtype=np.uint16
        )
        for split in ("train", "development", "meta_test", "replication"):
            base.tofile(directory / f"{split}.bin")
    return Corpus(root, ["alpha", "beta"])


def run_sanity_checks() -> dict[str, Any]:
    positive = synthetic_world(prerequisite_matters=True)
    negative = synthetic_world(prerequisite_matters=False)
    with tempfile.TemporaryDirectory(prefix="nlrr-sanity-") as temporary:
        root = Path(temporary)
        config = _toy_config(root)
        corpus = _make_toy_corpus(root / "prepared")
        seed_everything(991, 1)
        model, optimizer = build_model_and_optimizer(config)
        sampler = DomainSampler(corpus, "train", "balanced", 8, 2, 8128)
        global_tokens, _ = train_steps(model, optimizer, sampler, 4, 0, config)
        checkpoint = root / "checkpoint.pt"
        save_checkpoint(checkpoint, model, optimizer, global_tokens, sampler.state_dict())

        expected_random = torch.rand(8)
        global_tokens_expected, _ = train_steps(
            model, optimizer, sampler, 3, global_tokens, config
        )
        expected_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

        resumed_model, resumed_optimizer = build_model_and_optimizer(config)
        resumed_sampler = DomainSampler(corpus, "train", "balanced", 8, 2, 8128)
        payload = load_checkpoint(checkpoint, resumed_model, resumed_optimizer, restore_rng=True)
        resumed_sampler.load_state_dict(payload["sampler"])
        resumed_random = torch.rand(8)
        global_tokens_resumed, _ = train_steps(
            resumed_model,
            resumed_optimizer,
            resumed_sampler,
            3,
            int(payload["global_tokens"]),
            config,
        )
        exact_resume = global_tokens_resumed == global_tokens_expected and all(
            torch.equal(expected_state[name], tensor)
            for name, tensor in resumed_model.state_dict().items()
        )
        rng_resume = torch.equal(expected_random, resumed_random)

        fork_one, optimizer_one = build_model_and_optimizer(config)
        fork_two, optimizer_two = build_model_and_optimizer(config)
        sampler_one = DomainSampler(corpus, "train", "balanced", 8, 2, 8128)
        sampler_two = DomainSampler(corpus, "train", "balanced", 8, 2, 8128)
        payload_one = load_checkpoint(checkpoint, fork_one, optimizer_one, restore_rng=True)
        sampler_one.load_state_dict(payload_one["sampler"])
        payload_two = load_checkpoint(checkpoint, fork_two, optimizer_two, restore_rng=True)
        sampler_two.load_state_dict(payload_two["sampler"])
        first_x, first_y = sampler_one.next_batch()
        second_x, second_y = sampler_two.next_batch()
        identical_fork_batch = torch.equal(first_x, second_x) and torch.equal(first_y, second_y)

        hidden_split_blocked = False
        try:
            corpus.tokens("alpha", "meta_test", purpose="discovery_eval")
        except PermissionError:
            hidden_split_blocked = True
        corpus.close()

    checks = {
        "synthetic_positive_reversal_recovered": bool(positive["reversal"]),
        "synthetic_negative_control_has_no_reversal": not bool(negative["reversal"]),
        "exact_interrupted_resume": exact_resume,
        "rng_state_resume": rng_resume,
        "bit_identical_fork_batch": identical_fork_batch,
        "meta_test_blocked_from_discovery": hidden_split_blocked,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "positive_world": positive,
        "negative_world": negative,
        "environment": environment_manifest(),
    }
