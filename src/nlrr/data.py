from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .utils import stable_seed


EOS_TOKEN = 257
ALLOWED_TRAIN_SPLITS = {"train"}
ALLOWED_DISCOVERY_EVAL_SPLITS = {"development"}


class Corpus:
    def __init__(self, prepared_dir: str | Path, domains: list[str]) -> None:
        self.prepared_dir = Path(prepared_dir)
        self.domains = list(domains)
        manifest_path = self.prepared_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Prepared corpus missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._arrays: dict[tuple[str, str], np.memmap] = {}

    def close(self) -> None:
        for array in self._arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._arrays.clear()

    def tokens(self, domain: str, split: str, purpose: str = "train") -> np.memmap:
        if purpose not in {"train", "discovery_eval"}:
            raise ValueError(f"Unknown corpus access purpose: {purpose!r}")
        if domain not in self.domains:
            raise KeyError(domain)
        if purpose == "train" and split not in ALLOWED_TRAIN_SPLITS:
            raise PermissionError(f"Training code cannot access split {split!r}")
        if purpose == "discovery_eval" and split not in ALLOWED_DISCOVERY_EVAL_SPLITS:
            raise PermissionError(f"Discovery code cannot access split {split!r}")
        key = (domain, split)
        if key not in self._arrays:
            path = self.prepared_dir / domain / f"{split}.bin"
            self._arrays[key] = np.memmap(path, dtype=np.uint16, mode="r")
        return self._arrays[key]


class DomainSampler:
    """Serializable deterministic random-block sampler."""

    def __init__(
        self,
        corpus: Corpus,
        split: str,
        action: str,
        block_size: int,
        batch_size: int,
        seed: int,
    ) -> None:
        self.corpus = corpus
        self.split = split
        self.action = action
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.balanced_cursor = 0
        self.batch_count = 0
        self.sequence_digest = hashlib.sha256()

    def state_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "action": self.action,
            "block_size": self.block_size,
            "batch_size": self.batch_size,
            "rng_state": self.rng.bit_generator.state,
            "balanced_cursor": self.balanced_cursor,
            "batch_count": self.batch_count,
            "sequence_digest": self.sequence_digest.hexdigest(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state["split"] != self.split or state["action"] != self.action:
            raise ValueError("Sampler identity mismatch")
        self.rng.bit_generator.state = state["rng_state"]
        self.balanced_cursor = int(state["balanced_cursor"])
        self.batch_count = int(state["batch_count"])
        # A hashlib internal state is not serializable. The prior digest is included
        # as an audit marker; resumed sequence equality is checked from the resume point.
        self.sequence_digest = hashlib.sha256()
        self.sequence_digest.update(state["sequence_digest"].encode("ascii"))

    def _next_domain(self) -> str:
        if self.action != "balanced":
            return self.action
        domain = self.corpus.domains[self.balanced_cursor % len(self.corpus.domains)]
        self.balanced_cursor += 1
        return domain

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        domain = self._next_domain()
        values = self.corpus.tokens(domain, self.split, purpose="train")
        maximum = len(values) - self.block_size - 1
        if maximum <= 0:
            raise ValueError(f"Not enough tokens in {domain}/{self.split}")
        starts = self.rng.integers(0, maximum, size=self.batch_size, dtype=np.int64)
        x = np.stack([values[start : start + self.block_size] for start in starts]).astype(np.int64)
        y = np.stack([values[start + 1 : start + self.block_size + 1] for start in starts]).astype(np.int64)
        self.sequence_digest.update(domain.encode("utf-8") + b"\0" + starts.tobytes())
        self.batch_count += 1
        return torch.from_numpy(x), torch.from_numpy(y)


def fixed_eval_batches(
    corpus: Corpus,
    domain: str,
    block_size: int,
    batch_size: int,
    batches: int,
    experiment_id: str,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    values = corpus.tokens(domain, "development", purpose="discovery_eval")
    maximum = len(values) - block_size - 1
    rng = np.random.Generator(np.random.PCG64(stable_seed(experiment_id, "eval", domain)))
    output: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(batches):
        starts = rng.integers(0, maximum, size=batch_size, dtype=np.int64)
        x = np.stack([values[start : start + block_size] for start in starts]).astype(np.int64)
        y = np.stack([values[start + 1 : start + block_size + 1] for start in starts]).astype(np.int64)
        output.append((torch.from_numpy(x), torch.from_numpy(y)))
    return output
