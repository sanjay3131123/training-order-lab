from __future__ import annotations

import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .data import EOS_TOKEN
from .prepare import Document, _clean_text, deduplicate, download
from .utils import atomic_json, sha256_file


GUTENBERG_IDS = [
    11, 35, 36, 43, 55, 74, 76, 84, 98, 105, 120, 134, 158, 160, 161, 174,
    201, 244, 408, 514, 768, 829, 1080, 1184, 1228, 1232, 1260, 1342, 1400,
    1661, 2542, 2701, 345, 5200,
]
OPENSTAX_URL = (
    "https://codeload.github.com/openstax/osbooks-calculus-bundle/zip/refs/heads/main"
)
GO_URL = "https://codeload.github.com/golang/go/zip/refs/tags/go1.23.0"
DOLLY_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)


def acquire_replication_sources(raw_dir: Path) -> list[dict[str, str | int]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    gutenberg = raw_dir / "gutenberg"
    gutenberg.mkdir(exist_ok=True)
    failures = []
    for book_id in GUTENBERG_IDS:
        target = gutenberg / f"pg{book_id}.txt"
        try:
            download(
                f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
                target,
                timeout=120,
            )
        except Exception as error:
            failures.append({"book_id": book_id, "error": str(error)})
    if failures:
        atomic_json(raw_dir / "gutenberg_failures.json", failures)
    download(OPENSTAX_URL, raw_dir / "openstax-calculus-main.zip", timeout=600)
    download(GO_URL, raw_dir / "go1.23.0.zip", timeout=600)
    download(DOLLY_URL, raw_dir / "databricks-dolly-15k.jsonl", timeout=300)
    sources = []
    for path in sorted(file for file in raw_dir.rglob("*") if file.is_file() and not file.name.endswith(".partial")):
        sources.append(
            {
                "relative_path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sources


def _strip_gutenberg_boilerplate(text: str) -> str:
    start_patterns = [
        r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]
    end_patterns = [
        r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*",
        r"\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*",
    ]
    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end() :]
            break
    for pattern in end_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = text[: match.start()]
            break
    return _clean_text(text)


def iter_gutenberg(raw_dir: Path) -> Iterator[Document]:
    for path in sorted((raw_dir / "gutenberg").glob("pg*.txt")):
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        text = _strip_gutenberg_boilerplate(text)
        if text:
            yield Document("prose", "project-gutenberg", path.stem, text)


def _xml_text(raw: bytes) -> str:
    try:
        root = ET.fromstring(raw)
        return _clean_text(" ".join(root.itertext()))
    except ET.ParseError:
        return ""


def iter_openstax(raw_dir: Path) -> Iterator[Document]:
    archive = raw_dir / "openstax-calculus-main.zip"
    with zipfile.ZipFile(archive) as handle:
        candidates = [
            name
            for name in handle.namelist()
            if name.lower().endswith((".cnxml", ".xhtml", ".xml"))
            and "media/" not in name.lower()
            and "metadata" not in name.lower()
            and not name.lower().endswith(("collection.xml", "manifest.xml"))
        ]
        for name in sorted(candidates):
            text = _xml_text(handle.read(name))
            if text:
                yield Document("science_math", "openstax-calculus", name, text)


def iter_go(raw_dir: Path) -> Iterator[Document]:
    archive = raw_dir / "go1.23.0.zip"
    with zipfile.ZipFile(archive) as handle:
        candidates = [
            name
            for name in handle.namelist()
            if "/src/" in name
            and name.endswith(".go")
            and not name.endswith("_test.go")
            and "/testdata/" not in name
            and "/vendor/" not in name
            and "/cmd/" not in name
        ]
        for name in sorted(candidates):
            try:
                text = handle.read(name).decode("utf-8")
            except UnicodeDecodeError:
                continue
            text = _clean_text(text)
            if text:
                yield Document("code", "go1.23.0-stdlib", name, text)


def iter_dolly(raw_dir: Path) -> Iterator[Document]:
    path = raw_dir / "databricks-dolly-15k.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pieces = [f"Instruction: {record.get('instruction', '').strip()}"]
            context = record.get("context", "").strip()
            if context:
                pieces.append(f"Context: {context}")
            pieces.append(f"Response: {record.get('response', '').strip()}")
            text = _clean_text("\n\n".join(pieces))
            if text:
                yield Document("dialogue_qa", "databricks-dolly-15k", f"record:{index}", text)


def _stratified_split(documents: list[tuple[Document, str, int]]) -> dict[str, list[tuple[Document, str]]]:
    output = {"train": [], "development": [], "meta_test": [], "replication": []}
    order = ["train"] * 7 + ["development", "meta_test", "replication"]
    for index, (document, digest, _) in enumerate(sorted(documents, key=lambda item: item[1])):
        output[order[index % 10]].append((document, digest))
    return output


def _phase1_hashes(cache: Path) -> set[str]:
    path = cache / "data" / "prepared" / "nlrr1-discovery-v1" / "document_manifest.jsonl"
    hashes = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                hashes.add(json.loads(line)["content_sha256"])
    return hashes


def prepare_replication_corpus(config: dict, repo_root: Path) -> dict:
    cache = Path(config["cache_dir"])
    raw_dir = cache / "data" / "raw-replication-v1"
    prepared_dir = cache / "data" / "prepared" / config["experiment_id"]
    prepared_dir.mkdir(parents=True, exist_ok=True)
    source_files = acquire_replication_sources(raw_dir)
    iterators: dict[str, Iterable[Document]] = {
        "prose": iter_gutenberg(raw_dir),
        "science_math": iter_openstax(raw_dir),
        "code": iter_go(raw_dir),
        "dialogue_qa": iter_dolly(raw_dir),
    }
    minimum = int(config["data"]["minimum_document_bytes"])
    maximum = int(config["data"]["maximum_document_bytes"])
    target_bytes = {key: int(value) for key, value in config["data"]["target_bytes"].items()}
    prior_hashes = _phase1_hashes(cache)

    by_domain: dict[str, list[tuple[Document, str, int]]] = {}
    dedup_stats = {}
    prior_overlap_removed = {}
    for domain in config["domains"]:
        retained, stats = deduplicate(iterators[domain], minimum, maximum)
        overlap = sum(1 for _, digest, _ in retained if digest in prior_hashes)
        retained = [item for item in retained if item[1] not in prior_hashes]
        by_domain[domain] = retained
        dedup_stats[domain] = stats
        prior_overlap_removed[domain] = overlap

    global_seen: set[str] = set()
    splits = {}
    cross_domain_removed = 0
    for domain in config["domains"]:
        unique = []
        for item in by_domain[domain]:
            if item[1] in global_seen:
                cross_domain_removed += 1
                continue
            global_seen.add(item[1])
            unique.append(item)
        splits[domain] = _stratified_split(unique)

    available = {
        domain: {
            split: sum(len(document.text.encode("utf-8")) for document, _ in documents)
            for split, documents in domain_splits.items()
        }
        for domain, domain_splits in splits.items()
    }
    actual_targets = {
        split: min(target_bytes[split], *(available[domain][split] for domain in config["domains"]))
        for split in target_bytes
    }
    if min(actual_targets.values()) < 200_000:
        raise RuntimeError(
            f"Independent source corpus is too small for the frozen screen: {actual_targets}; available={available}"
        )

    document_manifest = prepared_dir / "document_manifest.jsonl"
    with document_manifest.open("w", encoding="utf-8", newline="\n") as records:
        for domain in config["domains"]:
            domain_dir = prepared_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            for split, documents in splits[domain].items():
                remaining = actual_targets[split]
                chunks = []
                used_bytes = 0
                used_docs = 0
                for document, digest in documents:
                    if remaining <= 0:
                        break
                    raw = document.text.encode("utf-8")[:remaining]
                    if not raw:
                        continue
                    chunks.append(np.frombuffer(raw, dtype=np.uint8).astype(np.uint16))
                    chunks.append(np.asarray([EOS_TOKEN], dtype=np.uint16))
                    used_bytes += len(raw)
                    remaining -= len(raw)
                    used_docs += 1
                    records.write(
                        json.dumps(
                            {
                                "domain": domain,
                                "split": split,
                                "source": document.source,
                                "source_id": document.source_id,
                                "content_sha256": digest,
                                "bytes_used": len(raw),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                if remaining:
                    raise RuntimeError(f"Could not fill {domain}/{split}: {remaining} bytes short")
                array = np.concatenate(chunks)
                array.tofile(domain_dir / f"{split}.bin")
                atomic_json(
                    domain_dir / f"{split}.json",
                    {"text_bytes": used_bytes, "tokens": int(array.size), "documents": used_docs},
                )

    prepared_files = []
    for path in sorted(file for file in prepared_dir.rglob("*") if file.is_file() and file.name != "manifest.json"):
        prepared_files.append(
            {
                "relative_path": path.relative_to(prepared_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "experiment_id": config["experiment_id"],
        "status": "frozen replication-screen corpus",
        "format": "uint16 byte tokens; 0-255=UTF-8 byte, 257=EOS",
        "split_policy": "source-document SHA-256 order, repeated 70/10/10/10 stratified cycle",
        "dedup_policy": "normalized SHA-256 exact; 64-bit 3-word SimHash, Hamming <= 3 near-duplicate",
        "requested_target_text_bytes": target_bytes,
        "actual_target_text_bytes_per_domain": actual_targets,
        "available_text_bytes": available,
        "dedup_stats": dedup_stats,
        "phase1_exact_overlaps_removed": prior_overlap_removed,
        "cross_domain_exact_duplicates_removed": cross_domain_removed,
        "sources": source_files,
        "prepared_files": prepared_files,
        "licenses": {
            "project-gutenberg": "Project Gutenberg public-domain/usage terms per work",
            "openstax-calculus": "Creative Commons Attribution 4.0",
            "go1.23.0-stdlib": "BSD-style Go license",
            "databricks-dolly-15k": "Creative Commons Attribution-ShareAlike 3.0",
        },
    }
    atomic_json(prepared_dir / "manifest.json", manifest)
    artifact_dir = repo_root / "artifacts" / config["experiment_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_dir / "source_and_prepared_manifest.json", manifest)
    return manifest
