from __future__ import annotations

import hashlib
import html
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from .data import EOS_TOKEN
from .utils import atomic_json, sha256_file


WIKITEXT_URLS = {
    "train": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
    "valid": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
    "test": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
}
CPYTHON_URL = "https://codeload.github.com/python/cpython/zip/refs/tags/v3.12.10"
STACKEXCHANGE_URL = (
    "https://daft-oss-public-data.s3.amazonaws.com/"
    "redpajama-1t-sample/stackexchange_sample.jsonl"
)
OPENALEX_CONCEPTS = {
    "physics": "C121332964",
    "mathematics": "C33923547",
    "biology": "C86803240",
    "chemistry": "C185592680",
}


@dataclass(frozen=True)
class Document:
    domain: str
    source: str
    source_id: str
    text: str


def download(url: str, target: Path, timeout: int = 120) -> None:
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "NLRR-1/0.1 research"})
    with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(target)


def _clean_text(text: str) -> str:
    text = html.unescape(text).replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def iter_wikitext(raw_dir: Path) -> Iterator[Document]:
    top_heading = re.compile(r"^\s*=\s+[^=].*?\s+=\s*$")
    for native_split in WIKITEXT_URLS:
        path = raw_dir / f"wikitext2-{native_split}.txt"
        lines = path.read_text(encoding="utf-8").splitlines()
        current: list[str] = []
        index = 0
        for line in lines:
            if top_heading.match(line) and current:
                text = _clean_text("\n".join(current))
                if text:
                    yield Document("prose", "wikitext2", f"{native_split}:{index}", text)
                    index += 1
                current = [line]
            else:
                current.append(line)
        text = _clean_text("\n".join(current))
        if text:
            yield Document("prose", "wikitext2", f"{native_split}:{index}", text)


def iter_cpython(raw_dir: Path) -> Iterator[Document]:
    archive = raw_dir / "cpython-3.12.10.zip"
    with zipfile.ZipFile(archive) as handle:
        names = sorted(
            name
            for name in handle.namelist()
            if "/Lib/" in name
            and name.endswith(".py")
            and "/Lib/test/" not in name
            and "/Lib/idlelib/idle_test/" not in name
        )
        for name in names:
            try:
                text = handle.read(name).decode("utf-8")
            except UnicodeDecodeError:
                continue
            text = _clean_text(text)
            if text:
                yield Document("code", "cpython-3.12.10", name, text)


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    last = max((position for positions in index.values() for position in positions), default=-1)
    words = [""] * (last + 1)
    for word, positions in index.items():
        for position in positions:
            if 0 <= position < len(words):
                words[position] = word
    return " ".join(words)


def iter_openalex(raw_dir: Path) -> Iterator[Document]:
    for path in sorted((raw_dir / "openalex").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for work in payload.get("results", []):
            abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
            title = work.get("title") or ""
            text = _clean_text(f"{title}\n\n{abstract}")
            if abstract and text:
                yield Document("science_math", "openalex", str(work.get("id", path.stem)), text)


def iter_stackexchange(raw_dir: Path) -> Iterator[Document]:
    path = raw_dir / "stackexchange_sample.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = _clean_text(payload.get("text", ""))
            metadata = payload.get("meta") or {}
            source_id = str(metadata.get("url") or f"line:{index}")
            if text:
                yield Document("dialogue_qa", "redpajama-stackexchange", source_id, text)


def _normalized_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _simhash64(text: str) -> int:
    words = re.findall(r"[\w']+", text.casefold())
    if len(words) < 3:
        shingles = words
    else:
        shingles = [" ".join(words[index : index + 3]) for index in range(len(words) - 2)]
    if len(shingles) > 512:
        step = len(shingles) / 512
        shingles = [shingles[int(index * step)] for index in range(512)]
    weights = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    fingerprint = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            fingerprint |= 1 << bit
    return fingerprint


def deduplicate(
    documents: Iterable[Document], minimum_bytes: int, maximum_bytes: int
) -> tuple[list[tuple[Document, str, int]], dict[str, int]]:
    exact_seen: set[str] = set()
    lsh: dict[tuple[int, int], list[int]] = {}
    retained: list[tuple[Document, str, int]] = []
    stats = {"input": 0, "too_short": 0, "exact_duplicate": 0, "near_duplicate": 0, "retained": 0}
    for document in documents:
        stats["input"] += 1
        encoded = document.text.encode("utf-8")
        if len(encoded) < minimum_bytes:
            stats["too_short"] += 1
            continue
        text = encoded[:maximum_bytes].decode("utf-8", errors="ignore")
        document = Document(document.domain, document.source, document.source_id, text)
        normalized = _normalized_for_dedup(text)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in exact_seen:
            stats["exact_duplicate"] += 1
            continue
        fingerprint = _simhash64(normalized)
        candidates: set[int] = set()
        for band in range(4):
            candidates.update(lsh.get((band, (fingerprint >> (band * 16)) & 0xFFFF), []))
        if any((fingerprint ^ prior).bit_count() <= 3 for prior in candidates):
            stats["near_duplicate"] += 1
            continue
        exact_seen.add(digest)
        for band in range(4):
            lsh.setdefault((band, (fingerprint >> (band * 16)) & 0xFFFF), []).append(fingerprint)
        retained.append((document, digest, fingerprint))
        stats["retained"] += 1
    return retained, stats


def _split_for_digest(digest: str) -> str:
    bucket = int(digest[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "development"
    if bucket < 90:
        return "meta_test"
    return "replication"


def fetch_openalex(raw_dir: Path, pages: int, per_page: int) -> None:
    output = raw_dir / "openalex"
    output.mkdir(parents=True, exist_ok=True)
    for concept_name, concept_id in OPENALEX_CONCEPTS.items():
        cursor = "*"
        for page_index in range(pages):
            target = output / f"{concept_name}-{page_index:03d}.json"
            if target.exists() and target.stat().st_size:
                payload = json.loads(target.read_text(encoding="utf-8"))
                cursor = payload.get("meta", {}).get("next_cursor") or cursor
                continue
            parameters = {
                "filter": f"concepts.id:{concept_id},has_abstract:true",
                "sort": "cited_by_count:desc",
                "per-page": str(per_page),
                "cursor": cursor,
                "select": "id,title,abstract_inverted_index,publication_year,primary_topic",
            }
            url = "https://api.openalex.org/works?" + urllib.parse.urlencode(parameters)
            request = urllib.request.Request(url, headers={"User-Agent": "NLRR-1/0.1 research"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        raw = response.read()
                    payload = json.loads(raw)
                    target.write_bytes(raw)
                    cursor = payload.get("meta", {}).get("next_cursor") or cursor
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
            time.sleep(0.12)


def acquire_sources(config: dict, raw_dir: Path) -> list[dict[str, str | int]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split, url in WIKITEXT_URLS.items():
        download(url, raw_dir / f"wikitext2-{split}.txt")
    download(CPYTHON_URL, raw_dir / "cpython-3.12.10.zip")
    download(STACKEXCHANGE_URL, raw_dir / "stackexchange_sample.jsonl", timeout=300)
    fetch_openalex(
        raw_dir,
        int(config["data"]["openalex_pages_per_concept"]),
        int(config["data"]["openalex_per_page"]),
    )
    sources: list[dict[str, str | int]] = []
    for path in sorted(file for file in raw_dir.rglob("*") if file.is_file() and not file.name.endswith(".partial")):
        sources.append(
            {
                "relative_path": path.relative_to(raw_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sources


def prepare_corpus(config: dict, repo_root: Path) -> dict:
    cache = Path(config["cache_dir"])
    raw_dir = cache / "data" / "raw"
    prepared_dir = cache / "data" / "prepared" / config["experiment_id"]
    prepared_dir.mkdir(parents=True, exist_ok=True)
    source_files = acquire_sources(config, raw_dir)

    domain_iterators = {
        "prose": iter_wikitext(raw_dir),
        "science_math": iter_openalex(raw_dir),
        "code": iter_cpython(raw_dir),
        "dialogue_qa": iter_stackexchange(raw_dir),
    }
    minimum = int(config["data"]["minimum_document_bytes"])
    maximum = int(config["data"]["maximum_document_bytes"])
    target_bytes = {key: int(value) for key, value in config["data"]["target_bytes"].items()}

    by_domain: dict[str, list[tuple[Document, str, int]]] = {}
    dedup_stats: dict[str, dict[str, int]] = {}
    # Domain-level processing keeps source semantics fixed. Cross-domain exact
    # collisions are removed below before assigning splits.
    for domain in config["domains"]:
        retained, stats = deduplicate(domain_iterators[domain], minimum, maximum)
        by_domain[domain] = retained
        dedup_stats[domain] = stats

    cross_domain_seen: set[str] = set()
    split_docs: dict[str, dict[str, list[tuple[Document, str]]]] = {
        domain: {split: [] for split in target_bytes} for domain in config["domains"]
    }
    cross_domain_removed = 0
    for domain in config["domains"]:
        for document, digest, _ in sorted(by_domain[domain], key=lambda item: item[1]):
            if digest in cross_domain_seen:
                cross_domain_removed += 1
                continue
            cross_domain_seen.add(digest)
            split_docs[domain][_split_for_digest(digest)].append((document, digest))

    available = {
        domain: {
            split: sum(len(document.text.encode("utf-8")) for document, _ in documents)
            for split, documents in splits.items()
        }
        for domain, splits in split_docs.items()
    }
    actual_targets = {
        split: min(target_bytes[split], *(available[domain][split] for domain in config["domains"]))
        for split in target_bytes
    }
    if min(actual_targets.values()) < 100_000:
        raise RuntimeError(f"A source/split is too small for the experiment: {actual_targets}")

    records_path = prepared_dir / "document_manifest.jsonl"
    with records_path.open("w", encoding="utf-8", newline="\n") as records:
        for domain in config["domains"]:
            domain_dir = prepared_dir / domain
            domain_dir.mkdir(parents=True, exist_ok=True)
            for split, documents in split_docs[domain].items():
                remaining = actual_targets[split]
                chunks: list[np.ndarray] = []
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
        "format": "uint16 byte tokens; 0-255=UTF-8 byte, 257=EOS",
        "split_policy": "sha256(normalized document) modulo 100: 70/10/10/10",
        "dedup_policy": "normalized SHA-256 exact; 64-bit 3-word SimHash, Hamming <= 3 near-duplicate",
        "requested_target_text_bytes": target_bytes,
        "actual_target_text_bytes_per_domain": actual_targets,
        "available_text_bytes": available,
        "dedup_stats": dedup_stats,
        "cross_domain_exact_duplicates_removed": cross_domain_removed,
        "sources": source_files,
        "prepared_files": prepared_files,
        "licenses": {
            "wikitext2": "Creative Commons Attribution-ShareAlike (source dataset terms)",
            "openalex": "OpenAlex dataset/API terms; bibliographic metadata is CC0, underlying abstract rights may vary",
            "cpython-3.12.10": "Python Software Foundation License Version 2",
            "redpajama-stackexchange": "Stack Exchange content under applicable Creative Commons license",
        },
    }
    atomic_json(prepared_dir / "manifest.json", manifest)
    artifact_dir = repo_root / "artifacts" / config["experiment_id"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(artifact_dir / "source_and_prepared_manifest.json", manifest)
    return manifest
