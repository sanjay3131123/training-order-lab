from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.prepare import prepare_corpus  # noqa: E402
from nlrr.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    manifest = prepare_corpus(config, ROOT)
    summary = {
        "experiment_id": manifest["experiment_id"],
        "actual_target_text_bytes_per_domain": manifest["actual_target_text_bytes_per_domain"],
        "dedup_stats": manifest["dedup_stats"],
        "cross_domain_exact_duplicates_removed": manifest["cross_domain_exact_duplicates_removed"],
        "source_files": len(manifest["sources"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
