from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.prepare_replication import prepare_replication_corpus  # noqa: E402
from nlrr.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = prepare_replication_corpus(load_config(arguments.config), ROOT)
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "actual_target_text_bytes_per_domain": manifest["actual_target_text_bytes_per_domain"],
                "available_text_bytes": manifest["available_text_bytes"],
                "dedup_stats": manifest["dedup_stats"],
                "phase1_exact_overlaps_removed": manifest["phase1_exact_overlaps_removed"],
                "source_files": len(manifest["sources"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
