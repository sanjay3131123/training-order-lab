from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.replication_analysis import analyze_replication_screen  # noqa: E402
from nlrr.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze_replication_screen(load_config(arguments.config), ROOT)
    print(
        json.dumps(
            {
                "screen_status": summary["screen_status"],
                "decision": summary["decision"],
                "checks": summary["checks"],
                "mean_science_primary_advantage_bpb": summary[
                    "mean_science_primary_advantage_bpb"
                ],
                "seed_results": summary["seed_results"],
                "total_training_tokens": summary["total_training_tokens"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
