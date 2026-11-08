from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.analysis import analyze_phase1  # noqa: E402
from nlrr.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    summary = analyze_phase1(load_config(arguments.config), ROOT)
    print(
        json.dumps(
            {
                "conclusion": summary["conclusion"],
                "branches": summary["branches"],
                "parameter_count": summary["parameter_count"],
                "total_training_tokens": summary["total_training_tokens"],
                "passed_discovery_gate": summary["passed_discovery_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
