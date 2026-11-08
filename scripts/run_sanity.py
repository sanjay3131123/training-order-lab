from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.sanity import run_sanity_checks  # noqa: E402
from nlrr.utils import atomic_json, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    report = run_sanity_checks()
    output = ROOT / "artifacts" / config["experiment_id"] / "stage_minus_one_report.json"
    atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
