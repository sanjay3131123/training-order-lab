from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.experiment import run_phase1  # noqa: E402
from nlrr.utils import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    results = run_phase1(load_config(arguments.config), ROOT)
    print(f"Completed {len(results)} branch records.", flush=True)


if __name__ == "__main__":
    main()
