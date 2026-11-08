from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nlrr.sanity import run_sanity_checks, synthetic_world  # noqa: E402


class HarnessTests(unittest.TestCase):
    def test_inserted_rank_reversal(self) -> None:
        self.assertTrue(synthetic_world(prerequisite_matters=True)["reversal"])

    def test_negative_control(self) -> None:
        self.assertFalse(synthetic_world(prerequisite_matters=False)["reversal"])

    def test_checkpoint_and_split_controls(self) -> None:
        report = run_sanity_checks()
        self.assertEqual(report["status"], "pass", report)


if __name__ == "__main__":
    unittest.main()
