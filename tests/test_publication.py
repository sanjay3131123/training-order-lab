from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from check_publication import findings
from verify_results import read_json, verify
from nlrr.data import Corpus


class PublicationTests(unittest.TestCase):
    def test_recorded_results_are_consistent(self):
        self.assertEqual(verify()["verification"], "pass")

    def test_tampered_results_are_rejected(self):
        def tampered(path):
            value = copy.deepcopy(read_json(path))
            if path.name == "replication_screen_results.json":
                value["mean_science_primary_advantage_bpb"] += 0.1
            return value

        with patch("verify_results.read_json", side_effect=tampered):
            with self.assertRaisesRegex(ValueError, "Summary effect mismatch"):
                verify()

    def test_secret_patterns_and_binary_files(self):
        fake_token = ("gh" + "p_" + "a" * 36).encode()
        self.assertIn("github-token", findings("example.txt", fake_token))
        self.assertIn("non-text-file", findings("example.txt", bytes([255])))
        self.assertIn("unexpected-file-type", findings("weights.pt", b"data"))
        self.assertEqual(findings("README.md", b"Public research results."), [])

    def test_unknown_access_purpose_cannot_bypass_split_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
            corpus = Corpus(root, ["prose"])
            with self.assertRaisesRegex(ValueError, "Unknown corpus access"):
                corpus.tokens("prose", "meta_test", purpose="typo")
            with self.assertRaises(PermissionError):
                corpus.tokens("prose", "meta_test", purpose="discovery_eval")


if __name__ == "__main__":
    unittest.main()
