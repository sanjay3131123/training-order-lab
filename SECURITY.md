# Security and publication scope

The initial release is an allowlisted export of source code, configurations, tests, and aggregate/branch metrics. It does not contain the original workspace history, raw training text, personal records, cloud account records, model checkpoints, or credentials. Machine-specific checkpoint paths were removed. Publication checks inspect the tracked tree; automated checks are safeguards, not a guarantee against every possible secret format.

Before committing new results, inspect `git diff --cached`, run `python scripts/check_publication.py`, and review unexpected files manually. Never add credentials or private datasets, even temporarily: deleting a file later does not remove it from Git history.

The training harness loads its own trusted checkpoints with PyTorch's full serialization support to restore optimizer and RNG state. **Do not load checkpoints from untrusted sources.** Dataset preparation downloads third-party public content; review upstream terms and use an isolated virtual environment.

If a credential is ever exposed, revoke/rotate it before rewriting history. Do not paste secret values into public issues.
