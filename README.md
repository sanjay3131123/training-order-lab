# Training Order Lab

**Does the training data that helps a language model now also help it learn later?**

A controlled PyTorch research project by Sanjay Kumar investigating delayed effects of training-data selection. Built a 2.53M-parameter byte-level Transformer, branched training from identical checkpoints, and compared immediate performance with performance after identical follow-up training.

**36 controlled branches · 6 random seeds · 72.1M training tokens · 4 data domains**

## Problem and approach

Choosing data by its immediate loss improvement can overlook benefits that emerge later. This project tests that premise before investing in a learned curriculum controller.

```text
Shared parent checkpoint (model + optimizer + RNG + sampler)
    ├── Science/math intervention ──┐
    ├── Balanced intervention ──────┤
    └── Other domain interventions ┤
                                   ↓
                      Identical continuation batches
                                   ↓
                   Compare immediate vs. future rankings
```

The implementation includes deterministic data sampling, exact-resume checkpoint tests, atomic result writes, SHA-256 parent/data-stream audits, and evaluation split guards. A separate-corpus screen tests whether discovery results survive new seeds and source families.

## Results

| Experiment | Design | Recorded result |
|---|---|---|
| Discovery | 30 branches, 3 seeds, 2 checkpoints, 5 interventions | Science/math was initially worse than balanced, then better by **0.018563 bits/byte** after 1M continuation tokens at the early checkpoint. |
| Independent-source screen | 6 branches, 3 new seeds, 2 interventions | All 3 seeds reproduced the ranking reversal; mean terminal advantage **0.006721 bits/byte** after 2M continuation tokens. |
| Experimental controls | Matched parent weights and continuation streams | All recorded paired groups passed both hash-based checks. |

The screen's improvement corresponds to approximately **0.46% lower per-byte perplexity** (`1 - 2**(-0.006721)`). It was **below the predeclared 0.010 bits/byte practical-effect threshold**, so the screen formally failed and a learned controller was not built. This is evidence of a small data-order effect in this setting, not proof of improved general learning ability or large-model gains. Domain specialization remains an alternative explanation; code-domain performance regressed slightly.

Read the [discovery report](artifacts/nlrr1-discovery-v1/PHASE1_REPORT.md), [replication report](artifacts/nlrr1-replication-screen-v1/REPLICATION_SCREEN_REPORT.md), and [methodology](PLAN.md). Historical reports retain their original chronological conclusions; the replication result above supersedes the discovery-stage recommendation to continue.

## Verify the recorded evidence — no training required

Requires Python 3.10+; this command uses only the standard library, makes no network requests, and does not download model weights or data:

```bash
python scripts/verify_results.py
```

It recomputes token totals, paired checkpoint/continuation checks, per-seed replication contrasts, and the practical-effect decision directly from the 36 included branch records. These checks validate internal consistency, not independent reproduction of training.

## Run locally

From the repository root, create and activate a Python virtual environment, then:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/prepare_data.py --config configs/phase1.json
python scripts/run_phase1.py --config configs/phase1.json
python scripts/analyze_results.py --config configs/phase1.json
```

For the independent-source replication screen:

```bash
python scripts/prepare_replication_data.py --config configs/replication_screen.json
python scripts/run_replication_screen.py --config configs/replication_screen.json
python scripts/analyze_replication_screen.py --config configs/replication_screen.json
```

Preparation downloads public data and needs network access. Training uses CPU by default; historical discovery and replication took approximately 3.01 and 0.87 hours respectively on the original machine, not a hardware-independent benchmark. Generated data and checkpoints go to the gitignored `.cache/training-order-lab` directory. Analysis regenerates the corresponding reports and metrics, so use a separate clone if preserving the checked-in snapshot unchanged.

## Data and reproducibility

| Domain | Discovery sources | Replication sources |
|---|---|---|
| Prose | WikiText-2 mirror | Project Gutenberg |
| Science/math | OpenAlex abstracts | OpenStax calculus |
| Code | CPython standard library | Go source |
| Dialogue/Q&A | RedPajama StackExchange sample | Databricks Dolly 15k |

Raw datasets are not redistributed. Upstream licenses and access conditions apply; live APIs and unpinned upstream content can change, so fresh downloads are not guaranteed to reproduce the historical corpus byte-for-byte. The published metrics preserve the recorded run, and new preparations generate source manifests and hashes locally. Evaluation uses the development splits; permanent meta-test and final-replication splits remained sealed.

## Repository scope

This is a curated standalone release of the training-order study, not the entire broader research workspace. It includes the model, training/data/analysis code, portable configurations, tests, and complete discovery/replication branch metrics. It excludes personal datasets, journals, cloud launchers, billing records, credentials, checkpoints, and unrelated follow-up experiments. Local checkpoint-location fields were removed from published metrics; numeric results and audit hashes were preserved.

See [SECURITY.md](SECURITY.md) for publication safeguards and checkpoint safety. No predictive curriculum controller or performance optimization beyond the recorded experiments is claimed.

## Automated checks

The local test suite covers exact checkpoint recovery, synthetic positive/negative controls, recorded-result consistency, tamper detection, publication scanning, and evaluation-split access guards. The standalone release additionally rejects unknown access-purpose values to prevent bypassing those guards; this hardening does not change the historical numerical results.

A ready-to-enable [GitHub Actions template](docs/ci-template.yml) runs the publication audit, offline verification, and tests on CPU. It is provided as a template rather than an active workflow because the publishing account's token does not grant workflow-write permission. Copy it to `.github/workflows/ci.yml` through an appropriately authorized GitHub session to enable CI.
