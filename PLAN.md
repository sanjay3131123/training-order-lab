# Experimental design and interpretation

This document summarizes the original NLRR-1 protocol for the standalone release; it is not a newly preregistered experiment.

## Discovery

- Byte-level causal Transformer: 2,525,600 parameters, four blocks, four attention heads, embedding width 224, context length 256; AdamW training.
- Seeds: 1103, 2207, 3301. Parent checkpoints at nominal 2M and 5M trunk tokens.
- Five actions: prose, science/math, code, dialogue/Q&A, balanced. Each intervention receives nominally 250K tokens.
- Each matched group then receives identical balanced continuation, evaluated at 250K, 500K, and 1M tokens.
- Discovery utility normalizes domain bits/byte against balanced-control seed variability, with a worst-domain penalty and scale floor. Candidate gates require consistent reversal across all three seeds and at least two horizons, plus a normalized noise gate.
- Two related candidate pairs passed at the early checkpoint; none passed at the late checkpoint. Discovery comparisons are exploratory, not confirmatory multiple-testing-adjusted claims.

## Independent-source R1 screen

- New seeds: 4409, 5501, 6607; new source families across all four domains.
- Science/math versus balanced, from the nominal 2M checkpoint; 250K-token intervention.
- Primary horizon: 2M identical continuation tokens. Supporting horizon: 1M.
- Require all seeds to reverse, mean primary advantage at least 0.010 bits/byte, and no domain mean regression worse than 0.020 bits/byte.
- Observed primary advantage: 0.006721 bits/byte. The practical-effect gate failed; the planned larger R2 confirmation was not run.
- The paired 95% t interval is descriptive with only three seeds and is not the decision rule. A positive interval does not override the practical-effect threshold.

## Controls and limitations

Branches share exact parent weights and continuation batch sequences, checked with hashes. The checkpoint harness tests restoration of optimizer, RNG, and sampler state. Positive/negative synthetic controls check detection logic, not real-world efficacy. Splits beyond development were kept sealed.

Token budgets are rounded to whole batches; recorded totals count actual tokens, including each seed's shared trunk once. The discovery and screen together use 72,124,416 training tokens across 36 branches and six distinct seeds.

Results apply to a small model, these data sources, and these budgets. They do not establish general capability gains, transfer, optimal data order, or a working predictive curriculum algorithm. The public snapshot deliberately retains the unsuccessful practical-effect decision alongside the qualitative reversal.
