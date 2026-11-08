# Phase 1 report: Natural Long-Horizon Rank Reversal

Experiment: `nlrr1-discovery-v1`

## Outcome

A natural long-horizon rank-reversal candidate passed the predeclared discovery gate. It requires six-seed, separate-corpus replication before interpretation.

The run completed 30 paired branches using 2,525,600 parameters and 52,604,928 training tokens (approximately 7.972e+14 training FLOPs) in 3.01 hours of local wall time.

All paired branches passed the exact-parent and identical-continuation-stream audit.

## Passed discovery candidates

- At 2,000,000 trunk tokens: `dialogue_qa` looked better immediately, while `science_math` was better after horizons [250000, 500000, 1000000]. In raw macro bits-per-byte, `science_math` was 0.083989 worse immediately but 0.020616 better at 1,000,000 continuation tokens (approximately 1.44% better per-byte perplexity factor).
- At 2,000,000 trunk tokens: `balanced` looked better immediately, while `science_math` was better after horizons [250000, 500000, 1000000]. In raw macro bits-per-byte, `science_math` was 0.176144 worse immediately but 0.018563 better at 1,000,000 continuation tokens (approximately 1.29% better per-byte perplexity factor).

## Aggregate rankings

- Checkpoint 2,000,000, immediate: balanced (-0.176), dialogue_qa (-14.792), code (-15.348), science_math (-18.753), prose (-30.638)
- Checkpoint 2,000,000, continuation_250000: science_math (+0.038), balanced (-0.148), prose (-0.984), code (-1.015), dialogue_qa (-1.756)
- Checkpoint 2,000,000, continuation_500000: science_math (+1.146), prose (+0.424), balanced (-0.137), code (-0.437), dialogue_qa (-0.825)
- Checkpoint 2,000,000, continuation_1000000: science_math (+1.755), prose (+0.335), balanced (-0.191), code (-0.218), dialogue_qa (-0.574)
- Checkpoint 5,000,000, immediate: balanced (-0.142), science_math (-7.778), prose (-7.814), code (-8.922), dialogue_qa (-9.121)
- Checkpoint 5,000,000, continuation_250000: balanced (-0.098), science_math (-0.377), prose (-0.607), dialogue_qa (-0.961), code (-1.272)
- Checkpoint 5,000,000, continuation_500000: balanced (-0.101), science_math (-0.120), prose (-0.210), dialogue_qa (-0.501), code (-0.700)
- Checkpoint 5,000,000, continuation_1000000: science_math (+0.003), balanced (-0.066), prose (-0.100), dialogue_qa (-0.568), code (-0.648)

## Developmental-state result

Both strict candidates occurred at the 2M-token checkpoint. No action pair passed the full three-seed, multiple-horizon, noise-threshold gate at the 5M-token checkpoint. Immediate-to-future rank correlations were substantially less stable at 2M than at 5M, suggesting the detected effect is concentrated earlier in development for this model and horizon.

## Interpretation limits

This is a tiny-model discovery experiment. A passing candidate is not a breakthrough claim and must pass the six-seed, longer-horizon, separate-corpus or larger-scale replication gate in PLAN.md. A null result only applies to these domains, intervention lengths, checkpoints, model size, and forward-token objective.

The permanent meta-test and replication splits were prepared and hashed but were not loaded by the Phase 1 runner.
