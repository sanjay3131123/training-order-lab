# NLRR-1 rapid replication screen

Status: **FAIL**

Do not advance to R2; downgrade the long-horizon data-selection arm.

## Primary result

Science/math's mean immediate penalty was 0.190902 bits/byte. Its mean advantage after 2,000,000 identical continuation tokens was 0.006721 bits/byte, corresponding to approximately 0.47% better per-byte perplexity factor.

Descriptive paired 95% t interval: [0.006533, 0.006910] bits/byte. This interval is reported but is not the predeclared R1 decision rule.

## Per-seed results

| Seed | Immediate penalty | Advantage at 1M | Advantage at 2M | Reversed |
|---:|---:|---:|---:|:---:|
| 4409 | +0.196741 | +0.000477 | +0.006640 | yes |
| 5501 | +0.187025 | +0.001666 | +0.006735 | yes |
| 6607 | +0.188939 | +0.006303 | +0.006789 | yes |

## Gate checks

- causal_audit: pass
- all_seeds_reverse: pass
- all_seeds_better_at_supporting_horizon: pass
- all_seeds_better_at_primary_horizon: pass
- minimum_mean_terminal_advantage: fail
- no_large_domain_regression: pass

## Held-out domain effects at 2M

- prose: +0.003447 bits/byte
- science_math: +0.029744 bits/byte
- code: -0.008398 bits/byte
- dialogue_qa: +0.002093 bits/byte

Positive values favor science/math. The meta-test and final-replication splits remained sealed.

## Scope

The screen used 2,525,600 parameters and 19,519,488 training tokens in 0.87 local hours. A pass permits R2; it does not establish transfer or an architectural breakthrough.
