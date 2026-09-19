# Frozen V22 oversaturated same-layout generalization test

This experiment uses only strict additional rows with maximum degree of
saturation x >= 1.0. All exact model-input overlaps with the locked
Training/Validation/Test union and five near-duplicate sensitivity rows
were removed upstream. V22 was frozen; no refitting, selection, or tuning
used these rows.

This is a same-layout oversaturation stress test, not unseen-intersection
or external validation. Confidence intervals are percentile intervals from
2,000 timing-plan cluster bootstrap draws.

## Main results

| Scope | N | Plans | R2 | 95% CI | RMSE | MAE | Bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| I1 | 749 | 532 | 0.9624 | [0.9571, 0.9671] | 1.8702 | 1.3673 | -0.0722 |
| I2 | 748 | 530 | 0.9306 | [0.9145, 0.9441] | 3.1436 | 2.1627 | 0.1882 |
| I3 | 514 | 352 | 0.9132 | [0.8969, 0.9269] | 2.3364 | 1.7873 | -0.1355 |
| I4 | 678 | 165 | 0.9411 | [0.9249, 0.9535] | 0.8579 | 0.5588 | 0.1116 |
| I5 | 420 | 252 | 0.9039 | [0.8814, 0.9221] | 3.2149 | 2.3058 | -0.1877 |
| I6 | 535 | 47 | 0.9500 | [0.9275, 0.9647] | 0.5710 | 0.3625 | 0.0300 |
| Macro-I1-I6 | 3,644 | 1,878 | 0.9335 | [0.9263, 0.9390] | 1.9990 | 1.4241 | -0.0109 |
| Pooled-I1-I6 | 3,644 | 1,878 | 0.9744 | [0.9704, 0.9782] | 2.2121 | 1.4000 | 0.0082 |

## Coverage

| Intersection | Oversaturated rows | Severe x >= 1.3 | Unseen timing rows |
|---|---:|---:|---:|
| I1 | 749 | 674 | 358 |
| I2 | 748 | 691 | 367 |
| I3 | 514 | 492 | 189 |
| I4 | 678 | 590 | 78 |
| I5 | 420 | 410 | 89 |
| I6 | 535 | 376 | 4 |

Pooled R2 is secondary because between-intersection delay-scale
differences can raise it. Use the per-intersection and macro results as
the primary evidence.
