# Training and evaluation protocol

This document describes the public [runner](../code/methods/cosydelay/run.py) and its [default configuration](../configs/cosydelay.json).

## Data splits

The six intersections use fixed Training (5,909 samples), Validation (1,512), and Test (2,853) partitions. File identities, sample assignments, and split seed `20260712` are recorded in [the split directory](../data/locked_splits/).

Training and Validation remain separate throughout structure search and expression selection. They are combined only for the final coefficient refit.

## Search and final expression selection

Candidate coefficients are fitted on Training only. Training Fitness drives evolutionary feedback and population updates:

```text
Fitness_train = mean_a(max(0, R2_train,a)) + Physical_train
```

Here `a` is a valid approach. The physical component is binary: 1 only when all seven physical checks pass for every included movement, otherwise 0.

After search, the actual final population of `P` survivors and their Training-fitted coefficients are saved before Validation is opened. Validation evaluates only those survivors, including any retained parents from earlier generations. It does not use the whole search archive or the last `P` offspring.

The final expression maximizes:

```text
Fitness_val = mean_a(max(0, R2_val,a)) + Physical_train
```

Validation does not refit coefficients. Ties are resolved by lower direction-average Validation RMSE, then lower candidate ID. Pooled R2 is not used for selection.

The configured population size determines `P`. The generation setting counts total batches, including initialization. An archive without its recorded final survivors is rejected rather than assigned an inferred population.

## Coefficient refit and physical recheck

Once the expression structure is selected, only its movement-specific coefficients are refitted on Training + Validation. No further structure search takes place.

All seven physical principles are checked again using the new coefficients. A failed or resource-incomplete check gives a final physical component of 0; it does not inherit the Training score or trigger Test-based reselection. The refitted model is retained.

Numerical fitting or prediction failures fail the run. Rows are not silently dropped, and an earlier model is not substituted.

The final model, physical report, and hashes are persisted before Test is opened.

## Test metrics

R2, RMSE, and MAE are computed separately for each valid approach and then averaged equally. Reported R2 is raw, not clipped; pooled R2 is not used. I4-E and I6-W are the predeclared invalid/constant target directions and are excluded.

Test Fitness clips each approach's R2 below zero and adds the **final refitted model's** physical component:

```text
Fitness_test = mean_a(max(0, R2_test,a)) + Physical_final
```

Test labels never select an expression, coefficient vector, or fallback model.

## Prompt and parsing

The prompt requests text sections headed `### Expression` and `### Explanation`, followed by symbolic parsing and validation of the expression. The HTTP response envelope is JSON; the generated expression content is not required to be a JSON object.

## Saved records and timing

Each run records its effective settings, source revision/hashes, data hashes, final coefficients, and stage timings.

| File | Contents |
|---|---|
| `configuration.json` | Effective configuration and source information |
| `selection_pool.json` | Eligible final survivors and Training coefficients |
| `validation_candidates.json` | Validation scores for the eligible candidates |
| `selection_frozen.json` | Selected structure and selection evidence |
| `final_physics.json` | Physical checks on the refitted coefficients |
| `model.json`, `freeze.json` | Final model and pre-Test freeze evidence |
| `split_access_audit.json` | Data-split access audit |
| `result.json` | Final metrics and stage timings |

Training-search elapsed time includes LLM requests, API waiting/retries, candidate coefficient fitting, and candidate physical evaluation. Selection, final refitting, physical rechecking, and testing are timed separately. Generated runs stay local and are ignored by Git.

When reporting an experiment, identify its source revision, configuration, and split manifest rather than assuming historical results used the current default protocol.
