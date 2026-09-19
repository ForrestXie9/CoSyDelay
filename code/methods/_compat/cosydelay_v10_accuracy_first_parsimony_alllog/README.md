# CoSyDelay V10 prompt-v6 + efficient physical-audit execution

This is a prospective candidate confirmed by Training-only P3/G2 pilots on I1
and I2. It is not yet a formal P10/G10 paper result and it is not a Test result.
The frozen V9 directory is unchanged.

The candidate retains the paper Training Fitness, hard physical admission,
Training-only evolution, no external incumbent, P10/G10 target budget,
L-BFGS-B, ten retained raw restart points, parent inheritance in one existing
restart slot, and up to four approach workers.

Its prospective changes are:

1. all ten positive-parameter fits use log coordinates; and
2. the prompt requests the minimum complexity needed to capture delay changes,
   without prescribing a coefficient count. Large mutation replaces,
   reorganizes, merges, or simplifies traffic mechanisms. `family_unique`
   rejects coefficient-renamed copies of the same structural family.
3. the final initialization and mutation prompts follow the concise structure
   shown by the paper-era implementation. Mutation receives its current parent
   plus Training-only Fitness/R2/RMSE/MAE and physical feedback. Validation and
   Test remain sealed. Program-side duplicate checking still uses the full
   archive, while no more than ten representative expressions are printed in
   one LLM prompt.
4. execution contract `v6_equivalent_audit_reuse_family_cache_v1` reuses the
   already-computed standard fitted audit and the coefficient-robust R8/R9
   endpoint proof in the enhanced audit. A sufficient fixed-inverse-green R9
   certificate avoids isolated SymPy limits only when it can prove the same
   positive-infinity result; otherwise the original proof remains the fallback.
5. after a fitted physical failure, coefficient-renamed copies of that rejected
   structural family are rejected before another ten-restart fit. The retry
   receives the compact failed-rule repair guidance.

The frozen prompt contract is
`paper_structured_clean_training_feedback_v6`. Its manuscript-display structure
is recorded in `PROMPTS.md`; the executable source of truth is
`final_prompt.py`.

## P3/G2 confirmation (engineering evidence only)

| Intersection | Training R2 | pooled RMSE | pooled MAE | physics | attempts | wall s |
|---|---:|---:|---:|:---:|---:|---:|
| I1 | 0.781369 | 5.077177 | 3.454800 | pass | 18 | 393.6 |
| I2 | 0.723970 | 6.423566 | 4.674853 | pass | 14 | 377.5 |
| Mean | 0.752669 | 5.750372 | 4.064827 | 2/2 pass | 16 | 385.5 |

These wall times are the pre-efficiency baseline. Offline replay of all 32 v6
prefit attempts and all eight archived fitted cases with recoverable parameter
sets produced 32/32 and 8/8 identical physical pass/fail decisions. Twenty-one
of the 32 expressions used the safe fast R9 certificate. The replay artifact is
`experiments/v6_efficiency_equivalence_replay.json`.

## Efficiency-v1 confirmation

The deterministic offline P2/G1 smoke selected the same expression with the
same Training R2 (0.578454) and a physical pass. Wall time decreased from 57.9
to 18.3 seconds (68.4%). A fresh real-API I1 P3/G2 run completed in 189.1
seconds versus the earlier 393.6-second v6 run (52.0% less wall time); its
Training R2 was 0.758899, pooled RMSE 5.332743, pooled MAE 3.664096, and all
physical rules passed. Because the API generated a fresh candidate sequence,
the two live accuracies are not a paired causal comparison; the archived
same-candidate replay is the equivalence evidence.

The earlier long accuracy-first prompt pilots averaged Training R2 0.638362,
pooled RMSE 6.909902, and pooled MAE 4.878895 on the same I1/I2 P3/G2 scope.
The v6 pilots therefore support preserving performance after cleanup, but they
do not replace the required fresh P10/G10 confirmation.

The simplest physical candidate within 0.002 Fitness of the best may be
reported as a diagnostic, but it does not change evolution or final selection.
