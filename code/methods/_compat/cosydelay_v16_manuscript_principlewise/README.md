# CoSyDelay V16: manuscript principle-wise candidate

Status: `P3/G2 API pilots complete; P10/G10 I1--I6 confirmation pending`.

V16 is an isolated candidate for the recovered 2026-07-11 revised manuscript.
It retains V15's Training-only P10/G10 evolution, L-BFGS-B with ten restarts,
minimum-Training-MSE restart choice, positive log-coordinate coefficients, and
at most four approach workers.  It changes only the physical method contract:

- the verifier uses the manuscript's ordered R1--R7 schema;
- R2, R3, and R6 contribute grid pass fractions;
- R5 requires a finite nonnegative low-demand limit but does not require exact
  zero delay;
- R7 requires a symbolic positive-infinite zero-green limit and has no numeric
  10,000-second threshold;
- R4 uses exact symbolic degree-one homogeneity in `Cycle_Time` while keeping
  fitted coefficients symbolic and dimensionless; it does not replace a
  fitted time exponent by one;
- the physical component is the equal mean over every declared
  movement-by-seven-rule score and enters Fitness directly;
- strict joint pass is reported as a diagnostic and is not a hard admission
  gate.

Before fitting, V16 also enforces the prompt grammar literally: coefficient
identifiers are consecutive from `a1` and one coefficient cannot serve
simultaneously as a fitted power exponent and an `exp()` coefficient. This
makes the three frozen bound classes executable without a hidden precedence
rule; it adds no physical score and does not change Fitness.

Each mutation receives its current parent plus the compact R1--R7 physical
score vector and repair hints only for partial/failed rules.  No numerical
Training, Validation, or Test metric is placed in the LLM prompt; Training
Fitness remains an external programmatic selection signal.

The ten optimizer starts are now disclosed exactly. An initialization
candidate uses nine retained local starts and one deterministic role-aware
Sobol wide start. If a mutation shares coefficient names with its immediate
parent, one of those ten slots uses the parent's fitted values from this same
intersection and Training-only run, followed by eight retained local starts
and one Sobol start; otherwise it uses the initialization composition. The
warm start replaces a slot rather than adding an eleventh optimization and no
external-run coefficient or incumbent is accepted.

The available I1--I6 files contain approach-delay targets only.  Accordingly,
the accuracy component is the mean nonnegative Training R2 over observed
approaches.  Movement-specific coefficients are jointly calibrated through
the flow-weighted approach predictions; they are not independently supervised
by unavailable movement-delay labels.  The manuscript must use this exact
data/aggregation statement unless movement-level labels are supplied.

The candidate domain follows the already written reviewer-workstream proposal:
`flow_lane=[0.001,2.0]`, `GR_phase=[0.02,0.95]`, and
`Cycle_Time=[30,240]` seconds, with 128 deterministic Latin-hypercube points,
seed 42, all eight joint domain corners, and the retained non-corner high-flow
anchor `(2, 0.2, 120)`.  After deduplication, each movement has 137
finite-domain numerical check points.  A separate `(0, 0.5, 120)` point
supports only the analytic low-demand boundary and is not counted among those
137 points.  This
covers all currently audited Training support, but the
author must freeze the physical meaning and domain before a formal run.

The shared runtime's retained seven-rule private function currently fails
because an eight-rule refactor changed its global key order.  V16 repairs that
boundary in an isolated, locked adapter and serializes its own immutable schema
instead of modifying V15's shared runtime during the active confirmation.

V16 reuses one isolated symbolic-limit worker for the duration of a search.
The operation remains `sympy.limit`, the per-request timeout remains 8 s, and
a timeout terminates the worker before the next request.  An exact
five-expression replay matched every physical score and feedback item while
reducing verifier time from 60.86 s to 3.35 s (18.17x for the verifier stage,
not for the full method).

R5/R7 are scored for every fitted movement component.  The redundant generic
positive-coefficient calculations are not executed because special fitted
coefficient equalities can change a limit.  V16 substitutes the full fitted
coefficients as exact decimal rationals and evaluates each required limit in
the same isolated 8-second worker.  It never uses the retained
two-significant-digit symbolic copy; an undecidable or timed-out fitted limit
fails closed for that rule.

The isolated foundation passes its focused unit and integration tests.  A
no-LLM I1 Training-only smoke fitted the fixed positive-low-demand-limit form
`Cycle_Time*(a1+a2*flow_lane/GR_phase)` with the retained ten-restart
optimizer.  The first smoke exposed unused legacy numeric-R9 fields in the
serialized compatibility config and is superseded.  The clean rerun removes
those fields entirely; neither a numerical probe nor a 10,000-second threshold
belongs to the V16 schema.  This is execution evidence, not an accuracy or
promotion experiment.

After the fitted-limit correction and removal of the redundant generic check,
the final smoke retained Training accuracy 0.733971, physical score 1.0, and
Fitness 1.733971.  It completed coefficient fitting in 3.51 s and the complete
12-movement seven-rule check in 2.49 s.  Total smoke time was 6.45 s.  The
standalone five-expression replay measured the persistent worker's larger
benefit after startup: identical decisions and scores with verifier time
falling from 60.86 s to 3.35 s (18.17x).

The P3/G2 screening artifacts predate the final diagnostic cleanup: their
retained fitter computed and logged the legacy near-zero-green probe, but
`r9_constrained_restart_selection=false`, so it never selected a restart or
changed those pilot metrics.  The P10/G10 V16 adapter now omits that probe
entirely while preserving the same MSE objective, ten starts, analytic
gradient, log coordinates, coefficient bounds, and approach worker pool.
Formal history validation rejects any leaked legacy probe field.

Population update uses global \((\mu+\lambda)\) elitism: the ten parents and
ten new offspring are canonically deduplicated, ranked first by manuscript
Fitness, then only on exact ties by raw Training R2 and lower Training RMSE,
and truncated to ten survivors.  No feasible archive, early stopping, or wall
budget is enabled in the formal runner.

The inherited epsilon-parsimony output is a post-search diagnostic only.  If
no evaluated candidate has a strict seven-rule joint pass, V16 records this
diagnostic as unavailable instead of raising or replacing the manuscript-
Fitness winner.  Expression complexity never participates in survivor or
final-winner ordering.

Before fitting, V16 filters only parser/grammar violations, sampled numerical
legality hazards, non-consecutive coefficient identifiers, and ambiguous
nonlinear coefficient roles.  Log arguments and bases carrying a fitted
exponent are screened at fixed boundary/interior points with every positive
coefficient represented by 1; denominator bases are screened on a 65-by-65
grid over the declared finite domain at coefficient 1 and cycle 120 s.  The
`flow_lane=0` point is used only to protect the analytic low-demand boundary,
not as a fitted data point.  These inexpensive checks are not global proofs.
V16 does not reject a candidate for failing any of R1--R7 before fitting;
those principles are scored after calibration through Fitness.  This routing
is installed explicitly inside V16 rather than depending on inherited wrapper
order.

For R2 and R3, SymPy differentiates the fitted expression analytically, but
the score is the fitted derivative-sign pass fraction on the fixed 137-point
finite-domain grid; it is not a whole-domain symbolic sign proof.  R6 is the
finite, nonnegative fitted-value fraction on the same grid, with non-finite
values counted as failures.  R1, R4, R5, and R7 remain binary structural,
exact-homogeneity, or analytic-limit checks.

Every successful formal or sensitivity API call archives the instantiated prompt and the
normalized assistant-message content together with matching SHA-256 hashes.
The formal wrapper now fails closed if either text or hash is missing or if an
API-key/authorization field or the configured secret appears in the record.
This is not a byte-for-byte archive of the complete provider response envelope.
Across the completed I1/I2 P3/G2 pilots, all 18 successful HTTP-200 records
returned `response_model=gpt-4.1-mini`; no record exposed a dated or otherwise
snapshot-specific identifier.  Accordingly, V16 reports both the requested
and returned alias but leaves the exact provider snapshot explicitly unknown.
Programmatic novelty and retry instructions are inserted before the immutable
output block, so the exact output schema is always the final prompt section.

The executable contract freezes P10/G10.  The recovered manuscript's
reproducibility table still says `m=20`; it must be changed to `m=10` for this
protocol or supported by a separately declared P20 rerun.  Prospective
Training-only P3/G2 pilots have run on I1 and I2; they are screening evidence,
not promotion evidence.  V15 remains the
original binary-method confirmation and must not be relabeled as V16 evidence.

The formal six-intersection launcher now writes a source-and-Training hash
freeze before its first API call and verifies every completed result against
that freeze.  It also recomputes the bounded Training Fitness and rejects any
mismatch, failed Prompt-contract flag, incomplete Prompt/response hash audit,
or optimizer restart-composition drift.  The launcher additionally rebuilds
the seven rule means, the overall movement-by-rule physical mean, and strict
joint-pass flag from the serialized movement cells, and requires a complete
finite positive coefficient vector for every declared movement.  Only after
all six P10/G10 Training
searches pass these integrity checks may `evaluate_frozen_test.py` open a Test
file.  That evaluator performs
prediction only: it neither regenerates an expression nor refits/reselects a
coefficient.  It writes the same sample-level CSV columns used by the fixed 17
raw-feature baselines and ranks pooled Test R2/RMSE/MAE by the same
intersection-wise convention.  The exact fixed-17 summary is precommitted in
the frozen evaluator by SHA-256
`2263c47a80a263a11f71123578d51a24b45752aa03580894a00e8a7cc86b948a`;
a different same-format table is rejected.  Because the project's existing Test files were
seen during earlier development, these safeguards prevent within-run leakage
but do not turn those files into a newly blind external test set.

The V16 source manifest contains the complete loaded local import closure for
the formal Training launcher and Test-only evaluator: 96 files, with zero
loaded local Python modules left outside the manifest in the pre-freeze audit.
Package entry points and inherited Prompt/diagnostic adapters are included as
well as the direct V16 modules.

I1 obtained R2 0.816809, RMSE 4.638588, MAE 3.131606, physical score 1.0,
and wall 160.36 s.  I2 obtained R2 0.777931, RMSE 5.801311, MAE 4.145007,
physical score 1.0, and wall 140.09 s.  Against V15, V16 was slightly better
at I1 and lower by 0.00538 R2 at I2, while using fewer API attempts and much
less wall time.  I1--I6 P10/G10 must determine whether this generalizes.
