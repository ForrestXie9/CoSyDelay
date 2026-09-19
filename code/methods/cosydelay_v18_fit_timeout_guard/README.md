# CoSyDelay V18 - conservative R7 pre-fit gate and bounded fitting

V18 preserves V17's paper Fitness, fitted seven-rule physical score, prompt,
Training-only selection, P10/G10 budget, all-positive-log L-BFGS-B optimizer,
ten restarts, and up-to-four approach workers. V17 remains frozen and is not
edited.

Before coefficient fitting, V18 now rejects one narrow class of physically
impossible formulas: expressions whose complete response is proved finite as
`GR_phase -> 0+` for arbitrary positive finite coefficients, flow, and cycle
time. This is a structure-only R7 check. It reads no Training values or
targets, and no Validation or Test data. If the proof is inconclusive, the
formula is retained and proceeds to bounded fitting and the existing fitted
seven-rule score. The gate does not change Fitness and does not add a guard
term to the generated expression.

On a retrospective structure-only replay of all 660 completed V17 run01
candidates, the gate rejected 19 formulas; every one had fitted R7 = 0, and no
fitted R7 = 1 formula was rejected. All six historical winners were retained.
The check is intentionally conservative and does not attempt to reject every
formula that may fail R7 after fitting.

The bounded fitter addresses an observed V17 run that spent more than six
hours inside one candidate coefficient-fit stage. In completed V17 run01, 660
candidate fits had median 7.16 s, p95 41.27 s, p99 56.23 s, and maximum
84.59 s. Five completed candidates exceeded 60 s. V18 therefore uses a fixed
100 s wall-clock allowance for the complete fit call. This leaves roughly
12 s above the historical maximum after the observed first-supervisor startup
overhead, while stopping a pathological fit sooner than the earlier 120 s
draft threshold.

The allowance covers symbolic derivative construction, restart payload
construction, all approach optimizations, and result collection. The
persistent approach pool lives inside a dedicated supervisor process. On a
timeout or worker failure, V18 terminates the supervisor and all descendants,
starts a clean supervisor, rejects the exact canonical expression, and asks
the existing evolution loop to regenerate the same slot. Parent optimizer RNG
state is committed only after a successful fit. Rejected attempts do not count
toward the exact 110 successful candidates, and timeout is not part of Fitness.

Active launch waits are also bounded: supervisor startup 30 s, supervisor close
5 s, each intersection process 3 h, full six-intersection launcher 19 h, and
prediction-only Test evaluation 30 min. A timeout uses terminate-then-kill
cleanup for the complete process tree and is recorded in the audit.

`run_real_fit_smoke.py` compares the retained direct fitter with the guarded
fitter on one real I1 Training expression using identical optimizer RNG state.
It is an implementation-equivalence check, not accuracy evidence.

`run_offline_integration_smoke.py` runs the actual V18 composition at P2/G1
with deterministic local responses. It exercises generation, pre-fit R7
screening, legality, fitting, physical scoring, evolutionary selection, and
bounded cleanup without an API call or Test access.
