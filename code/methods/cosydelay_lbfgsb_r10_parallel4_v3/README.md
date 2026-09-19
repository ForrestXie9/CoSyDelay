# CoSyDelay v3 (prospective)

This folder develops accuracy and physical-robustness improvements without
changing or importing results into the retained v2 method directory.

The v3 runtime keeps one optimizer (L-BFGS-B), exactly ten restarts per
approach, and at most four approach workers. It uses the validated analytic
Jacobian, nine retained local starts with one slot exchanged for a role-aware
wide Sobol start, parent coefficient inheritance, selective log coordinates for
the wide exploration start, and a final best-vector
polish. The solver remains L-BFGS-B throughout; there is no hybrid optimizer and
no fixed R9 guard.

The ten optimizer endpoints are treated as a feasible archive: when at least
one endpoint passes the retained numerical R9 probe, the lowest-MSE endpoint
inside that feasible subset is selected. If final polish would turn an already
enhanced-physics-feasible solution into an infeasible one, the pre-polish vector
is retained. This changes neither the expression nor the optimizer.

Structure selection is Training-only. Evolution fits on three inner folds and
scores on the fourth. After evolution, only the top three physically feasible
candidates are refitted in four-fold cross-validation. Their ranking score is

`macro R2 - 0.05 * normalized RMSE - 0.05 * normalized MAE`.

After this ranking is frozen, the winner is refitted on all Training rows and
polished. Full-Training accuracy is reporting-only. Outer validation and Test
must not be supplied to this runtime.

When upgrading an existing retained v2 run, one of the three CV slots is
reserved for its Training-selected physical incumbent; the other two remain
available to newly evolved v3 expressions. The incumbent is not automatically
selected: it competes under the same four-fold Training OOF score and final
enhanced physical audit. This prevents evolutionary forgetting without using
Validation or Test.

The final model must pass the existing eight-rule fitted verifier plus exact
symbolic R8, a symbolic positive-infinity R9 proof, and a deterministic dense
operational-boundary value/derivative audit. Numerical R9 remains useful as a
diagnostic but cannot replace the symbolic proof. Generated expressions are
never modified after generation.

Status: prospective candidate, not an official holdout result. The coefficient
component passed its predeclared Training-only gate on 12 fixed-expression
cases (I1/I4/I6 x four folds): mean delta R2 +0.0037695, RMSE -0.037043,
MAE -0.033931, 12/12 enhanced-physics pass, and worst relative RMSE degradation
+0.00145%. The measured v3 wall time was 10.40 seconds higher per case, although
that timing includes the enhanced audit while the reference fitter timing does
not, so it is reported as a secondary non-like-for-like measurement.

The whole-search Training-only study contains six paired runs (I1/I4/I6 x two
fixed seeds). Mean OOF R2 increased from 0.686997 to 0.779261; mean relative
RMSE and MAE changes were -13.44% and -14.69%. No pair had worse RMSE. All six
v3 winners passed the enhanced audit, versus five of six retained-v2
references. I4 exposed stochastic evolutionary forgetting; reserving one CV
slot for the retained physical incumbent restored its R2 from 0.59573 to
0.81483 without selecting on Validation or Test. Comparable search-plus-OOF
wall time increased by 22.0% using the ratio-of-means calculation.

These are development OOF results, not an external generalization claim. The
implementation is ready to be frozen as a candidate and evaluated once on a
new untouched holdout; it must not be tuned from that holdout result.

The fixed P4/G2 evolutionary budget remains in force. Population/generation
escalation will only be enabled if a Training-only efficiency study gives a
predeclared trigger and benefit.
