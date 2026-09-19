# Prospective optimizer conditioning v1

This directory is an isolated Training-only ablation.  It does not modify or
replace the frozen `cosydelay_v9_clean_from_scratch` method.

The staged questions are:

1. Does using log coordinates for every strictly positive coefficient improve
   convergence or Training accuracy while retaining L-BFGS-B, ten restarts,
   the same raw restart points, the same MSE objective, and the same bounds?
2. Which generated expressions reuse one coefficient in incompatible nonlinear
   roles such as both a fitted power and an `exp` coefficient?
3. After rejecting those ambiguous expressions, do predeclared nonlinear-role
   bounds improve accuracy, convergence, boundary behavior, or runtime?

No Validation or Test file may be opened by this experiment.  Test performance
must not be used to select an arm.

Current status and numerical decisions are recorded in `STATUS.md`. Nothing in
this directory is part of the frozen formal V9 method unless a later fresh
end-to-end Training-only confirmation passes and the method is explicitly
versioned and frozen.
