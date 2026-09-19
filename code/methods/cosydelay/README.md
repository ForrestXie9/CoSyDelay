# CoSyDelay experimental overlay

This directory is the final CoSyDelay method. Every outer restart uses the
validated global-selection P10/G10 search and the same coefficient fitter; only the
reproducible seed changes.  Initialization is unchanged.  Regeneration adds
one concise request for a structurally distinct, rather than minimally edited,
compact and traffic-physically interpretable expression.

The coefficient optimizer remains all-positive log-coordinate L-BFGS-B.  Its
role-defined ranges are fixed before any CoSyDelay search and do not read
Validation/Test: scale `(0.001, 1000)`, power exponent `(0.01, 8)`, and
exponential coefficient `(0.00001, 2)`.  This profile won all six paired
Training fits on the I1/I3/I5 three-site, two-new-seed screen; the complete
artifact is retained under `experiments/range_confirm_i135_*`.

Run the stages in order: `launch_uniform_restarts` (Training only),
`select_uniform_restarts_validation` (Validation only), then
`refit_train_validation` (Training+Validation refit followed by one Test
evaluation).
