# CoSyDelay V15: symbolic-R9 candidate

V15 preserves the V14 search, prompt, paper Training Fitness, P10/G10 budget,
L-BFGS-B with ten restarts, and approach parallelism. It changes one physical
interpretation: R9 is the paper-stated symbolic condition that fitted delay
approaches positive infinity as `GR_phase -> 0+`. The numerical value at
`GR_phase=1e-6` is retained only as a diagnostic and is not required to exceed
an arbitrary 10,000-second threshold.

The ten optimizer restarts are selected solely by minimum Training MSE. No R9
probe is allowed to override a lower-MSE restart; the fitted winner still has
to pass the symbolic R9 and all remaining enhanced physical checks.

Training-only P3/G2 API pilots on I1 and I2 improved Training R2 from
0.812295/0.766248 (V14) to 0.816095/0.783309 and reduced wall time from
350.1/616.7 s to 257.6/462.1 s. Both winners passed the complete enhanced
physical audit. These two pilots support a full confirmation run but do not by
themselves promote V15 to the paper method.

Formal promotion requires fresh from-scratch P10/G10 Training searches on
I1--I6, followed by the predeclared evaluation protocol. Validation and Test
must not enter expression generation, coefficient fitting, evolution, or
variant selection.
