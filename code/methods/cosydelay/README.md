# CoSyDelay modules

```text
run.py           Command-line entry point
pipeline.py      Validation selection, coefficient refit, and final testing
prompt.py        Expression-generation prompts
regeneration.py  Candidate regeneration
fitter.py        Movement-specific coefficient calibration
physics.py       Physical-consistency checks
range_policy.py  Coefficient bounds
engine/          Search and numerical implementations
support/         Data loading, expression parsing, and LLM interface
```

Start with `run.py` to follow a complete run, or `pipeline.py` for the data-split and final-model workflow.

[Installation and usage](../../../README.md) · [Evaluation protocol](../../../docs/PROTOCOL.md)
