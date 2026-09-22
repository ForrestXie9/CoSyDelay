# CoSyDelay source modules

The public method is `methods.cosydelay`.

- `methods/cosydelay/` contains the user-facing method package and runner.
- `methods/cosydelay/_runtime/` contains the current data, expression,
  physics, optimization, and LLM support modules.
- `methods/cosydelay/_internal/` contains the private implementation helpers
  used by the final method. It is not a collection of released baselines or
  experiment results.

No search output, history, API key, or reviewer experiment is required from
this directory.
