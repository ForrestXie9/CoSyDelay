# CoSyDelay: Cognitive Symbolic Regression for Traffic Delay Estimation

CoSyDelay learns interpretable traffic-delay equations for signalized intersections. It combines LLM-guided symbolic regression, numerical coefficient calibration, and traffic-physics checks.

## Repository

```text
code/methods/cosydelay/   CoSyDelay implementation
configs/                  Default settings
data/                     Modeling data and SUMO scenario assets
docs/                     Setup and evaluation details
```

## Quick start

Python 3.11 is recommended. From PowerShell:

```powershell
git clone https://github.com/ForrestXie9/CoSyDelay.git
cd CoSyDelay

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt

$env:PYTHONPATH = (Resolve-Path .\code).Path
$env:LLM_API_KEY = "your-api-key"

python -m methods.cosydelay.run --intersection 1 --output runs/example_i1
```

Use a key for the configured LLM provider; the defaults are `llm-api.net` and `gpt-4.1-mini`. See the [setup guide](docs/SETUP.md) for other providers and macOS/Linux commands.

Choose an intersection from `1` to `6` and a new output directory for each run. The learned model is saved to `model.json`; test metrics are saved to `result.json`.

Optional `--population` and `--generations` arguments override the [default configuration](configs/cosydelay.json). Run `python -m methods.cosydelay.run --help` for all options.

## Data

- [I1–I6 splits](data/locked_splits/): fixed Training, Validation, and Test datasets.
- [AASUMO extension](data/augmented_aasumo/): same-layout data for high-demand and oversaturation evaluation.
- [SingleTSCBaselines scenarios](data/signal_optimization/SingleTSCBaselines/): three attributed SUMO junction scenarios, provided as data only.

## Further reading

[Evaluation protocol](docs/PROTOCOL.md) · [Source modules](code/methods/cosydelay/README.md) · [Citations and data attribution](CITATIONS.md)

Source code is released under the [MIT License](LICENSE). Third-party data and scenario assets retain their own terms; see their notices before redistribution.
