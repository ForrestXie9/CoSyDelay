# CoSyDelay: Cognitive Symbolic Regression for Traffic Delay Estimation

CoSyDelay is a symbolic-learning method for estimating traffic delay at signalized intersections. Its goal is to learn a compact, readable delay equation from movement-level traffic data, while keeping the equation consistent with basic traffic principles.

## Repository structure

```text
code/
  methods/cosydelay/       public CoSyDelay entry point and command-line runner
    pipeline.py             validation selection, final refit, and frozen testing
    physics.py              bounded execution of the seven physical checks
    engine/                 search, fitting, physics, and candidate selection
    support/                data, expression, numerical, and LLM support
data/
  locked_splits/           I1-I6 Training/Validation/Test JSONL files
  augmented_aasumo/        supplied same-layout oversaturation input and metadata
  signal_optimization/
    SingleTSCBaselines/    three attributed junction scenario packages
CITATIONS.md               method citation and third-party data notices
configs/cosydelay.json      default search and data-use configuration
requirements.txt           runtime dependency specification
requirements-lock.txt     pinned environment used for the release
LICENSE                    source-license terms
```

## Environment setup

Python 3.11 or newer is recommended. From the repository root, create an isolated environment and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
$env:PYTHONPATH = (Resolve-Path .\code).Path
```

The LLM search requires a provider key in the process environment. The key is never stored in this repository:

```powershell
$env:LLM_API_KEY = "<your-provider-key>"
```

The default gateway settings are `llm-api.net` and `gpt-4.1-mini`. They can be overridden without editing source files:

```powershell
$env:LLM_API_ENDPOINT = "llm-api.net"
$env:LLM_MODEL = "gpt-4.1-mini"
```

If an existing Windows/Anaconda installation has an incompatible optional `gmpy2` DLL, force pure-Python arithmetic before importing SymPy:

```powershell
$env:SYMPY_GROUND_TYPES = "python"
$env:MPMATH_NOGMPY = "1"
```

These two variables are normally unnecessary in a clean virtual environment created from the pinned requirements.

## Running CoSyDelay

Run one intersection from the repository root. Use a new output directory for every run:

```powershell
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"

python -u -m methods.cosydelay.run `
  --intersection 1 `
  --data-dir .\data\locked_splits `
  --output ".\runs\cosydelay_i1_$stamp" `
  --population 10 `
  --generations 10
```

Replace `1` with an intersection ID from `1` to `6`. `--population` and
`--generations` are optional; their defaults are `10` and `10`, respectively.
The default configuration is [`configs/cosydelay.json`](configs/cosydelay.json).
The runner fits candidate coefficients and drives evolution using Training
only, then selects from all completed candidates using Validation Fitness:

`F_val = mean_a(max(0, R2_val,a)) + P_train`.

Here `a` denotes a valid approach; `P_train` is 1 only when all seven physical
checks pass for the Training-fitted coefficients, otherwise 0. Ties use lower
direction-average Validation RMSE, then lower candidate ID. Validation does
not refit the candidates, and pooled R2 is not used for selection.

After selection, the structure is fixed and only its coefficients are refitted
on Training+Validation. All seven physical checks are rerun with those new
coefficients. A failed or resource-limited check gives a final physical
component of 0; it never inherits the earlier pass score or triggers Test-based
reselection. The persisted final model is frozen before Test is opened. Test
R2, RMSE and MAE are raw per-approach metrics averaged equally across valid
approaches; Test Fitness uses clipped per-approach R2 plus the **final** binary
physical component. Invalid predictions fail the run rather than dropping rows
or reverting to a previous model.

Local outputs include `configuration.json`, `validation_candidates.json`,
`selection_frozen.json`, `final_physics.json`, `model.json`, `freeze.json`,
`split_access_audit.json` and `result.json`. They record the effective settings,
source commit/source hashes, data hashes, final coefficients and stage timings.
Search time and selection/refit/check/testing time are reported separately.
Generated runs remain local and are ignored by Git.

Before making an API call, you can verify the public import path without a key:

```powershell
python -c "import methods.cosydelay.fitter; import expression_rules; print('CoSyDelay import OK')"
```

## Included data

`data/locked_splits/` contains the six intersection-specific Training (5,909),
Validation (1,512), and Test (2,853) JSONL records plus frozen split metadata
(seed `20260712`). The sample assignments are unchanged. Training and
Validation remain separate during search and selection; they are combined only
for the final coefficient refit. Test never participates in fitting or selection.

`data/augmented_aasumo/AASumo-I1to6_AllSites-14220.csv` is the supplied same-layout extension for high-demand and oversaturation stress tests. Its `manifest.json` and `protocol.json` describe the data and its intended boundary. Only input data and metadata are distributed; no predictions or evaluation outputs are included.

`data/signal_optimization/SingleTSCBaselines/` contains data-only scenarios for `Beijing_Gaojiaoyuan`, `Chengdu_Guanghua`, and `Tianjin_zhijingdao`. These are third-party network and route assets, not CoSyDelay data. See `CITATIONS.md` and the local `NOTICE.md` before redistributing or replaying them.

## Citation and license

Please cite the accompanying CoSyDelay paper and identify the released source revision used in an experiment. See `CITATIONS.md` for the method citation, the third-party scenario attribution, and the SUMO reference for any external replay. The included datasets and scenario files may have terms separate from the CoSyDelay source license; see `LICENSE` and the notices in the data directories.
