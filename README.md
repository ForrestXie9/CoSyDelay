# CoSyDelay: Cognitive Symbolic Regression for Traffic Delay Estimation

CoSyDelay is a symbolic-learning method for estimating traffic delay at signalized intersections. Its goal is to learn a compact, readable delay equation from movement-level traffic data, while keeping the equation consistent with basic traffic principles.

The model uses three directly meaningful inputs for each movement:

- `flow_lane`: movement flow ratio;
- `GR_phase`: effective green ratio;
- `Cycle_Time`: signal cycle time in seconds.

The released implementation is intended for movement-level delay modeling at the six intersections (`I1`-`I6`) supplied in the locked data split. It includes the original split, the supplied oversaturation extension, and three attributed signal-optimization scenarios.

## How CoSyDelay works

CoSyDelay combines language-model-assisted symbolic search with numerical fitting and explicit traffic-physics checks:

1. The LLM proposes a symbolic structure, without proposing fitted coefficient values.
2. The implementation parses and normalizes the expression, rejects unsupported or unsafe structures, and removes canonical duplicates.
3. Positive movement-specific coefficients are fitted using the Training split only.
4. Each fitted candidate is checked against the declared traffic principles, including required variables, flow and green-ratio monotonicity, time-unit consistency, non-negativity, low-demand behavior, and the zero-green limit.
5. Candidates evolve through initialization and regeneration. Survivor selection uses the declared Training fitness, combining Training R-squared with the binary physical-compliance term.
6. The best expression is retained as a compact, interpretable delay model. Validation and Test data are kept separate from the search and are not used to select candidates.

The default search uses a population of 10 expressions and 10 total generation batches: 10 initial candidates plus 9 regeneration rounds, or 100 evaluated candidates in total. The population and generation counts are configurable at runtime; generated runs, audit logs, and result files stay local and are ignored by Git.

## Repository structure

```text
code/
  methods/cosydelay/       public CoSyDelay entry point and command-line runner
    engine/                 search, fitting, physics, and candidate selection
    support/                data, expression, numerical, and LLM support
data/
  locked_splits/           I1-I6 Training/Validation/Test JSONL files
  augmented_aasumo/        supplied same-layout oversaturation input and metadata
  signal_optimization/
    SingleTSCBaselines/    three attributed junction scenario packages
CITATIONS.md               method citation and third-party data notices
requirements.txt           runtime dependency specification
requirements-lock.txt     pinned environment used for the release
LICENSE                    source-license terms
```

`methods/cosydelay/` is the only released method. Its `engine/` and `support/`
subdirectories are implementation details, not historical methods or
alternative variants. The public Python interface is `methods.cosydelay`.

The repository intentionally does not contain API keys, historical variant folders, search logs, generated predictions, result tables, or bootstrap outputs.

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
The candidate budget is their product, so the default evaluates 100 candidate
structures. The runner reads the corresponding Training file from
`data/locked_splits`. It does not use Validation or Test data to choose
expressions. Keep the resulting `runs/` directory local if you want to inspect
the search history; it is excluded from the release.

Before making an API call, you can verify the public import path without a key:

```powershell
python -c "import methods.cosydelay.fitter; import expression_rules; print('CoSyDelay import OK')"
```

## Included data

`data/locked_splits/` contains the six intersection-specific Training, Validation, and Test JSONL files plus the frozen split metadata. Validation is separate from Training, and Test is not used during candidate selection.

`data/augmented_aasumo/AASumo-I1to6_AllSites-14220.csv` is the supplied same-layout extension for high-demand and oversaturation stress tests. Its `manifest.json` and `protocol.json` describe the data and its intended boundary. Only input data and metadata are distributed; no predictions or evaluation outputs are included.

`data/signal_optimization/SingleTSCBaselines/` contains data-only scenarios for `Beijing_Gaojiaoyuan`, `Chengdu_Guanghua`, and `Tianjin_zhijingdao`. These are third-party network and route assets, not CoSyDelay data. See `CITATIONS.md` and the local `NOTICE.md` before redistributing or replaying them.

## Citation and license

Please cite the accompanying CoSyDelay paper and identify the released source revision used in an experiment. See `CITATIONS.md` for the method citation, the third-party scenario attribution, and the SUMO reference for any external replay. The included datasets and scenario files may have terms separate from the CoSyDelay source license; see `LICENSE` and the notices in the data directories.
