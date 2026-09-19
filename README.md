# CoSyDelay

CoSyDelay is a symbolic traffic-delay model. This repository is the clean
distribution of the method: source modules, the locked data used by the
method, the oversaturation extension, and three clearly attributed junction
scenario packages.

No search logs, experiment outputs, result tables, API keys, or historical
variant folders are part of the release.

## Repository layout

```text
code/
  methods/cosydelay/       public method entry point and command line runner
  methods/_runtime/        data, physics, optimization, and LLM support modules
  methods/_compat/         internal compatibility modules required at runtime
data/
  locked_splits/           I1-I6 Training/Validation/Test JSONL files
  augmented_aasumo/        supplied same-layout oversaturation input data
  signal_optimization/     three attributed SingleTSCBaselines scenarios
```

`methods/_runtime/` and `methods/_compat/` are implementation dependencies;
they are not separate methods or experimental variants. The public interface
is only `methods.cosydelay`.

## Installation

Python 3.11 or newer is recommended. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
$env:PYTHONPATH = (Resolve-Path .\code).Path
```

The LLM search needs a locally configured `LLM_API_KEY`. Credentials are not
stored in this repository. On Windows/Anaconda installations with an
incompatible optional `gmpy2` DLL, force both SymPy and mpmath to use
pure-Python arithmetic before running the package:

```powershell
$env:SYMPY_GROUND_TYPES = "python"
$env:MPMATH_NOGMPY = "1"
```

This setting is not needed in a clean virtual environment created from the
provided requirements.

## Run the method

The runner uses the locked Training/Validation/Test split. Candidate
structures and parameters are selected without reading Test; Test is evaluated
only after the model is frozen.

```powershell
python -u -m methods.cosydelay.run_p10g10_100 `
  --intersection 1 `
  --data-dir .\data\locked_splits `
  --output .\runs\cosydelay_i1
```

Use an intersection ID from `1` to `6` and a separate output directory for
each run. The default search budget is P10/G10 (100 evaluated candidates).
Generated runs remain local and are ignored by Git.

For an offline import check (no API key and no search):

```powershell
python -c "import methods.cosydelay.fitter; import expression_rules; print('CoSyDelay import OK')"
```

## Included data

### Locked splits

`data/locked_splits/` contains the six intersection-specific Training,
Validation, and Test JSONL files, together with the frozen split manifest.
Validation is separate from Training and Test is not used for selection.

### Oversaturation input

`data/augmented_aasumo/AASumo-I1to6_AllSites-14220.csv` is the supplied
same-layout extension used to stress-test the frozen method at high demand.
`manifest.json` and `protocol.json` describe its construction and claim
boundary. The release includes the input and metadata only, not predictions,
metrics, audits, or bootstrap output.

### Signal scenarios

`data/signal_optimization/SingleTSCBaselines/` contains data-only scenarios
for Beijing_Gaojiaoyuan, Chengdu_Guanghua, and Tianjin_zhijingdao. These are
third-party network and route assets; this repository does not include a SUMO,
TraCI, or signal-optimization runner. See `CITATIONS.md` and the local
`NOTICE.md` before redistributing them.

## License and citation

See `LICENSE` for the CoSyDelay source license and `CITATIONS.md` for the
method citation and third-party scenario attribution. The included datasets
and third-party scenario files may have separate terms.
