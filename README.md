# CoSyDelay reproducible release

This repository is a copy-only release of the final CoSyDelay symbolic
traffic-delay method. It is deliberately separate from the research workspace
used to run experiments; preparing or using this release does not modify that
workspace.

## Release contents

```text
code/
  methods/cosydelay/       public CoSyDelay search, fitter, prompts, and tests
  methods/_runtime/        packaged preprocessing, physics, optimization, and LLM runtime
  methods/_compat/         private compatibility modules required by the final method
data/
  locked_splits/           I1-I6 Train/Validation/Test JSONL split
  augmented_aasumo/        input-deduplicated oversaturation extension
  signal_optimization/     three third-party SingleTSCBaselines scenarios (data only)
```

The release contains only the CoSyDelay method implementation. Reviewer-only
experiment runners, historical standalone method folders, and the SUMO/TSO
execution code are not included. The signal-optimization directory contains
scenario data for reproducibility; it does not contain a signal controller or
simulation runner. See `CITATIONS.md` and the included upstream `NOTICE.md`
before redistributing those third-party files.

The original `data/original_jsonl/` records, API keys, generated search
histories, caches, and large result tables are not redistributed. The locked
I1-I6 split is included so the reported protocol can be reproduced directly.

## Environment

Use Python 3.11 or newer (pinned versions are listed in
`requirements-lock.txt`):

```powershell
Set-Location "<path>\CoSyDelay_OpenSource"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
$env:PYTHONPATH = (Resolve-Path .\code).Path
```

LLM searches require a locally supplied `LLM_API_KEY`. The packaged example
file is a template only; never commit credentials.

## Run CoSyDelay

The formal protocol uses the fixed Training/Validation/Test split. Structures
and coefficients are selected using Training and Validation only; Test is read
after the model is frozen.

```powershell
python -u -m methods.cosydelay.run_p10g10_100 `
  --intersection 1 `
  --data-dir .\data\locked_splits `
  --output .\runs\cosydelay_i1
```

Change `--intersection` to `1` through `6` and use a new output directory for
each run. The default budget is P10/G10 (100 evaluated candidates). Run each
intersection separately when reproducing the full I1-I6 protocol.

For an offline import check (no API key required):

```powershell
python -c "import methods.cosydelay.fitter; import expression_rules; print('CoSyDelay import OK')"
```

## Oversaturation extension

`data/augmented_aasumo/` contains the input-deduplicated AASUMO extension for
I1-I6. Rows with maximum degree of saturation `x >= 1.0` are evaluated after
CoSyDelay is frozen. No refitting, selection, or tuning is performed on these
rows. See its `README.md` and `protocol.json` for hashes and the audit protocol.

## Signal-optimization scenarios

Only three data-only junction scenarios from
[Traffic-Alpha/SingleTSCBaselines](https://github.com/Traffic-Alpha/SingleTSCBaselines)
are included:

- `Beijing_Gaojiaoyuan`
- `Chengdu_Guanghua`
- `Tianjin_zhijingdao`

No SUMO/TraCI or TSO runner is part of this release. To replay these scenarios,
install and use the upstream tooling separately, and cite the sources listed
in `CITATIONS.md`.

## Data provenance and license

The split and oversaturation directories contain their own manifests and
protocol files. The original JSONL records are simulation-compatible records
whose complete upstream simulator/field provenance was not documented in the
source workspace; they must not be described as field observations without
further evidence. See `CITATIONS.md` for third-party scenario attribution.

The CoSyDelay source follows the included `LICENSE`. This does not
automatically relicense the locked/augmented datasets, SUMO, or the upstream
SingleTSCBaselines scenarios; verify their terms before redistribution.
