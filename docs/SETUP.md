# Setup and configuration

The [project README](../README.md) provides the PowerShell quick start. Python 3.11 is recommended; `requirements-lock.txt` pins the release dependencies.

## macOS and Linux

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt

export PYTHONPATH="$PWD/code"
export LLM_API_KEY="your-api-key"

python -m methods.cosydelay.run --intersection 1 --output runs/example_i1
```

Choose a new output directory for each run. Existing run directories are not overwritten.

## LLM provider

Use a key issued by the provider configured below. The defaults refer to the gateway already configured in the implementation, not to every OpenAI-compatible service.

| Environment variable | Default |
|---|---|
| `LLM_API_KEY` | Required; no source-code default |
| `LLM_API_ENDPOINT` | `llm-api.net` |
| `LLM_MODEL` | `gpt-4.1-mini` |
| `LLM_REQUEST_PATH` | `/v1/chat/completions` |

To change providers, set the host, model, and key together. For example, in PowerShell:

```powershell
$env:LLM_API_ENDPOINT = "your-api-host"
$env:LLM_MODEL = "your-model-name"
$env:LLM_API_KEY = "your-provider-key"
```

Replace the placeholders with the provider's settings. Keep credentials in the local environment; do not commit them.

## Search options

`--population` and `--generations` override the defaults in [configs/cosydelay.json](../configs/cosydelay.json). `--intersection` accepts IDs 1–6, `--data-dir` points to the locked split files, and `--seed-base` overrides the configured seed.

Inspect the available options without making an API call:

```powershell
python -m methods.cosydelay.run --help
```

For complete split usage, selection criteria, refitting, and physical-check policies, see the [evaluation protocol](PROTOCOL.md).

## Optional Windows/Anaconda workaround

An existing installation with an incompatible optional `gmpy2` DLL may need pure-Python arithmetic. Set these variables before importing SymPy:

```powershell
$env:SYMPY_GROUND_TYPES = "python"
$env:MPMATH_NOGMPY = "1"
```

These settings are normally unnecessary in a clean environment installed from the pinned requirements.
