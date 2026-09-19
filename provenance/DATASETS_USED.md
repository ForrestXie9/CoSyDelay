# Data used by the release

This document describes the data that are actually redistributed in this
bundle. The original research workspace is not required to read the locked
split files, and no API key or original all-record JSONL file is included.

## 1. Locked I1–I6 delay split

`data/locked_splits/` contains the exact leakage-free split used by the main
comparison. Each intersection has separate JSONL files for Training,
Validation, and Test. The split seed is `20260712`; the Test partition was
never used for structure selection or coefficient fitting.

| Intersection | Training | Validation | Test |
|---:|---:|---:|---:|
| I1 | 1,009 | 242 | 503 |
| I2 | 1,027 | 223 | 503 |
| I3 | 934 | 284 | 438 |
| I4 | 987 | 264 | 501 |
| I5 | 966 | 240 | 412 |
| I6 | 986 | 259 | 496 |
| **Total** | **5,909** | **1,512** | **2,853** |

`FROZEN_SPLIT_MANIFEST.json` records SHA-256 hashes, row counts, and pairwise
overlap checks. The original all-record JSONL source is intentionally omitted;
the included split is the reproducibility artifact.

## 2. Same-layout oversaturation extension

`data/augmented_aasumo/AASumo-I1to6_AllSites-14220.csv` is an augmented
AASUMO table for the same six layouts. Its SHA-256 is
`b5f4d44188c3f0f397b497ad7545e1a701b7ace02779f31f3780edbf14d9f234`.
The extension contains 14,219 source rows. After removing exact overlaps with
the locked union and five near-duplicates, 3,940 strict additional rows
remain; 3,644 of them satisfy the oversaturation definition
`max movement degree of saturation x >= 1.0`.

CoSyDelay is frozen before this evaluation. No expression selection,
coefficient refitting, or threshold tuning uses the extension. The audit,
coverage, predictions, and frozen metrics are stored alongside the CSV.
This is a same-layout stress test, not unseen-intersection or field
validation. See `data/augmented_aasumo/protocol.json`.

## 3. Signal-optimization scenarios

`data/signal_optimization/SingleTSCBaselines/junction_scenarios/` contains
only the three layouts used by the offline timing replay:

* `Beijing_Gaojiaoyuan`
* `Chengdu_Guanghua`
* `Tianjin_zhijingdao`

These network, route, and SUMO configuration files come from the upstream
`Traffic-Alpha/SingleTSCBaselines` repository at revision
`3147aa9aef16e5c78f83a557d2935a41c1fde079`; they are third-party data, not
CoSyDelay data. Attribution and license guidance are in the scenario
`NOTICE.md` and repository-level `CITATIONS.md`.

This release contains scenario data only; it does not include the TSO scripts
or a SUMO runner. If replayed with external tooling, fixed CoSyDelay, Webster,
HCM, Akcelik, and optimized timing plans produce SUMO simulation outputs, not
field-observation labels. Cite Lopez et al. (2018) for SUMO; see `CITATIONS.md`
for the complete reference.

## 4. Data boundaries

The package does not redistribute the original all-record JSONL, NGSIM/DLR
external-transfer files, generated search histories, or TSO result tables.
Those omissions prevent accidental mixing of exploratory artifacts with the
locked evaluation protocol and avoid making unsupported field-data claims.
