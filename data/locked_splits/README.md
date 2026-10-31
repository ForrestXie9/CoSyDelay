# Locked I1-I6 splits

This directory contains the six intersection-specific data partitions used by
the released method. Each intersection has separate `Train`, `Validation`,
and `Test` JSONL files. `FROZEN_SPLIT_MANIFEST.json` records row counts,
hashes, and the split protocol; `split_assignments.csv` records the frozen
assignment table.

The JSONL samples and their frozen assignments are unchanged. I1-I6 contain
5,909 Training, 1,512 Validation and 2,853 Test records in total.

Use only Training for candidate coefficient fitting and evolutionary feedback.
Select the final expression using Validation Fitness with those Training-fitted
coefficients. Only after structure selection, combine Training and Validation
for a coefficient-only refit. Recheck all seven physical principles using the
new coefficients and freeze the final model before opening Test. Test labels
never select a structure, a coefficient tuple, or a physical-check fallback.

`FROZEN_SPLIT_MANIFEST.json` schema 2 clarifies this permitted final refit;
the individual JSONL file hashes and split seed `20260712` remain unchanged.
The Git attributes disable line-ending conversion for these frozen JSONL
files, so the committed and downloaded bytes match the audited hashes too.
