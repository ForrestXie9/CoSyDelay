# Locked I1-I6 splits

This directory contains the six intersection-specific data partitions used by
the released method. Each intersection has separate `Train`, `Validation`,
and `Test` JSONL files. `FROZEN_SPLIT_MANIFEST.json` records row counts,
hashes, and the split protocol; `split_assignments.csv` records the frozen
assignment table.

Use Training for parameter fitting, Validation for structure/restart
selection, and Test only for the final frozen evaluation. Do not merge these
partitions during model selection.
