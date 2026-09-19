# Oversaturation input data

This directory contains the supplied AASUMO input extension for the six
locked intersection layouts (I1-I6). It is intended for a frozen-model
stress test under different traffic regimes, including oversaturated demand.

The CSV is input data; it is not a model result. `manifest.json` records the
source and curation metadata, and `protocol.json` records the intended
evaluation boundary. No fitting, selection, or tuning should use these rows
before the model is frozen.

Predictions, metrics, audit tables, bootstrap intervals, and search logs are
deliberately not distributed in this clean release.
