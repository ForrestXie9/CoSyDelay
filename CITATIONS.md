# Citations and data notices

## CoSyDelay

When using the method, cite the accompanying CoSyDelay paper and identify the
released source revision used in the experiment.

## SingleTSCBaselines scenarios

The three junction scenarios under
`data/signal_optimization/SingleTSCBaselines/junction_scenarios/` are selected
from the public
[Traffic-Alpha/SingleTSCBaselines repository](https://github.com/Traffic-Alpha/SingleTSCBaselines),
revision `3147aa9aef16e5c78f83a557d2935a41c1fde079`. They are third-party
network and route files, not CoSyDelay data. Preserve the upstream notice and
license terms and check the upstream repository before redistribution.

Included scenarios:

- `Beijing_Gaojiaoyuan`
- `Chengdu_Guanghua`
- `Tianjin_zhijingdao`

No generated result tables or optimization outputs are included here.

## SUMO (only for external replay)

This repository contains scenario data only; it does not include a SUMO,
TraCI, or signal-control runner. If the scenarios are replayed externally,
cite:

> Pablo A. Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann,
> Yun-Pang Flotterod, Robert Hilbrich, Leonhard Lucken, Johannes Rummel,
> Peter Wagner, and Evamarie WieBner, "Microscopic Traffic Simulation using
> SUMO," 2018 IEEE Intelligent Transportation Systems Conference (ITSC),
> 2018, doi: [10.1109/ITSC.2018.8569938](https://doi.org/10.1109/ITSC.2018.8569938).

Any trajectories, delays, or optimization outcomes produced by an external
SUMO run are simulation outputs, not field measurements.
