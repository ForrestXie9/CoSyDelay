# Citations and third-party data

## CoSyDelay

This repository contains the CoSyDelay symbolic traffic-delay implementation
and release-specific evaluation scripts. Please cite the accompanying paper
when using the method.

## SingleTSCBaselines scenarios

The three signal-optimization layouts bundled under
`data/signal_optimization/SingleTSCBaselines/junction_scenarios/` were taken
from the public [Traffic-Alpha/SingleTSCBaselines repository](https://github.com/Traffic-Alpha/SingleTSCBaselines),
revision `3147aa9aef16e5c78f83a557d2935a41c1fde079`. They are third-party
network/scenario files; retain the upstream copyright and license notices and
check the upstream repository for any updated terms before redistribution.

Only `Beijing_Gaojiaoyuan`, `Chengdu_Guanghua`, and `Tianjin_zhijingdao` are
included in this bundle. The other upstream scenarios and generated result
tables are intentionally omitted.

The benchmark environment is built on TransSimHub. For the associated
platform paper, cite M. Wang et al., “TranSimHub: A Unified Air-Ground
Simulation Platform for Multi-Modal Perception and Decision-Making,” arXiv
preprint arXiv:2510.15365 (2025),
[doi:10.48550/arXiv.2510.15365](https://doi.org/10.48550/arXiv.2510.15365).

## SUMO

The signal replay uses Eclipse SUMO and TraCI. Cite:

> Pablo A. Lopez, Michael Behrisch, Laura Bieker-Walz, Jakob Erdmann,
> Yun-Pang Flötteröd, Robert Hilbrich, Leonhard Lücken, Johannes Rummel,
> Peter Wagner, and Evamarie Wießner, “Microscopic Traffic Simulation using
> SUMO,” 2018 IEEE Intelligent Transportation Systems Conference (ITSC),
> 2018, doi: [10.1109/ITSC.2018.8569938](https://doi.org/10.1109/ITSC.2018.8569938).

The trajectories and delays produced by the TSO scripts are SUMO simulation
outputs, not field observations.
