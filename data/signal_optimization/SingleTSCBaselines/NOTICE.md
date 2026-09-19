# Third-party scenario notice

The files below this directory are selected network and route scenarios from
the upstream [Traffic-Alpha/SingleTSCBaselines](https://github.com/Traffic-Alpha/SingleTSCBaselines)
repository, fixed at revision `3147aa9aef16e5c78f83a557d2935a41c1fde079`.
They are not CoSyDelay code and are not claimed as original data. Please
preserve the upstream license/copyright terms and consult the upstream
repository before redistributing modified copies.

This release includes only three scenarios:

- `Beijing_Gaojiaoyuan`
- `Chengdu_Guanghua`
- `Tianjin_zhijingdao`

This release contains scenario data only; no TSO controller or SUMO runner is
included. If the scenarios are replayed with external tooling, any generated
vehicle trajectories, delays, or optimization outcomes are simulation outputs
rather than field measurements. See the repository-level `CITATIONS.md` for
the SUMO reference.
