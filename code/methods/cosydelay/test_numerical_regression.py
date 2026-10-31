"""Real-topology regression checks, independent of LLM sampling or fitting."""
from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from constants import INTERSECTION_CONFIGS
from . import pipeline as p


EXPRESSION = (
    "Cycle_Time * (a1 / GR_phase + a2 * flow_lane / (GR_phase + flow_lane) "
    "+ a3 * log(1 + flow_lane) / GR_phase)"
)


def numpy_predictions(frame, config, parameters):
    """Compute the chosen equation and movement weights without predictor helpers."""
    predictions = {}
    cycle = frame["Cycle_Time"].to_numpy(dtype=float)
    for approach in config["approaches"]:
        total = np.zeros(len(frame))
        weighted = np.zeros(len(frame))
        for lane, coefficient in parameters.items():
            direction, movement = lane.split("_")
            if direction != approach:
                continue
            flow = frame[f"flow_{lane}"].to_numpy(dtype=float)
            phases = [phase for phase, mapping in config["phase_movements"].items()
                      if movement in mapping.get(direction, ())]
            green = sum(frame[f"GR_{phase}"].to_numpy(dtype=float) for phase in phases)
            if not phases:
                green = np.full(len(frame), .5)
            delay = cycle * (
                coefficient["a1"] / green
                + coefficient["a2"] * flow / (green + flow)
                + coefficient["a3"] * np.log1p(flow) / green
            )
            total += flow
            weighted += flow * np.maximum(delay, 0.)
        predictions[approach] = np.divide(
            weighted, total, out=np.full(len(frame), np.nan), where=total > 1e-6
        )
    return predictions


@pytest.mark.parametrize("intersection", [1, 3, 4, 6])
def test_real_topologies_match_independent_numpy_predictions_and_metrics(tmp_path, intersection):
    manifest = json.loads((p.DEFAULT_DATA / "FROZEN_SPLIT_MANIFEST.json").read_text(encoding="utf-8"))
    frame, _ = p.read_split(p.DEFAULT_DATA, intersection, "Train", manifest)
    frame = frame.iloc[:64].copy()
    runner = p.NumericalRunner(frame, intersection, tmp_path, p.load_config())
    parameters = {lane: {"a1": .02 + .002 * index, "a2": .25 + .01 * index, "a3": .1}
                  for index, lane in enumerate(runner.lanes)}
    try:
        actual = runner.predict(frame, EXPRESSION, parameters)
        expected = numpy_predictions(frame, INTERSECTION_CONFIGS[intersection], parameters)
        for approach in runner.all_approaches:
            np.testing.assert_allclose(actual[approach], expected[approach], rtol=1e-13, atol=1e-12,
                                       equal_nan=True)
        assert runner.approaches == tuple(
            a for a in runner.all_approaches if (intersection, a) not in {(4, "E"), (6, "W")}
        )
        metrics = p.prediction_metrics(frame, actual, runner.approaches)
        reference = {
            "r2": [r2_score(frame[f"Delay_{a}"], expected[a]) for a in runner.approaches],
            "rmse": [np.sqrt(mean_squared_error(frame[f"Delay_{a}"], expected[a]))
                     for a in runner.approaches],
            "mae": [mean_absolute_error(frame[f"Delay_{a}"], expected[a]) for a in runner.approaches],
        }
        for key, values in reference.items():
            assert metrics["diravg"][key] == pytest.approx(np.mean(values), rel=1e-13, abs=1e-12)
        assert metrics["fitness_accuracy"] == pytest.approx(np.maximum(reference["r2"], 0).mean())
    finally:
        runner.close()
