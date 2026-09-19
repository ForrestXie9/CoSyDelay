# main_lane_level.py
import os
import json
import sys
import numpy as np
from typing import List

from constants import INTERSECTION_CONFIGS
from data_processing import load_dataset_flexible, preprocess_data_flexible
from population_evolution_lane import evolve_universal_lane_expression
from optimization_lane import calculate_approach_delays_from_universal
from methods.cosydelay_lbfgsb_r10_parallel4 import (
    OFFICIAL_RESTARTS,
    P4G2_ADAPTIVE_SEARCH,
    install_retained_evolution,
)
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_lanes_and_mapping(intersection_id: int):
    """Get all lanes and lane-to-approach mapping"""
    config = INTERSECTION_CONFIGS[intersection_id]
    lanes = []
    lane_to_approach = {}

    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            lane_to_approach[lane] = approach

    return lanes, lane_to_approach


def main(generations_list: List[int] = [2], pop_size_list: List[int] = [4]):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    n_runs = int(os.environ.get("COSYDELAY_RUNS", "3"))

    if "COSYDELAY_GENERATIONS" in os.environ:
        generations_list = [
            int(value.strip())
            for value in os.environ["COSYDELAY_GENERATIONS"].split(",")
            if value.strip()
        ]
    if "COSYDELAY_POPULATIONS" in os.environ:
        pop_size_list = [
            int(value.strip())
            for value in os.environ["COSYDELAY_POPULATIONS"].split(",")
            if value.strip()
        ]
    if not generations_list or not pop_size_list:
        raise ValueError("Generation/population overrides cannot be empty")
    search_policy = os.environ.get(
        "COSYDELAY_SEARCH_POLICY", "p4g2_feasible_adaptive"
    ).strip().lower()
    if search_policy not in {"p4g2_feasible_adaptive", "standard"}:
        raise ValueError(
            "COSYDELAY_SEARCH_POLICY must be p4g2_feasible_adaptive or standard"
        )

    base_path = os.environ.get("COSYDELAY_DATA_DIR") or os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "Final_cosy_delay",
            "jsonl_files",
        )
    )
    intersection_text = os.environ.get(
        "COSYDELAY_INTERSECTIONS", "1,2,3,4,5,6,7,8,9"
    )
    intersection_ids = [
        int(value.strip())
        for value in intersection_text.split(",")
        if value.strip()
    ]
    if not intersection_ids:
        raise ValueError("COSYDELAY_INTERSECTIONS must contain at least one ID")
    output_base_path = os.environ.get(
        "COSYDELAY_OUTPUT_DIR", os.path.join(os.getcwd(), "traffic_models_lane_level")
    )

    os.makedirs(output_base_path, exist_ok=True)

    for intersection_id in intersection_ids:
        print(f"\n{'=' * 80}")
        print(f"{'LANE-LEVEL MODELING - INTERSECTION ' + str(intersection_id):^80}")
        print(f"{'=' * 80}")

        config = INTERSECTION_CONFIGS[intersection_id]

        # File paths
        train_path = os.path.join(base_path, f"Intersection_{intersection_id}_Train.jsonl")
        test_path = os.path.join(base_path, f"Intersection_{intersection_id}_Test.jsonl")
        output_path = os.path.join(output_base_path, f"Intersection_{intersection_id}")
        os.makedirs(output_path, exist_ok=True)

        # Load data
        df_train = preprocess_data_flexible(load_dataset_flexible(train_path, intersection_id), intersection_id)
        df_test = preprocess_data_flexible(load_dataset_flexible(test_path, intersection_id), intersection_id)

        print(f"\nData loaded:")
        print(f"  Train: {df_train.shape}")
        print(f"  Test: {df_test.shape}")

        # Get lane configuration
        lanes, lane_to_approach = get_lanes_and_mapping(intersection_id)

        print(f"\nLane configuration:")
        print(f"  Total lanes: {len(lanes)}")
        for approach in config["approaches"]:
            approach_lanes = [l for l in lanes if lane_to_approach[l] == approach]
            print(f"  {approach} approach: {approach_lanes}")

        # Get approach targets
        approach_targets_train = {}
        approach_targets_test = {}
        for approach in config["approaches"]:
            target_col = f"Delay_{approach}"
            if target_col in df_train.columns:
                approach_targets_train[approach] = df_train[target_col]
            if target_col in df_test.columns:
                approach_targets_test[approach] = df_test[target_col]

        # Run for different parameter combinations
        for generations in generations_list:
            for pop_size in pop_size_list:
                if search_policy == "p4g2_feasible_adaptive" and (
                    generations != P4G2_ADAPTIVE_SEARCH.generations
                    or pop_size != P4G2_ADAPTIVE_SEARCH.population
                ):
                    raise ValueError(
                        "The retained adaptive policy is fixed at P4/G2. Set "
                        "COSYDELAY_SEARCH_POLICY=standard for other budgets."
                    )
                search_kwargs = (
                    P4G2_ADAPTIVE_SEARCH.evolution_kwargs()
                    if search_policy == "p4g2_feasible_adaptive"
                    else {}
                )
                print(f"\n{'=' * 80}")
                print(f"Configuration: Generations={generations}, Population={pop_size}")
                print(f"Search policy: {search_policy}")
                print(f"Running {n_runs} independent repetitions...")
                print(f"{'=' * 80}")

                # Store results from all runs
                all_runs_results = []

                for run_idx in range(n_runs):
                    print(f"\n{'#' * 80}")
                    print(f"{'RUN ' + str(run_idx + 1) + ' / ' + str(n_runs):^80}")
                    print(f"{'#' * 80}")

                    # EVOLVE: every generation uses training accuracy and
                    # post-fit physics only. Frozen evaluation data is absent.
                    with install_retained_evolution():
                        (
                            best_expr,
                            best_thought,
                            best_explanation,
                            best_lane_params,
                            history,
                        ) = evolve_universal_lane_expression(
                                df_train=df_train,
                                lanes=lanes,
                                lane_to_approach=lane_to_approach,
                                approach_targets=approach_targets_train,
                                universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                                generations=generations,
                                pop_size=pop_size,
                                intersection_id=intersection_id,
                                # Strict binary physical reward: every rule must
                                # pass to receive 1; otherwise the score is 0.
                                score_mode="binary",
                                physics_weight=float(os.environ.get("COSYDELAY_PHYSICS_WEIGHT", "1.0")),
                                prompt_knowledge=os.environ.get("COSYDELAY_PROMPT_KNOWLEDGE", "true").lower()
                                in {"1", "true", "yes", "on"},
                                seed=(
                                    int(os.environ.get("COSYDELAY_SEED", "20260712"))
                                    + intersection_id * 10000
                                    + generations * 100
                                    + pop_size * 10
                                    + run_idx
                                ),
                                optimizer_restarts=int(
                                    os.environ.get(
                                        "COSYDELAY_OPTIMIZER_RESTARTS",
                                        str(OFFICIAL_RESTARTS),
                                    )
                                ),
                                **search_kwargs,
                            )

                    # Evaluate on TRAIN set
                    train_predictions = calculate_approach_delays_from_universal(
                        df=df_train,
                        universal_expr=best_expr,
                        lane_parameters=best_lane_params,
                        lanes=lanes,
                        lane_to_approach=lane_to_approach,
                        intersection_id=intersection_id
                    )

                    train_metrics = {}
                    for approach in config["approaches"]:
                        if approach in approach_targets_train and approach in train_predictions:
                            y_true = approach_targets_train[approach].values
                            y_pred = train_predictions[approach]

                            r2 = r2_score(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae = mean_absolute_error(y_true, y_pred)
                            mape = calculate_mape(y_true, y_pred)

                            train_metrics[approach] = {
                                "r2": float(r2),
                                "rmse": float(rmse),
                                "mae": float(mae),
                                "mape": float(mape)
                            }

                    # Evaluate on TEST set
                    test_predictions = calculate_approach_delays_from_universal(
                        df=df_test,
                        universal_expr=best_expr,
                        lane_parameters=best_lane_params,
                        lanes=lanes,
                        lane_to_approach=lane_to_approach,
                        intersection_id=intersection_id
                    )

                    test_metrics = {}
                    for approach in config["approaches"]:
                        if approach in approach_targets_test and approach in test_predictions:
                            y_true = approach_targets_test[approach].values
                            y_pred = test_predictions[approach]

                            r2 = r2_score(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            mae = mean_absolute_error(y_true, y_pred)
                            mape = calculate_mape(y_true, y_pred)

                            test_metrics[approach] = {
                                "r2": float(r2),
                                "rmse": float(rmse),
                                "mae": float(mae),
                                "mape": float(mape)
                            }

                    # Calculate averages for this run
                    if train_metrics:
                        avg_train_r2 = np.mean([m["r2"] for m in train_metrics.values()])
                        avg_train_rmse = np.mean([m["rmse"] for m in train_metrics.values()])
                        avg_train_mae = np.mean([m["mae"] for m in train_metrics.values()])
                        avg_train_mape = np.mean([m["mape"] for m in train_metrics.values()])

                    if test_metrics:
                        avg_test_r2 = np.mean([m["r2"] for m in test_metrics.values()])
                        avg_test_rmse = np.mean([m["rmse"] for m in test_metrics.values()])
                        avg_test_mae = np.mean([m["mae"] for m in test_metrics.values()])
                        avg_test_mape = np.mean([m["mape"] for m in test_metrics.values()])

                    print(f"\nRun {run_idx + 1} Results:")
                    print(
                        f"  Train - R²: {avg_train_r2:.4f}, RMSE: {avg_train_rmse:.2f}, MAE: {avg_train_mae:.2f}, MAPE: {avg_train_mape:.2f}%")
                    print(
                        f"  Test  - R²: {avg_test_r2:.4f}, RMSE: {avg_test_rmse:.2f}, MAE: {avg_test_mae:.2f}, MAPE: {avg_test_mape:.2f}%")

                    # Store this run's results
                    run_result = {
                        "run_id": run_idx + 1,
                        "universal_expression": {
                            "template": best_expr,
                            "thought": best_thought,
                            "explanation": best_explanation
                        },
                        "lane_parameters": {
                            lane: {
                                "parameters": params,
                                "approach": lane_to_approach[lane]
                            }
                            for lane, params in best_lane_params.items()
                        },
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "train_average": {
                            "r2": float(avg_train_r2),
                            "rmse": float(avg_train_rmse),
                            "mae": float(avg_train_mae),
                            "mape": float(avg_train_mape)
                        },
                        "test_average": {
                            "r2": float(avg_test_r2),
                            "rmse": float(avg_test_rmse),
                            "mae": float(avg_test_mae),
                            "mape": float(avg_test_mape)
                        }
                    }
                    all_runs_results.append(run_result)

                    # Save individual run result
                    run_filename = f"lane_model_intersection_{intersection_id}_g{generations}_p{pop_size}_run{run_idx + 1}.json"
                    run_output_file = os.path.join(output_path, run_filename)

                    individual_run_data = {
                        "parameters": {
                            "generations": generations,
                            "pop_size": pop_size,
                            "intersection_id": intersection_id,
                            "modeling_approach": "lane_level_universal",
                            "run_id": run_idx + 1,
                            "score_mode": history[0].get("score_mode", "unknown"),
                            "prompt_knowledge": history[0].get("prompt_knowledge"),
                            "search_policy": search_policy,
                            "seed": history[0].get("seed"),
                        },
                        "universal_expression": {
                            "template": best_expr,
                            "thought": best_thought,
                            "explanation": best_explanation
                        },
                        "lane_parameters": {
                            lane: {
                                "parameters": params,
                                "approach": lane_to_approach[lane]
                            }
                            for lane, params in best_lane_params.items()
                        },
                        "approach_metrics": {
                            "train": train_metrics,
                            "test": test_metrics
                        },
                        "average_metrics": {
                            "train": {
                                "r2": float(avg_train_r2),
                                "rmse": float(avg_train_rmse),
                                "mae": float(avg_train_mae),
                                "mape": float(avg_train_mape)
                            },
                            "test": {
                                "r2": float(avg_test_r2),
                                "rmse": float(avg_test_rmse),
                                "mae": float(avg_test_mae),
                                "mape": float(avg_test_mape)
                            }
                        }
                    }

                    with open(run_output_file, 'w', encoding='utf-8') as f:
                        json.dump(individual_run_data, f, indent=4)
                    print(f"  ✓ Run {run_idx + 1} results saved to {run_output_file}")

                    # Save individual run history
                    run_history_file = os.path.join(output_path,
                                                    f"history_intersection_{intersection_id}_g{generations}_p{pop_size}_run{run_idx + 1}.json")
                    with open(run_history_file, 'w', encoding='utf-8') as f:
                        json.dump(history, f, indent=4)
                    print(f"  ✓ Run {run_idx + 1} history saved to {run_history_file}")

                # Calculate statistics across all runs
                print(f"\n{'=' * 80}")
                print(f"AGGREGATED RESULTS ACROSS {n_runs} RUNS")
                print(f"{'=' * 80}")

                # Collect metrics across runs
                train_r2_list = [run["train_average"]["r2"] for run in all_runs_results]
                train_rmse_list = [run["train_average"]["rmse"] for run in all_runs_results]
                train_mae_list = [run["train_average"]["mae"] for run in all_runs_results]
                train_mape_list = [run["train_average"]["mape"] for run in all_runs_results]

                test_r2_list = [run["test_average"]["r2"] for run in all_runs_results]
                test_rmse_list = [run["test_average"]["rmse"] for run in all_runs_results]
                test_mae_list = [run["test_average"]["mae"] for run in all_runs_results]
                test_mape_list = [run["test_average"]["mape"] for run in all_runs_results]

                # Calculate mean and std
                statistics = {
                    "train": {
                        "r2": {"mean": float(np.mean(train_r2_list)), "std": float(np.std(train_r2_list))},
                        "rmse": {"mean": float(np.mean(train_rmse_list)), "std": float(np.std(train_rmse_list))},
                        "mae": {"mean": float(np.mean(train_mae_list)), "std": float(np.std(train_mae_list))},
                        "mape": {"mean": float(np.mean(train_mape_list)), "std": float(np.std(train_mape_list))}
                    },
                    "test": {
                        "r2": {"mean": float(np.mean(test_r2_list)), "std": float(np.std(test_r2_list))},
                        "rmse": {"mean": float(np.mean(test_rmse_list)), "std": float(np.std(test_rmse_list))},
                        "mae": {"mean": float(np.mean(test_mae_list)), "std": float(np.std(test_mae_list))},
                        "mape": {"mean": float(np.mean(test_mape_list)), "std": float(np.std(test_mape_list))}
                    }
                }

                print("\nTrain Set Statistics:")
                print(f"  R²:   {statistics['train']['r2']['mean']:.4f} ± {statistics['train']['r2']['std']:.4f}")
                print(f"  RMSE: {statistics['train']['rmse']['mean']:.2f} ± {statistics['train']['rmse']['std']:.2f}")
                print(f"  MAE:  {statistics['train']['mae']['mean']:.2f} ± {statistics['train']['mae']['std']:.2f}")
                print(f"  MAPE: {statistics['train']['mape']['mean']:.2f}% ± {statistics['train']['mape']['std']:.2f}%")

                print("\nTest Set Statistics:")
                print(f"  R²:   {statistics['test']['r2']['mean']:.4f} ± {statistics['test']['r2']['std']:.4f}")
                print(f"  RMSE: {statistics['test']['rmse']['mean']:.2f} ± {statistics['test']['rmse']['std']:.2f}")
                print(f"  MAE:  {statistics['test']['mae']['mean']:.2f} ± {statistics['test']['mae']['std']:.2f}")
                print(f"  MAPE: {statistics['test']['mape']['mean']:.2f}% ± {statistics['test']['mape']['std']:.2f}%")

                # Calculate per-approach statistics
                approach_statistics = {}
                for approach in config["approaches"]:
                    approach_statistics[approach] = {
                        "train": {"r2": [], "rmse": [], "mae": [], "mape": []},
                        "test": {"r2": [], "rmse": [], "mae": [], "mape": []}
                    }

                # Collect per-approach metrics across runs
                for run in all_runs_results:
                    for approach in config["approaches"]:
                        if approach in run["train_metrics"]:
                            for metric in ["r2", "rmse", "mae", "mape"]:
                                approach_statistics[approach]["train"][metric].append(
                                    run["train_metrics"][approach][metric]
                                )
                        if approach in run["test_metrics"]:
                            for metric in ["r2", "rmse", "mae", "mape"]:
                                approach_statistics[approach]["test"][metric].append(
                                    run["test_metrics"][approach][metric]
                                )

                # Calculate statistics for each approach
                approach_stats_summary = {}
                for approach in config["approaches"]:
                    approach_stats_summary[approach] = {
                        "train": {},
                        "test": {}
                    }
                    for dataset in ["train", "test"]:
                        for metric in ["r2", "rmse", "mae", "mape"]:
                            values = approach_statistics[approach][dataset][metric]
                            if values:
                                approach_stats_summary[approach][dataset][metric] = {
                                    "mean": float(np.mean(values)),
                                    "std": float(np.std(values))
                                }

                print(f"\n{'=' * 80}")
                print("PER-APPROACH STATISTICS")
                print(f"{'=' * 80}")
                for approach in config["approaches"]:
                    if approach in approach_stats_summary:
                        print(f"\n{approach} Approach:")
                        if approach_stats_summary[approach]["test"]:
                            test_stats = approach_stats_summary[approach]["test"]
                            if "r2" in test_stats:
                                print(f"  Test R²:   {test_stats['r2']['mean']:.4f} ± {test_stats['r2']['std']:.4f}")
                                print(
                                    f"  Test RMSE: {test_stats['rmse']['mean']:.2f} ± {test_stats['rmse']['std']:.2f}")
                                print(f"  Test MAE:  {test_stats['mae']['mean']:.2f} ± {test_stats['mae']['std']:.2f}")
                                print(
                                    f"  Test MAPE: {test_stats['mape']['mean']:.2f}% ± {test_stats['mape']['std']:.2f}%")

                # Save results
                results = {
                    "parameters": {
                        "generations": generations,
                        "pop_size": pop_size,
                        "intersection_id": intersection_id,
                        "modeling_approach": "lane_level_universal",
                        "n_runs": n_runs,
                        "search_policy": search_policy,
                    },
                    "aggregated_statistics": {
                        "overall": statistics,
                        "per_approach": approach_stats_summary
                    },
                    "individual_runs": all_runs_results
                }

                # Save aggregated results
                filename = f"lane_model_intersection_{intersection_id}_g{generations}_p{pop_size}_aggregated.json"
                output_file = os.path.join(output_path, filename)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                print(f"\n✓ Aggregated results saved to {output_file}")

                # Create summary report
                print(f"\n{'=' * 80}")
                print("FINAL SUMMARY")
                print(f"{'=' * 80}")
                print(f"\nConfiguration: Generations={generations}, Population={pop_size}, Runs={n_runs}")
                print(f"\nOverall Performance (Mean ± Std):")
                print(
                    f"  Train - R²: {statistics['train']['r2']['mean']:.4f} ± {statistics['train']['r2']['std']:.4f}, "
                    f"RMSE: {statistics['train']['rmse']['mean']:.2f} ± {statistics['train']['rmse']['std']:.2f}, "
                    f"MAE: {statistics['train']['mae']['mean']:.2f} ± {statistics['train']['mae']['std']:.2f}, "
                    f"MAPE: {statistics['train']['mape']['mean']:.2f}% ± {statistics['train']['mape']['std']:.2f}%")
                print(f"  Test  - R²: {statistics['test']['r2']['mean']:.4f} ± {statistics['test']['r2']['std']:.4f}, "
                      f"RMSE: {statistics['test']['rmse']['mean']:.2f} ± {statistics['test']['rmse']['std']:.2f}, "
                      f"MAE: {statistics['test']['mae']['mean']:.2f} ± {statistics['test']['mae']['std']:.2f}, "
                      f"MAPE: {statistics['test']['mape']['mean']:.2f}% ± {statistics['test']['mape']['std']:.2f}%")


if __name__ == "__main__":
    main()
