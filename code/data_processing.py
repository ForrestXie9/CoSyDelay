# data_processing.py
import os
import json
import re
import pandas as pd
import numpy as np
import ast
from typing import List
from constants import SAT_FLOW, INTERSECTION_CONFIGS


def parse_dialogue_flexible(dialogue: str, intersection_id: int) -> List[int]:
    """Parse dialogue based on intersection configuration"""
    config = INTERSECTION_CONFIGS[intersection_id]

    traffic_pattern = r'(South|East|North|West)\s+Approach:.*?Left Turn\s*=\s*(\d+),\s*Through\s*=\s*(\d+),\s*Right Turn\s*=\s*(\d+)'
    phase_pattern = r'Phase\s+([ABCD]):\s*(\d+)'


    traffic_matches = re.findall(traffic_pattern, dialogue, re.DOTALL)
    phase_matches = re.findall(phase_pattern, dialogue)

    # Verify we have the right number of approaches and phases
    expected_approaches = len(config["approaches"])
    expected_phases = len(config["phases"])

    # if len(traffic_matches) != expected_approaches:
    #     raise ValueError(f"Expected {expected_approaches} traffic approaches, got {len(traffic_matches)}")
    # if len(phase_matches) != expected_phases:
    #     raise ValueError(f"Expected {expected_phases} phases, got {len(phase_matches)}")

    # Extract traffic values in fixed order: S, E, N, W (0 if approach doesn't exist)
    traffic_values = []
    approach_order = ["South", "East", "North", "West"]

    for approach in approach_order:
        found = False
        for match in traffic_matches:
            if match[0] == approach:
                traffic_values.extend([int(match[1]), int(match[2]), int(match[3])])
                found = True
                break
        if not found:
            traffic_values.extend([0, 0, 0])  # No traffic if approach doesn't exist

    # Extract phase values in order A, B, C, D (0 if phase doesn't exist)
    phase_values = []
    for phase in ["A", "B", "C", "D"]:
        found = False
        for match in phase_matches:
            if match[0] == phase:
                phase_values.append(int(match[1]))
                found = True
                break
        if not found:
            phase_values.append(0)  # 0 if phase doesn't exist

    return traffic_values + phase_values




def parse_dialogue(dialogue: str) -> List[int]:
    """Legacy function for backward compatibility"""
    return parse_dialogue_flexible(dialogue, 1)


def load_dataset_flexible(jsonl_path: str, intersection_id: int) -> pd.DataFrame:
    """Load dataset with intersection-specific parsing"""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")

    config = INTERSECTION_CONFIGS[intersection_id]
    data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:

            item = json.loads(line.strip())
            features = parse_dialogue_flexible(item["dialogue"], intersection_id)
            labels = ast.literal_eval(item["summary"])
            data.append(features + labels)


    # Create columns based on intersection configuration
    columns = []

    # Traffic columns - always in S, E, N, W order
    for approach in ["S", "E", "N", "W"]:
        for movement in ["L", "T", "R"]:
            columns.append(f"{approach}_{movement}")

    # Phase columns - always A, B, C, D
    for phase in ["A", "B", "C", "D"]:
        columns.append(f"Green_{phase}")

    # Delay columns - only for existing approaches
    for approach in ["S", "E", "N", "W"]:
        columns.append(f"Delay_{approach}")

    return pd.DataFrame(data, columns=columns)


def load_dataset(jsonl_path: str) -> pd.DataFrame:
    """Legacy function for backward compatibility"""
    return load_dataset_flexible(jsonl_path, 1)


def preprocess_data_flexible(df: pd.DataFrame, intersection_id: int) -> pd.DataFrame:
    """Preprocess data based on intersection configuration"""
    df = df.copy()
    config = INTERSECTION_CONFIGS[intersection_id]

    # Calculate cycle time using existing phases
    phase_columns = [f"Green_{phase}" for phase in config["phases"]]
    df["Cycle_Time"] = df[phase_columns].sum(axis=1) + config["cycle_offset"]

    # Calculate green ratios for existing phases
    for phase in config["phases"]:
        df[f"GR_{phase}"] = df[f"Green_{phase}"] / df["Cycle_Time"]

    # Set non-existing phases to 0
    for phase in ["A", "B", "C", "D"]:
        if phase not in config["phases"]:
            df[f"GR_{phase}"] = 0.0

    # Calculate flow ratios for existing movements
    for approach in ["S", "E", "N", "W"]:
        if approach in config["approaches"]:
            for movement in config["movements"][approach]:
                lane = f"{approach}_{movement}"
                df[f"flow_{lane}"] = df[lane] / SAT_FLOW
        else:
            # Set flow ratios to 0 for non-existing approaches/movements
            for movement in ["L", "T", "R"]:
                lane = f"{approach}_{movement}"
                df[f"flow_{lane}"] = 0.0

    # Handle missing movements within existing approaches
    for approach in config["approaches"]:
        for movement in ["L", "T", "R"]:
            if movement not in config["movements"][approach]:
                lane = f"{approach}_{movement}"
                df[f"flow_{lane}"] = 0.0

    # Create lane-phase mapping based on intersection configuration
    lane_phase_map = {}
    for phase, phase_movements in config["phase_movements"].items():
        for approach, movements in phase_movements.items():
            if approach in config["approaches"]:
                for movement in movements:
                    lane = f"{approach}_{movement}"
                    lane_phase_map[lane] = phase

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Legacy function for backward compatibility"""
    return preprocess_data_flexible(df, 1)