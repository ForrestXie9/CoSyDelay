# constants.py
SAT_FLOW = 1800
DEFAULT_CYCLE_OFFSET = 24

# Intersection-specific configurations
INTERSECTION_CONFIGS = {
    1: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T", "R"], "E": ["L", "T", "R"],
            "N": ["L", "T", "R"], "W": ["L", "T", "R"]
        },
        "phases": ["A", "B", "C", "D"],
        "cycle_offset": 24,
        "phase_movements": {
            "A": {"S": ["R"], "N": ["R"]},
            "B": {"S": ["L", "T"], "N": ["L", "T"]},
            "C": {"E": ["R"], "W": ["R"]},
            "D": {"E": ["L", "T"], "W": ["L", "T"]}
        }
    },
    2: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T", "R"], "E": ["L", "T", "R"],
            "N": ["L", "T", "R"], "W": ["L", "T", "R"]
        },
        "phases": ["A", "B", "C", "D"],
        "cycle_offset": 24,
        "phase_movements": {
            "A": {"S": ["R"], "N": ["R"]},
            "B": {"S": ["L", "T"], "N": ["L", "T"]},
            "C": {"E": ["R"], "W": ["R"]},
            "D": {"E": ["L", "T"], "W": ["L", "T"]}
        }
    },
    3: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T"], "E": ["L", "T"],
            "N": ["L", "T", "R"], "W": ["L", "T"]

        },
        "phases": ["A", "B", "C"],
        "cycle_offset": 18,  # 3 phases * 6 seconds (3+3)
        "phase_movements": {
            "A": {"S": ["L", "T"], "N": ["L", "T"], "W": ["L"]},
            "B": {"N": ["L", "T", "R"], "W": ["L"]},
            "C": {"E": ["L", "T"], "W": ["L", "T"]}
        }
    },
    4: {
        "approaches": ["S", "N", "W"],  # No East approach
        "movements": {
            "S": ["L", "T"], "N": ["T", "R"], "W": ["L", "R"]
        },
        "phases": ["A", "B", "C"],
        "cycle_offset": 18,
        "phase_movements": {
            "A": {"S": ["L", "T"], "N": ["T"]},
            "B": {"S": ["L"], "N": ["T", "R"], "W": ["L"]},
            "C": {"S": ["L"], "W": ["L", "R"]}
        }
    },
    5: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T"], "E": ["L", "T", "R"],
            "N": ["T"], "W": ["L", "R"]
        },
        "phases": ["A", "B", "C"],
        "cycle_offset": 18,
        "phase_movements": {
            "A": {"E": ["L", "R"], "W": ["L", "R"]},
            "B": {"E": ["L", "T", "R"], "W": ["L"]},
            "C": {"S": ["L", "T"], "N": ["T"], "W": ["L"]}
        }
    },
    6: {
        "approaches": ["S", "E", "N"],  # No West approach
        "movements": {
            "S": ["T", "R"], "E": ["L"], "N": ["L", "T"]
        },
        "phases": ["A", "B"],  # No phases C and D
        "cycle_offset": 12,  # 2 phases * 6 seconds (3+3)
        "phase_movements": {
            "A": {"S": ["T"], "N": ["L", "T"]},
            "B": {"S": ["T", "R"], "E": ["L"]}
        }
    },
    7: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T", "R"], "E": ["L", "T", "R"],
            "N": ["L", "T", "R"], "W": ["L", "T", "R"]
        },
        "phases": ["A", "B", "C", "D"],
        "cycle_offset": 24,
        "phase_movements": {
            "A": {"S": ["R"], "N": ["R"]},
            "B": {"S": ["L", "T"], "N": ["L", "T"]},
            "C": {"E": ["R"], "W": ["R"]},
            "D": {"E": ["L", "T"], "W": ["L", "T"]}
        }
    },
    8: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T", "R"], "E": ["L", "T", "R"],
            "N": ["L", "T", "R"], "W": ["L", "T", "R"]
        },
        "phases": ["A", "B", "C", "D"],
        "cycle_offset": 24,
        "phase_movements": {
            "A": {"S": ["R"], "N": ["R"]},
            "B": {"S": ["L", "T"], "N": ["L", "T"]},
            "C": {"E": ["R"], "W": ["R"]},
            "D": {"E": ["L", "T"], "W": ["L", "T"]}
        }
    },
    9: {
        "approaches": ["S", "E", "N", "W"],
        "movements": {
            "S": ["L", "T", "R"], "E": ["L", "T", "R"],
            "N": ["L", "T", "R"], "W": ["L", "T", "R"]
        },
        "phases": ["A", "B", "C", "D"],
        "cycle_offset": 24,
        "phase_movements": {
            "A": {"S": ["R"], "N": ["R"]},
            "B": {"S": ["L", "T"], "N": ["L", "T"]},
            "C": {"E": ["R"], "W": ["R"]},
            "D": {"E": ["L", "T"], "W": ["L", "T"]}
        }
    },
    # Reserved external-validation adapter.  Unlike I1--I6, DLR UT has
    # approach-level observed delay only; ``T`` is an internal single-lane
    # placeholder and must never be interpreted as a labelled through movement.
    98: {
        "approaches": ["N", "E", "S", "W"],
        "movements": {
            "N": ["T"], "E": ["T"], "S": ["T"], "W": ["T"]
        },
        "phases": ["A", "B", "C", "D"],
        "cycle_offset": 0,
        "phase_movements": {
            "A": {"N": ["T"]},
            "B": {"E": ["T"]},
            "C": {"S": ["T"]},
            "D": {"W": ["T"]},
        },
    },
    99: {
        "approaches": ["E", "S", "W"],
        "movements": {
            "E": ["T"], "S": ["T"], "W": ["T"]
        },
        "phases": ["A", "B", "C"],
        "cycle_offset": 0,
        "phase_movements": {
            "A": {"E": ["T"]},
            "B": {"S": ["T"]},
            "C": {"W": ["T"]},
        },
    },
}

feature_explanations = {
    # Traffic volume explanations - 这些是通用的，每个方向转向的含义不变
    "S_L": "Traffic volume for southbound left-turn movement (vehicles/hour)",
    "S_T": "Traffic volume for southbound through movement (vehicles/hour)",
    "S_R": "Traffic volume for southbound right-turn movement (vehicles/hour)",
    "N_L": "Traffic volume for northbound left-turn movement (vehicles/hour)",
    "N_T": "Traffic volume for northbound through movement (vehicles/hour)",
    "N_R": "Traffic volume for northbound right-turn movement (vehicles/hour)",
    "E_L": "Traffic volume for eastbound left-turn movement (vehicles/hour)",
    "E_T": "Traffic volume for eastbound through movement (vehicles/hour)",
    "E_R": "Traffic volume for eastbound right-turn movement (vehicles/hour)",
    "W_L": "Traffic volume for westbound left-turn movement (vehicles/hour)",
    "W_T": "Traffic volume for westbound through movement (vehicles/hour)",
    "W_R": "Traffic volume for westbound right-turn movement (vehicles/hour)",

    # Generic explanations - 不指定具体控制的转向，因为每个交叉口不同
    "GR_A": "Green ratio for Phase A (green time for Phase A/cycle time, unitless)",
    "GR_B": "Green ratio for Phase B (green time for Phase B/cycle time, unitless)",
    "GR_C": "Green ratio for Phase C (green time for Phase C/cycle time, unitless)",
    "GR_D": "Green ratio for Phase D (green time for Phase D/cycle time, unitless)",

    "Cycle_Time": "Total signal cycle time including all phases and clearance intervals (seconds)",

    # Generic flow ratio explanations - 不指定具体控制的phase，因为每个交叉口不同
    "flow_S_L": "Flow ratio for southbound left-turn lane (southbound left-turn volume/SAT_FLOW, unitless)",
    "flow_S_T": "Flow ratio for southbound through lane (southbound through volume/SAT_FLOW, unitless)",
    "flow_S_R": "Flow ratio for southbound right-turn lane (southbound right-turn volume/SAT_FLOW, unitless)",
    "flow_N_L": "Flow ratio for northbound left-turn lane (northbound left-turn volume/SAT_FLOW, unitless)",
    "flow_N_T": "Flow ratio for northbound through lane (northbound through volume/SAT_FLOW, unitless)",
    "flow_N_R": "Flow ratio for northbound right-turn lane (northbound right-turn volume/SAT_FLOW, unitless)",
    "flow_E_L": "Flow ratio for eastbound left-turn lane (eastbound left-turn volume/SAT_FLOW, unitless)",
    "flow_E_T": "Flow ratio for eastbound through lane (eastbound through volume/SAT_FLOW, unitless)",
    "flow_E_R": "Flow ratio for eastbound right-turn lane (eastbound right-turn volume/SAT_FLOW, unitless)",
    "flow_W_L": "Flow ratio for westbound left-turn lane (westbound left-turn volume/SAT_FLOW, unitless)",
    "flow_W_T": "Flow ratio for westbound through lane (westbound through volume/SAT_FLOW, unitless)",
    "flow_W_R": "Flow ratio for westbound right-turn lane (westbound right-turn volume/SAT_FLOW, unitless)",
}


def get_intersection_specific_feature_explanations(intersection_id: int) -> dict:
    """
    Generate intersection-specific feature explanations based on actual phase-movement mappings.
    Now supports movements controlled by multiple phases.
    """
    config = INTERSECTION_CONFIGS[intersection_id]
    explanations = {}

    # Base explanations for traffic volumes
    for approach in ["S", "E", "N", "W"]:
        approach_name = {"S": "southbound", "E": "eastbound", "N": "northbound", "W": "westbound"}[approach]
        for movement in ["L", "T", "R"]:
            movement_name = {"L": "left-turn", "T": "through", "R": "right-turn"}[movement]
            explanations[
                f"{approach}_{movement}"] = f"Traffic volume for {approach_name} {movement_name} movement (vehicles/hour)"

    # Phase-specific green ratio explanations
    for phase in config["phases"]:
        controlled_movements = []
        for approach, movements in config["phase_movements"][phase].items():
            approach_name = {"S": "southbound", "E": "eastbound", "N": "northbound", "W": "westbound"}[approach]
            for movement in movements:
                movement_name = {"L": "left-turn", "T": "through", "R": "right-turn"}[movement]
                controlled_movements.append(f"{approach_name} {movement_name}")

        movement_list = ", ".join(controlled_movements)
        explanations[
            f"GR_{phase}"] = f"Green ratio for Phase {phase} (green time for Phase {phase}/cycle time, unitless)"

    # Set unused phases to zero explanation
    for phase in ["A", "B", "C", "D"]:
        if phase not in config["phases"]:
            explanations[
                f"GR_{phase}"] = f"Green ratio for Phase {phase} - not used in intersection {intersection_id} (green time for Phase {phase}/cycle time, always 0, unitless)"

    # Cycle time explanation
    active_phases = ", ".join(config["phases"])
    explanations[
        "Cycle_Time"] = f"Total signal cycle time for intersection {intersection_id} including phases {active_phases} and clearance intervals (seconds)"

    # Flow ratio explanations with phase control information (支持多个phase控制)
    for approach in ["S", "E", "N", "W"]:
        approach_name = {"S": "southbound", "E": "eastbound", "N": "northbound", "W": "westbound"}[approach]

        if approach in config["approaches"]:
            available_movements = config["movements"][approach]
        else:
            available_movements = []

        for movement in ["L", "T", "R"]:
            movement_name = {"L": "left-turn", "T": "through", "R": "right-turn"}[movement]

            # 找到所有控制此转向的phase
            controlling_phases = []
            if approach in config.get("approaches", []):
                for phase, phase_movements in config["phase_movements"].items():
                    if approach in phase_movements and movement in phase_movements[approach]:
                        controlling_phases.append(phase)

            if movement in available_movements and controlling_phases:
                # 构建具体的流量描述
                volume_description = f"{approach_name} {movement_name} volume"

                if len(controlling_phases) == 1:
                    # 单个phase控制
                    explanations[f"flow_{approach}_{movement}"] = (
                        f"Flow ratio for {approach_name} {movement_name} lane - "
                        f"controlled by Phase {controlling_phases[0]} ({volume_description}/SAT_FLOW, unitless)"
                    )
                else:
                    # 多个phase控制
                    phase_list = ", ".join(sorted(controlling_phases))
                    explanations[f"flow_{approach}_{movement}"] = (
                        f"Flow ratio for {approach_name} {movement_name} lane - "
                        f"controlled by Phases {phase_list} ({volume_description}/SAT_FLOW, unitless)"
                    )
            else:
                explanations[f"flow_{approach}_{movement}"] = (
                    f"Flow ratio for {approach_name} {movement_name} lane - "
                    f"not available in intersection {intersection_id} (always 0, unitless)"
                )

    return explanations


def get_controlling_phases(intersection_id: int, approach: str, movement: str) -> list:
    """
    获取控制特定转向的所有phase

    Args:
        intersection_id: 交叉口ID
        approach: 方向 ("S", "E", "N", "W")
        movement: 转向 ("L", "T", "R")

    Returns:
        list: 控制该转向的所有phase列表
    """
    config = INTERSECTION_CONFIGS[intersection_id]
    controlling_phases = []

    if approach in config.get("approaches", []):
        for phase, phase_movements in config["phase_movements"].items():
            if approach in phase_movements and movement in phase_movements[approach]:
                controlling_phases.append(phase)

    return sorted(controlling_phases)


def get_phase_controlled_movements(intersection_id: int, phase: str) -> dict:
    """
    获取特定phase控制的所有转向

    Args:
        intersection_id: 交叉口ID
        phase: 相位 ("A", "B", "C", "D")

    Returns:
        dict: 该phase控制的转向，格式为 {approach: [movements]}
    """
    config = INTERSECTION_CONFIGS[intersection_id]

    if phase in config.get("phase_movements", {}):
        return config["phase_movements"][phase].copy()
    else:
        return {}


def get_movement_phase_mapping(intersection_id: int) -> dict:
    """
    获取完整的转向-相位映射关系

    Args:
        intersection_id: 交叉口ID

    Returns:
        dict: 映射关系，格式为 {(approach, movement): [phases]}
    """
    config = INTERSECTION_CONFIGS[intersection_id]
    mapping = {}

    for approach in ["S", "E", "N", "W"]:
        for movement in ["L", "T", "R"]:
            controlling_phases = get_controlling_phases(intersection_id, approach, movement)
            if controlling_phases:
                mapping[(approach, movement)] = controlling_phases

    return mapping
