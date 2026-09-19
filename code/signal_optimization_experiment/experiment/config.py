from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NETWORK_DIR = ROOT / "network"
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "models" / "symbolic_lane_model.json"
FIXED_PLAN_PATH = ROOT / "models" / "fixed_plan_intersection1.json"

TLS_ID = "J0"
SATURATION_FLOW = 1800.0  # veh/h/lane, matching the symbolic-regression data
LOST_TIME = 24.0
MIN_GREEN = 5.0
MAX_GREEN = 60.0
MIN_CYCLE = 60.0
MAX_CYCLE = 180.0
ACTUATED_GAP = 2.0
ACTUATED_DETECTOR_DISTANCE = 30.0
MAX_CLEARANCE_SECONDS = 7200
SIMULATION_SCHEMA_VERSION = 2
SUMO_STEP_LENGTH = 1.0
PHASES = ("A", "B", "C", "D")
GREEN_PHASE_INDEX = {"A": 0, "B": 3, "C": 6, "D": 9}
INDEX_TO_PHASE = {index: phase for phase, index in GREEN_PHASE_INDEX.items()}

LANES = (
    "S_R", "S_T", "S_L",
    "N_R", "N_T", "N_L",
    "E_R", "E_T", "E_L",
    "W_R", "W_T", "W_L",
)

PHASE_LANES = {
    "A": ("S_R", "N_R"),
    "B": ("S_T", "S_L", "N_T", "N_L"),
    "C": ("E_R", "W_R"),
    "D": ("E_T", "E_L", "W_T", "W_L"),
}
LANE_PHASE = {
    lane: phase for phase, lanes in PHASE_LANES.items() for lane in lanes
}

ROUTES = {
    "S_R": ("S_in", "E_out"),
    "S_T": ("S_in", "N_out"),
    "S_L": ("S_in", "W_out"),
    "N_R": ("N_in", "W_out"),
    "N_T": ("N_in", "S_out"),
    "N_L": ("N_in", "E_out"),
    "E_R": ("E_in", "N_out"),
    "E_T": ("E_in", "W_out"),
    "E_L": ("E_in", "S_out"),
    "W_R": ("W_in", "S_out"),
    "W_T": ("W_in", "E_out"),
    "W_L": ("W_in", "N_out"),
}

INBOUND_EDGES = ("S_in", "N_in", "E_in", "W_in")
INBOUND_LANES = tuple(f"{edge}_{index}" for edge in INBOUND_EDGES for index in range(3))
