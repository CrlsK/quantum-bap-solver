"""
Quantum QUBO/SQA BAP+QCA Solver v4.0
Berth Allocation + Quay Crane Assignment via QUBO formulation
with Simulated Quantum Annealing (Suzuki-Trotter decomposition).

v3.0 changes:
- Fixed QUBO cost scaling: removed /(n_vessels*2) divisor that made objectives negligible
- Adaptive penalty scaling: penalties set relative to estimated max objective cost
- Post-SQA greedy repair for unassigned vessels (ensures feasibility)
- Berth time-overlap penalty to prevent two vessels occupying the same berth simultaneously
- Increased default SQA sweeps to 1000 for better convergence
- Enhanced rich visual output for benchmarking dashboards

v3.1 changes (iteration 1):
- Stronger overlap penalty: penalty_overlap changed from max_cost * 1.5 to max_cost * 4.0
- Improved vessel spreading across berths to reduce congestion
- Added interactive Plotly.js visualizations with dark purple quantum theme
- Generated 5 standalone HTML visualization files for comprehensive analysis

v4.0 changes (iteration 2):
- CRITICAL: Fixed HTML template formatting bug (% char conflicts with CSS widths/margins)
- Reduced default SQA sweeps from 1000 to 500 for faster convergence
- Increased Trotter slices from 20 to 30 for better quantum tunneling representation
- Added warm-start initialization: first replica initialized with greedy-like solution
- Adjusted temperature schedule: T_init=5.0, T_final=0.01 (less aggressive annealing)
- Improved greedy repair: sorted vessels by priority, best-fit berth selection
- Added post-SQA 2-opt berth swap phase for local optimization
- Enhanced QUBO encoding: waiting cost penalties for busy berths, berth-balancing penalties
"""
import logging
import time
import math
import random
import os
import json

logger = logging.getLogger("qcentroid-user-log")


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    start_time = time.time()
    logger.info("=== Quantum QUBO/SQA BAP+QCA Solver v4.0 ===")

    # ── 1. Parse inputs ──────────────────────────────────────────────
    vessels = input_data.get("vessels", [])
    berths = input_data.get("berths", [])
    cranes_cfg = input_data.get("cranes", {})
    cost_weights = input_data.get("cost_weights", {})

    total_cranes = cranes_cfg.get("total_available", 10)
    min_cranes = cranes_cfg.get("min_per_vessel", 1)
    max_cranes = cranes_cfg.get("max_per_vessel", 4)

    w_wait = cost_weights.get("waiting_cost_per_hour", 500)
    w_handle = cost_weights.get("handling_cost_per_crane_hour", 150)
    w_delay = cost_weights.get("delay_penalty_per_hour", 1000)
    w_priority = cost_weights.get("priority_multiplier", 1.5)

    n_vessels = len(vessels)
    n_berths = len(berths)
    logger.info(f"Problem: {n_vessels} vessels, {n_berths} berths, {total_cranes} cranes")

    # SQA parameters (v4.0: reduced sweeps to 500, increased Trotter to 30, adjusted temps)
    n_trotter = solver_params.get("trotter_slices", 30)
    n_sweeps = solver_params.get("sqa_sweeps", 500)
    T_init = solver_params.get("temperature_init", 5.0)
    T_final = solver_params.get("temperature_final", 0.01)
    gamma_init = solver_params.get("transverse_field_init", 5.0)
    gamma_final = solver_params.get("transverse_field_final", 0.005)
    seed = solver_params.get("seed", 42)
    random.seed(seed)

    # ── 2. Estimate cost scale for adaptive penalty tuning ───────────