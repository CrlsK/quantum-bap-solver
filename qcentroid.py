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
