"""System and experimental parameters for the Drone Light Show Depot.

A closed-loop fleet of FLEET_SIZE drones cycles between flying and the
ground depot. While airborne each drone drains its battery; when the
charge drops below RETURN_THRESHOLD the drone lands at the depot, waits
for a swap station, then a test rig, then re-launches with a full pack.

Two configurations are compared in the assignment:
    A — N_SWAP_A swap stations  (the current depot)
    B — N_SWAP_B swap stations  (one additional rack purchased)
The test rig (N_TEST = 1) is shared across both configurations.

All times are in seconds.
"""

# ── Fleet & flight model ────────────────────────────────────────────────
FLEET_SIZE       = 200      # total drones in the show
FLIGHT_FULL_DUR  = 2400.0   # seconds of flight on a full battery (40 min @ 100%)
RETURN_THRESHOLD = 0.15     # drone returns when residual charge ≤ this

# Per-flight battery drain rate is randomized to model wind, altitude
# changes, and varying choreography intensity. A LogNormal on the
# "drain multiplier" works well: mean ~ 1.0, mild CV.
DRAIN_MEAN       = 1.0
DRAIN_CV         = 0.18

# ── Depot service ───────────────────────────────────────────────────────
# The swap station is the intended bottleneck. The test rig is fast
# and lightly loaded, so adding swap capacity actually reduces depot
# congestion — the policy comparison has bite.
SWAP_MEAN        = 45.0     # mean swap time (s) — replace battery + reseat
SWAP_CV          = 0.30
TEST_MEAN        = 8.0      # mean airworthiness test time (s) at the test rig
TEST_CV          = 0.25

# ── Configurations (A vs B) ─────────────────────────────────────────────
N_SWAP_A         = 5
N_SWAP_B         = 6
N_TEST           = 1        # shared across both configurations

# Convenience: the two policies as named tuples consumed by run.py
POLICIES = {
    "A": {"n_swap": N_SWAP_A, "n_test": N_TEST, "label": "A — 5 swap stations"},
    "B": {"n_swap": N_SWAP_B, "n_test": N_TEST, "label": "B — 6 swap stations"},
}

# ── Experiment length ───────────────────────────────────────────────────
# A rehearsal block is 6 hours, but for the output-analysis tasks we run
# longer windows so the steady-state regime is reachable.
SIM_DUR          = 12 * 3600.0     # seconds (12 h) — one replication
LONG_RUN_DUR     = 30 * 3600.0     # seconds (30 h) — single long run for MSER / batch means

# Replication-based analysis defaults (students may tune these in Task 3).
N_REPS_DEFAULT   = 20
WARMUP_DEFAULT   = 3600.0          # 1 h initial deletion (students must justify)

# Batch-means defaults — count is in completions, not seconds.
N_BATCHES_DEFAULT  = 30
BATCH_SIZE_DEFAULT = 300           # completions per batch

# ── Reproducibility ─────────────────────────────────────────────────────
# The canonical master seed used for all teams (autograder calibrates to it).
MASTER_SEED      = 20260601
