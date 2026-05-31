"""SimPy model for the Drone Light Show Battery Swap Depot.

Closed-loop population of FLEET_SIZE drones. Each drone executes:
    fly until battery ≤ RETURN_THRESHOLD
        → land at depot
        → wait + swap battery (one of N_SWAP stations)
        → wait + test (one of N_TEST rigs)
        → take off with a fresh battery

Two configurations differ in N_SWAP; everything else is identical and
CRN-paired across the two configurations.
"""

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import simpy

from config import (
    FLEET_SIZE, FLIGHT_FULL_DUR, RETURN_THRESHOLD,
    DRAIN_MEAN, DRAIN_CV,
    SWAP_MEAN, SWAP_CV, TEST_MEAN, TEST_CV,
    LONG_RUN_DUR,
)
from seeds import make_streams, Streams


# ═══════════════════════════════════════════════════════════════════════════
#  Helper: lognormal parameters from mean + CV
# ═══════════════════════════════════════════════════════════════════════════

def _lognormal_params(mean: float, cv: float) -> Tuple[float, float]:
    """Convert (mean, CV) of a lognormal random variable to (mu_N, sigma_N).

    See the convention used throughout IE 306: mu_N and sigma_N parameterise
    the *underlying Normal*, not the lognormal moments.
    """
    sigma2 = math.log(1.0 + cv * cv)
    return math.log(mean) - 0.5 * sigma2, math.sqrt(sigma2)


SWAP_MU, SWAP_SIG   = _lognormal_params(SWAP_MEAN, SWAP_CV)
TEST_MU, TEST_SIG   = _lognormal_params(TEST_MEAN, TEST_CV)
DRAIN_MU, DRAIN_SIG = _lognormal_params(DRAIN_MEAN, DRAIN_CV)


# ═══════════════════════════════════════════════════════════════════════════
#  State container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DepotState:
    """All mutable replication state lives here so the simulation logic
    is a thin layer over the data."""

    env: simpy.Environment
    streams: Streams
    swap: simpy.Resource
    test: simpy.Resource

    # Aggregate counts — initialised by the runner BEFORE drone processes
    # spawn so the invariant FLEET_SIZE = in_air + in_depot holds from t=0.
    in_air: int = 0
    in_depot: int = 0

    # Per-completion observations (collected at depot exit / re-launch).
    completions: List[Tuple[float, float, float, float]] = field(default_factory=list)
    # Each tuple: (t_landed, swap_wait, test_wait, sojourn)

    # Time series of "drones in air" sampled every SAMPLE_INTERVAL seconds.
    air_samples: List[Tuple[float, int]] = field(default_factory=list)

    # Invariant tracker (V&V): max absolute deviation from
    #     FLEET_SIZE = in_air + in_depot
    max_invariant_violation: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  Drone lifecycle
# ═══════════════════════════════════════════════════════════════════════════

def drone(env: simpy.Environment, state: DepotState, drone_id: int,
          initial_battery: float):
    """Single drone process: cycle fly → return → swap → test → fly ….

    The runner pre-credits in_air = FLEET_SIZE *before* spawning drone
    processes, so we do not bump it here.
    """
    battery = initial_battery
    while True:
        # ── In flight ──────────────────────────────────────────────────
        drain_mult = state.streams.drain.lognormal(DRAIN_MU, DRAIN_SIG)
        # Time to drain from `battery` down to RETURN_THRESHOLD.
        delta_charge = max(battery - RETURN_THRESHOLD, 1e-9)
        flight_dur = (delta_charge * FLIGHT_FULL_DUR) / drain_mult
        yield env.timeout(flight_dur)

        # ── Returned to depot ──────────────────────────────────────────
        t_landed = env.now
        state.in_air -= 1
        state.in_depot += 1
        _check_invariant(state)

        # ── Swap station ───────────────────────────────────────────────
        with state.swap.request() as req:
            yield req
            t_swap_start = env.now
            swap_dur = state.streams.swap.lognormal(SWAP_MU, SWAP_SIG)
            yield env.timeout(swap_dur)
        swap_wait = t_swap_start - t_landed

        # ── Test rig ───────────────────────────────────────────────────
        with state.test.request() as req:
            yield req
            t_test_start = env.now
            test_dur = state.streams.test.lognormal(TEST_MU, TEST_SIG)
            yield env.timeout(test_dur)
        test_wait = t_test_start - (t_swap_start + swap_dur)

        # ── Re-launch ──────────────────────────────────────────────────
        sojourn = env.now - t_landed
        state.completions.append((t_landed, swap_wait, test_wait, sojourn))
        state.in_depot -= 1
        state.in_air += 1
        _check_invariant(state)

        battery = 1.0   # full pack


def _check_invariant(state: DepotState) -> None:
    diff = abs(FLEET_SIZE - (state.in_air + state.in_depot))
    if diff > state.max_invariant_violation:
        state.max_invariant_violation = diff


def air_sampler(env: simpy.Environment, state: DepotState,
                interval: float):
    """Sample in_air every `interval` seconds (for fleet-coverage analysis)."""
    while True:
        state.air_samples.append((env.now, state.in_air))
        yield env.timeout(interval)


# ═══════════════════════════════════════════════════════════════════════════
#  Replication runner
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_INTERVAL = 5.0    # seconds, for the air-count time series


def run_replication(n_swap: int, n_test: int, master_seed: int,
                    rep_index: int, duration: float = LONG_RUN_DUR
                    ) -> DepotState:
    """Run one rehearsal block and return the populated state object."""
    env = simpy.Environment()
    streams = make_streams(master_seed, rep_index)

    swap_res = simpy.Resource(env, capacity=n_swap)
    test_res = simpy.Resource(env, capacity=n_test)

    state = DepotState(env=env, streams=streams,
                       swap=swap_res, test=test_res,
                       in_air=FLEET_SIZE, in_depot=0)

    # Initial battery distribution: every drone launches with a *full* pack
    # at t = 0 (the rehearsal block begins). With identical batteries the
    # first wave of returns clusters around t ≈ FLIGHT_FULL_DUR, creating
    # the warmup transient students must detect in Task 2. The drain
    # multiplier (lognormal) provides enough spread that returns are not
    # all simultaneous.
    _ = state.streams.init.random()    # consume one draw to keep stream aligned

    for i in range(FLEET_SIZE):
        env.process(drone(env, state, i, 1.0))

    env.process(air_sampler(env, state, SAMPLE_INTERVAL))
    env.run(until=duration)
    return state


# ═══════════════════════════════════════════════════════════════════════════
#  Output helpers
# ═══════════════════════════════════════════════════════════════════════════

def swap_wait_series(state: DepotState):
    """Return (t_landed array, swap_wait array) sorted by landing time."""
    if not state.completions:
        return np.array([]), np.array([])
    arr = np.asarray(state.completions, dtype=float)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1]


def air_count_series(state: DepotState):
    if not state.air_samples:
        return np.array([]), np.array([])
    arr = np.asarray(state.air_samples, dtype=float)
    return arr[:, 0], arr[:, 1]
