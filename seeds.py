"""Independent RNG streams via SeedSequence.spawn.

CRN works only if streams are paired across paired replications. We spawn
one child SeedSequence per source of randomness, derived deterministically
from a master seed + replication index. Two replications with the same
(master_seed, rep_index) share every stream — change only the *policy* and
you have a Common-Random-Numbers comparison.

Streams used by the depot model:
    drain  — per-flight battery drain multiplier
    swap   — swap service time
    test   — test rig service time
    init   — initial battery distribution at t=0 (fleet ramp-up)
"""

from dataclasses import dataclass
import numpy as np


STREAM_NAMES = ("drain", "swap", "test", "init")


@dataclass
class Streams:
    drain: np.random.Generator
    swap: np.random.Generator
    test: np.random.Generator
    init: np.random.Generator


def make_streams(master_seed: int, rep_index: int) -> Streams:
    """Spawn one Generator per source of randomness, paired across policies.

    Calling this with the same (master_seed, rep_index) twice — once for
    Policy A and once for Policy B — yields identical streams.  That is
    the CRN invariant the assignment relies on.
    """
    ss = np.random.SeedSequence([master_seed, rep_index])
    children = ss.spawn(len(STREAM_NAMES))
    gens = [np.random.default_rng(c) for c in children]
    return Streams(**dict(zip(STREAM_NAMES, gens)))
