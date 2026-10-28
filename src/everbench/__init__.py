"""
everbench: Contamination-Proof Benchmark, Stockfish Fishtest-style SPRT Gate,
and TIES/DARE Orthogonal Parameter Merger for Verifiable AI.
"""

from .seed_protocol import (
    EpochSeedProtocol,
    BenchmarkInstanceSpec,
    VERIFIABLE_DOMAINS,
)
from .sprt import (
    SPRTConfig,
    SPRTStatus,
    SPRTResult,
    MultiDomainGateReport,
    SPRTGate,
    estimate_elo_difference,
)
from .merger import (
    ParameterDelta,
    TIESMerger,
    DAREMerger,
)

__all__ = [
    "EpochSeedProtocol",
    "BenchmarkInstanceSpec",
    "VERIFIABLE_DOMAINS",
    "SPRTConfig",
    "SPRTStatus",
    "SPRTResult",
    "MultiDomainGateReport",
    "SPRTGate",
    "estimate_elo_difference",
    "ParameterDelta",
    "TIESMerger",
    "DAREMerger",
]
