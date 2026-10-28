"""
Orthogonal Parameter Merging (`TIESMerger` and `DAREMerger`).

Enables decentralized contributors who discover improvements in different verifiable
domains (`logic`, `silicon`, `systems`, `matter`) to submit portable `ParameterDelta`
pull requests (LoRA adapters, folded steering vectors, or dense layer deltas) and
fuse them into a single master checkpoint without destructive interference or
catastrophic forgetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Dict, List, Mapping, Sequence, Tuple


@dataclass
class ParameterDelta:
    """
    A contributor's portable parameter update PR (`Delta theta = theta_finetuned - theta_base`).
    """

    contributor_id: str
    domain: str
    tensors: Dict[str, List[float]]
    sprt_elo_gain: float = 0.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def l2_norm(self) -> float:
        total = 0.0
        for vec in self.tensors.values():
            total += sum(v * v for v in vec)
        return math.sqrt(total)


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


class TIESMerger:
    """
    TIES-Merging: Trim, Elect Sign, and Disjoint Merge (Yadav et al., NeurIPS 2023).

    1. **Trim:** For each contributor's task vector `tau_k`, keep only the top-`density`
       fraction of parameters by magnitude and reset the rest to `0`.
    2. **Elect Sign:** For each parameter index `p`, compute the aggregate sign
       `gamma_p = sgn(sum_k w_k * tau_k[p])` across all contributors.
    3. **Disjoint Merge:** Average only the non-zero parameter updates whose sign agrees
       with the elected consensus sign `gamma_p`.
    """

    def __init__(self, density: float = 0.5, scale: float = 1.0) -> None:
        if not (0.0 < density <= 1.0):
            raise ValueError("density must be in (0.0, 1.0].")
        self.density = density
        self.scale = scale

    def trim_vector(self, vec: Sequence[float]) -> List[float]:
        """Keep only the top-`density` fraction of elements by absolute magnitude."""
        n = len(vec)
        if n == 0:
            return []
        keep_k = max(1, int(math.ceil(n * self.density)))
        indexed = sorted(enumerate(vec), key=lambda item: abs(item[1]), reverse=True)
        out = [0.0] * n
        for idx, val in indexed[:keep_k]:
            out[idx] = val
        return out

    def merge(
        self,
        deltas: Sequence[ParameterDelta],
        elo_weighted: bool = True,
    ) -> ParameterDelta:
        """
        Fuse multiple contributors' `ParameterDelta` PRs into a single master update.
        """
        if not deltas:
            raise ValueError("deltas must be a non-empty sequence.")

        tensor_names = list(deltas[0].tensors.keys())
        for d in deltas[1:]:
            if set(d.tensors.keys()) != set(tensor_names):
                raise ValueError("All ParameterDeltas must share the same tensor keys.")

        # Contributor weights (optionally weighted by positive SPRT Elo gain)
        weights: List[float] = []
        for d in deltas:
            if elo_weighted and d.sprt_elo_gain > 0.0:
                weights.append(1.0 + math.log1p(d.sprt_elo_gain))
            else:
                weights.append(1.0)

        merged_tensors: Dict[str, List[float]] = {}
        for name in tensor_names:
            dim = len(deltas[0].tensors[name])
            if any(len(d.tensors[name]) != dim for d in deltas):
                raise ValueError(f"Dimension mismatch for tensor '{name}'.")

            trimmed_list = [self.trim_vector(d.tensors[name]) for d in deltas]
            merged_vec = [0.0] * dim

            for idx in range(dim):
                weighted_mass = sum(
                    w * t[idx] for w, t in zip(weights, trimmed_list)
                )
                if abs(weighted_mass) < 1e-15:
                    continue
                elected_sign = 1.0 if weighted_mass > 0.0 else -1.0

                aligned_vals: List[float] = []
                aligned_weights: List[float] = []
                for w, t in zip(weights, trimmed_list):
                    val = t[idx]
                    if val != 0.0 and (val > 0.0) == (elected_sign > 0.0):
                        aligned_vals.append(w * val)
                        aligned_weights.append(w)

                if aligned_weights:
                    merged_vec[idx] = self.scale * (
                        sum(aligned_vals) / sum(aligned_weights)
                    )

            merged_tensors[name] = merged_vec

        domains = sorted({d.domain for d in deltas})
        return ParameterDelta(
            contributor_id="ties_merged_master",
            domain="+".join(domains),
            tensors=merged_tensors,
            sprt_elo_gain=sum(max(0.0, d.sprt_elo_gain) for d in deltas),
            metadata={"num_contributors": str(len(deltas)), "merger": "TIES"},
        )


class DAREMerger:
    """
    DARE (Drop And REscale) + TIES Parameter Merger (Yu et al., ICML 2024).

    Randomly drops a fraction `drop_rate = p` of delta parameters using a deterministic
    cryptographic seed and rescales the remaining parameters by `1 / (1 - p)` before
    combining with TIES sign election.
    """

    def __init__(
        self,
        drop_rate: float = 0.3,
        ties_density: float = 0.7,
        seed: int = 2026,
    ) -> None:
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError("drop_rate must be in [0.0, 1.0).")
        self.drop_rate = drop_rate
        self.ties = TIESMerger(density=ties_density, scale=1.0)
        self.seed = seed

    def dare_drop_and_rescale(
        self, vec: Sequence[float], salt: str
    ) -> List[float]:
        """Apply deterministic Bernoulli dropout and `1 / (1 - p)` unbiased rescaling."""
        if self.drop_rate == 0.0:
            return list(vec)

        seed_bytes = hashlib.sha256(f"{self.seed}:{salt}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(seed_bytes[:8], "big"))
        rescale = 1.0 / (1.0 - self.drop_rate)
        return [
            (val * rescale if rng.random() >= self.drop_rate else 0.0)
            for val in vec
        ]

    def merge(self, deltas: Sequence[ParameterDelta]) -> ParameterDelta:
        """Run DARE drop-and-rescale on each contributor delta, then fuse via TIES."""
        rescaled_deltas: List[ParameterDelta] = []
        for idx, d in enumerate(deltas):
            new_tensors = {
                name: self.dare_drop_and_rescale(
                    vec, salt=f"{d.contributor_id}:{idx}:{name}"
                )
                for name, vec in d.tensors.items()
            }
            rescaled_deltas.append(
                ParameterDelta(
                    contributor_id=d.contributor_id,
                    domain=d.domain,
                    tensors=new_tensors,
                    sprt_elo_gain=d.sprt_elo_gain,
                    metadata=dict(d.metadata),
                )
            )
        merged = self.ties.merge(rescaled_deltas, elo_weighted=True)
        merged.metadata["merger"] = "DARE+TIES"
        return merged

    @staticmethod
    def domain_orthogonality_matrix(
        deltas: Sequence[ParameterDelta],
    ) -> List[List[float]]:
        """
        Compute pairwise cosine similarity across contributors' flattened parameter deltas
        to verify near-orthogonality across domains.
        """
        flattened: List[List[float]] = []
        for d in deltas:
            flat: List[float] = []
            for key in sorted(d.tensors.keys()):
                flat.extend(d.tensors[key])
            flattened.append(flat)

        k = len(flattened)
        matrix = [[0.0] * k for _ in range(k)]
        for i in range(k):
            for j in range(k):
                matrix[i][j] = _cosine_similarity(flattened[i], flattened[j])
        return matrix
