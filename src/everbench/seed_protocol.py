"""
Contamination-Proof Procedural Evaluation Seed Derivation (`EpochSeedProtocol`).

Uses HMAC-SHA256 keyed by an unpredictable beacon/epoch secret and domain identifier
to deterministically derive fresh, verifiable problem seeds across the four pillars:
- `logic`   (Lean 4 theorems / SMT synthesis)
- `silicon` (Verilog RTL equivalence & timing synthesis)
- `systems` (eBPF verifier-safe kernel policies & concurrency schedules)
- `matter`  (Conservation-law PDE surrogates & stoichiometric reaction networks)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Dict, List, Sequence, Tuple


VERIFIABLE_DOMAINS: Tuple[str, ...] = ("logic", "silicon", "systems", "matter")


@dataclass(frozen=True)
class BenchmarkInstanceSpec:
    """
    Cryptographically committed procedural benchmark instance specification.
    """

    epoch_id: str
    domain: str
    instance_index: int
    seed_int: int
    commitment_hex: str
    difficulty_level: int
    structural_parameters: Dict[str, int]


class EpochSeedProtocol:
    """
    Contamination-proof `HMAC-SHA256` procedural seed generator.

    Guarantees that:
    1. Anyone with the public `epoch_beacon` can audit and reproduce the exact test suite.
    2. Static training corpora prior to the epoch beacon cannot memorize the generated
       procedural problem parameters, variable permutations, or invariant constants.
    """

    def __init__(self, epoch_beacon: str, protocol_version: str = "everbench-v1") -> None:
        if not epoch_beacon or not epoch_beacon.strip():
            raise ValueError("epoch_beacon must be a non-empty string.")
        self.epoch_beacon = epoch_beacon.strip()
        self.protocol_version = protocol_version
        self._master_key = hmac.new(
            self.protocol_version.encode("utf-8"),
            self.epoch_beacon.encode("utf-8"),
            hashlib.sha256,
        ).digest()

    def derive_seed(self, domain: str, instance_index: int) -> Tuple[int, str]:
        """
        Derive a 64-bit deterministic integer seed and 256-bit hex commitment
        for `(domain, instance_index)`.
        """
        if domain not in VERIFIABLE_DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}'. Expected one of {VERIFIABLE_DOMAINS}."
            )
        if instance_index < 0:
            raise ValueError("instance_index must be non-negative.")

        message = f"{self.epoch_beacon}:{domain}:{instance_index}".encode("utf-8")
        digest = hmac.new(self._master_key, message, hashlib.sha256).digest()
        seed_int = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return seed_int, digest.hex()

    def generate_instance_spec(
        self,
        domain: str,
        instance_index: int,
        difficulty_level: int = 3,
    ) -> BenchmarkInstanceSpec:
        """
        Generate a domain-specific procedural problem specification from the HMAC seed.
        """
        if not (1 <= difficulty_level <= 10):
            raise ValueError("difficulty_level must be in [1, 10].")

        seed_int, commitment = self.derive_seed(domain, instance_index)
        # Derive domain-specific procedural structural parameters deterministically
        b0 = (seed_int >> 0) & 0xFFFF
        b1 = (seed_int >> 16) & 0xFFFF
        b2 = (seed_int >> 32) & 0xFFFF

        if domain == "logic":
            params = {
                "quantifier_depth": 1 + (b0 % difficulty_level),
                "lemma_chain_len": 2 + (b1 % (2 * difficulty_level)),
                "variable_permutation_id": b2,
            }
        elif domain == "silicon":
            params = {
                "bit_width": 8 * (1 + (b0 % 4)),
                "pipeline_stages": 1 + (b1 % difficulty_level),
                "fsm_states": 4 + (b2 % (2 * difficulty_level)),
            }
        elif domain == "systems":
            params = {
                "ebpf_max_branches": 4 + (b0 % (3 * difficulty_level)),
                "ringbuf_slots": 16 * (1 + (b1 % 8)),
                "packet_offset_mask": b2 & 0xFF,
            }
        else:  # matter
            params = {
                "species_count": 3 + (b0 % (2 * difficulty_level)),
                "conservation_laws": 2 + (b1 % 3),
                "grid_resolution": 16 + 8 * (b2 % 4),
            }

        return BenchmarkInstanceSpec(
            epoch_id=self.epoch_beacon,
            domain=domain,
            instance_index=instance_index,
            seed_int=seed_int,
            commitment_hex=commitment,
            difficulty_level=difficulty_level,
            structural_parameters=params,
        )

    def generate_domain_suite(
        self,
        domain: str,
        count: int = 32,
        difficulty_level: int = 3,
    ) -> List[BenchmarkInstanceSpec]:
        """Generate a batch of contamination-proof procedural benchmark specs."""
        return [
            self.generate_instance_spec(domain, i, difficulty_level=difficulty_level)
            for i in range(count)
        ]

    @staticmethod
    def verify_instance_commitment(
        spec: BenchmarkInstanceSpec,
        epoch_beacon: str,
        protocol_version: str = "everbench-v1",
    ) -> bool:
        """
        Cryptographically verify that a `BenchmarkInstanceSpec` was genuinely derived
        from `epoch_beacon` without cherry-picking.
        """
        proto = EpochSeedProtocol(epoch_beacon=epoch_beacon, protocol_version=protocol_version)
        expected_seed, expected_hex = proto.derive_seed(spec.domain, spec.instance_index)
        return spec.seed_int == expected_seed and hmac.compare_digest(
            spec.commitment_hex, expected_hex
        )
