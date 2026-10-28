import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from everbench import (
    DAREMerger,
    EpochSeedProtocol,
    ParameterDelta,
    SPRTConfig,
    SPRTGate,
    SPRTStatus,
    TIESMerger,
    VERIFIABLE_DOMAINS,
    estimate_elo_difference,
)


class TestEverBench(unittest.TestCase):
    def test_cryptographic_seed_isolation_and_verification(self) -> None:
        epoch_a = EpochSeedProtocol("nist-beacon-2026-10-28-epoch-401")
        epoch_b = EpochSeedProtocol("nist-beacon-2026-10-29-epoch-402")

        for domain in VERIFIABLE_DOMAINS:
            spec_a0 = epoch_a.generate_instance_spec(domain, 0, difficulty_level=4)
            spec_a0_repeat = epoch_a.generate_instance_spec(domain, 0, difficulty_level=4)
            spec_b0 = epoch_b.generate_instance_spec(domain, 0, difficulty_level=4)

            # Deterministic within the same epoch
            self.assertEqual(spec_a0.seed_int, spec_a0_repeat.seed_int)
            self.assertEqual(spec_a0.commitment_hex, spec_a0_repeat.commitment_hex)

            # Completely uncorrelated across epochs
            self.assertNotEqual(spec_a0.seed_int, spec_b0.seed_int)
            self.assertNotEqual(spec_a0.commitment_hex, spec_b0.commitment_hex)

            # Commitment verification passes on authentic beacon and fails on tampered beacon
            self.assertTrue(
                EpochSeedProtocol.verify_instance_commitment(
                    spec_a0, "nist-beacon-2026-10-28-epoch-401"
                )
            )
            self.assertFalse(
                EpochSeedProtocol.verify_instance_commitment(
                    spec_a0, "nist-beacon-2026-10-29-epoch-402"
                )
            )

    def test_sprt_hypothesis_acceptance_and_rejection(self) -> None:
        cfg = SPRTConfig(elo0=0.0, elo1=20.0, alpha=0.05, beta=0.05)
        self.assertAlmostEqual(cfg.lower_bound, -2.944438979, places=6)
        self.assertAlmostEqual(cfg.upper_bound, 2.944438979, places=6)

        gate = SPRTGate(improvement_config=cfg)

        # Strong winning candidate -> crosses upper Wald bound A -> ACCEPT_H1
        accept_res = gate.evaluate_counts(
            domain="silicon", wins=65, losses=30, draws=80, is_target_domain=True
        )
        self.assertEqual(accept_res.status, SPRTStatus.ACCEPT_H1)
        self.assertGreater(accept_res.llr, cfg.upper_bound)
        self.assertGreater(accept_res.elo_diff, 50.0)

        # Regressing candidate -> crosses lower Wald bound B -> REJECT_H0
        reject_res = gate.evaluate_counts(
            domain="silicon", wins=20, losses=55, draws=70, is_target_domain=True
        )
        self.assertEqual(reject_res.status, SPRTStatus.REJECT_H0)
        self.assertLess(reject_res.llr, cfg.lower_bound)
        self.assertLess(reject_res.elo_diff, -50.0)

    def test_multi_domain_non_regression_gate(self) -> None:
        gate = SPRTGate()

        # PR 1: Improves `silicon` significantly and does not regress `logic`, `systems`, `matter`
        good_pr = gate.validate_multi_domain_pr(
            target_domain="silicon",
            domain_wld={
                "silicon": (65, 28, 70),
                "logic": (35, 32, 85),
                "systems": (36, 33, 80),
                "matter": (34, 31, 90),
            },
        )
        self.assertTrue(good_pr.accepted)
        self.assertEqual(len(good_pr.rejection_reasons), 0)
        self.assertGreater(good_pr.aggregate_elo_gain, 15.0)

        # PR 2: Improves `silicon`, but catastrophically regresses `logic` -> REJECTED!
        bad_pr = gate.validate_multi_domain_pr(
            target_domain="silicon",
            domain_wld={
                "silicon": (65, 28, 70),
                "logic": (15, 58, 60),  # Severe regression in logic!
                "systems": (36, 33, 80),
                "matter": (34, 31, 90),
            },
        )
        self.assertFalse(bad_pr.accepted)
        self.assertTrue(any("logic" in reason for reason in bad_pr.rejection_reasons))

    def test_sequential_early_stopping(self) -> None:
        epoch = EpochSeedProtocol("epoch-sequential-test")
        gate = SPRTGate()

        # Candidate solves when seed_int % 5 != 0 (80%), baseline solves when seed_int % 2 == 0 (50%)
        res = gate.run_sequential_match(
            domain="systems",
            epoch_protocol=epoch,
            baseline_evaluator=lambda spec: (spec.seed_int % 2 == 0),
            candidate_evaluator=lambda spec: (spec.seed_int % 5 != 0),
            max_instances=150,
            is_target_domain=True,
        )
        self.assertEqual(res.status, SPRTStatus.ACCEPT_H1)
        self.assertLess(res.total_games, 150)  # Stopped early via Wald boundary!

    def test_ties_and_dare_multi_contributor_parameter_merging(self) -> None:
        # Four contributors submit orthogonal domain-specialized parameter updates
        pr_logic = ParameterDelta(
            contributor_id="alice_lean4",
            domain="logic",
            tensors={"layers.10.lora_B": [0.9, 0.8, 0.01, -0.02, 0.0, 0.0, 0.0, 0.0]},
            sprt_elo_gain=28.0,
        )
        pr_silicon = ParameterDelta(
            contributor_id="bob_verilog",
            domain="silicon",
            tensors={"layers.10.lora_B": [0.02, -0.01, 0.85, 0.75, 0.0, 0.0, 0.0, 0.0]},
            sprt_elo_gain=34.0,
        )
        pr_systems = ParameterDelta(
            contributor_id="carol_ebpf",
            domain="systems",
            tensors={"layers.10.lora_B": [0.0, 0.0, -0.01, 0.02, -0.8, -0.9, 0.01, 0.0]},
            sprt_elo_gain=22.0,
        )
        pr_matter = ParameterDelta(
            contributor_id="dave_chem",
            domain="matter",
            tensors={"layers.10.lora_B": [0.0, 0.0, 0.0, 0.0, 0.01, -0.01, 0.7, 0.95]},
            sprt_elo_gain=31.0,
        )

        prs = [pr_logic, pr_silicon, pr_systems, pr_matter]
        ties = TIESMerger(density=0.25, scale=1.0)
        master_delta = ties.merge(prs, elo_weighted=True)
        vec = master_delta.tensors["layers.10.lora_B"]

        # Verify TIES trimmed out the small cross-domain noise and preserved all 4 domain signals!
        self.assertAlmostEqual(vec[0], 0.9, places=6)
        self.assertAlmostEqual(vec[1], 0.8, places=6)
        self.assertAlmostEqual(vec[2], 0.85, places=6)
        self.assertAlmostEqual(vec[3], 0.75, places=6)
        self.assertAlmostEqual(vec[4], -0.8, places=6)
        self.assertAlmostEqual(vec[5], -0.9, places=6)
        self.assertAlmostEqual(vec[6], 0.7, places=6)
        self.assertAlmostEqual(vec[7], 0.95, places=6)

        # Verify DARE+TIES merger and orthogonality matrix
        dare = DAREMerger(drop_rate=0.2, ties_density=0.5, seed=42)
        dare_merged = dare.merge(prs)
        self.assertEqual(len(dare_merged.tensors["layers.10.lora_B"]), 8)

        ortho = DAREMerger.domain_orthogonality_matrix(prs)
        for i in range(4):
            self.assertAlmostEqual(ortho[i][i], 1.0, places=6)
            for j in range(4):
                if i != j:
                    self.assertLess(abs(ortho[i][j]), 0.05)


if __name__ == "__main__":
    unittest.main()
