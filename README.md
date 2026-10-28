# `everbench` — "Fishtest for LLMs": Contamination-Proof Benchmark, SPRT Gate & Orthogonal Parameter Merger

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Rust Kernel](https://img.shields.io/badge/Rust-everbench__core-orange.svg)](rust/src/lib.rs)

**`everbench`** is Pillar 1 and the **Decentralized Parameter Evolution Engine ("Stockfish Fishtest for Verifiable AI")** of the Open Verifiable AI Stack.

Just as [Stockfish Fishtest](https://tests.stockfishchess.org/) enabled thousands of open-source contributors to turn Stockfish into the strongest chess engine in history by testing small parameter patches via **Wald's Sequential Probability Ratio Test (SPRT)**, **`everbench`** allows anyone in the world to submit **"Pull Requests for Model Parameters"** (LoRA adapters, GRPO policy deltas, or `steerlens` activation steering vectors) and statistically prove they improve verifiable reasoning without regressing other domains.

---

## Core Architecture: How "Pull Requests for Model Parameters" Work

```
Contributor PR (ParameterDelta: LoRA / Folded Steering Vector)
       │
       ▼
1. EpochSeedProtocol (HMAC-SHA256 Contamination-Proof Procedural Seeds)
       │   Generates fresh unseen tasks across:
       │   • logic (Lean 4)   • silicon (Verilog RTL)   • systems (eBPF)   • matter (Conservation PDEs)
       ▼
2. SPRTGate (Wald Sequential Probability Ratio Test + Non-Regression Gate)
       │   Tracks (W, L, D) head-to-head against Master Baseline:
       │   • Target domain must cross upper LLR bound A = ln((1-β)/α)  (+ΔElo proven)
       │   • Other 3 domains must pass Non-Regression bound (no catastrophic forgetting)
       ▼
3. TIESMerger / DAREMerger (Orthogonal Multi-Contributor Parameter Fusion)
       │   • Trim: Keep top-k% magnitude parameter deltas per PR
       │   • Elect Sign: Resolve sign conflicts by Elo-weighted majority consensus
       │   • Disjoint Merge: Fuse improvements across logic, silicon, systems & matter!
       ▼
New Master Checkpoint (Strictly Stronger Across All Verifiable Pillars)
```

---

## Key Modules

### 1. Contamination-Proof Procedural Seeds (`everbench.seed_protocol`)
Static benchmarks saturate and leak into pretraining corpora within months. `EpochSeedProtocol` uses `HMAC-SHA256` keyed by a public randomness beacon (`epoch_beacon`) to deterministically derive fresh problem parameters (quantifier depth, RTL pipeline stages, eBPF branch complexity, stoichiometric species counts) that cannot be memorized ahead of time, yet remain 100% reproducible and cryptographically auditable.

### 2. Sequential Probability Ratio Test Gate (`everbench.sprt`)
Implements Wald's trinomial SPRT (`SPRTGate`) comparing null hypothesis $H_0: \Delta \text{Elo} = \text{elo}_0$ against $H_1: \Delta \text{Elo} = \text{elo}_1$:
$$B = \ln\left(\frac{\beta}{1 - \alpha}\right) < \text{LLR}(W, L, D) < \ln\left(\frac{1 - \beta}{\alpha}\right) = A$$
- **Early Stopping:** Terminates evaluation as soon as $\text{LLR} \ge A$ (`ACCEPT_H1`) or $\text{LLR} \le B$ (`REJECT_H0`), saving up to 70% of verifier compute on clearly good or bad parameter patches.
- **Multi-Domain Non-Regression (`validate_multi_domain_pr`):** Guarantees that a contributor's patch improving `silicon` does not degrade `logic`, `systems`, or `matter`.

### 3. TIES & DARE Orthogonal Parameter Merging (`everbench.merger`)
When multiple contributors pass the SPRT gate in parallel across different pillars, `TIESMerger` and `DAREMerger` combine their `ParameterDelta` tensors without destructive interference:
1. **Drop & Rescale (DARE):** Randomly drops redundant small delta weights with rate $p$ and rescales survivors by $\frac{1}{1-p}$.
2. **Trim (TIES):** Retains only the top-$k\%$ highest-magnitude deltas per contributor.
3. **Elect Sign & Disjoint Merge:** Resolves sign conflicts by Elo-weighted consensus mass and averages aligned non-zero updates.

---

## Quickstart

```python
from everbench import (
    EpochSeedProtocol,
    SPRTGate,
    ParameterDelta,
    TIESMerger,
)

# 1. Derive contamination-proof procedural benchmark tasks for an evaluation epoch
epoch = EpochSeedProtocol("nist-beacon-2026-10-28-epoch-401")
silicon_suite = epoch.generate_domain_suite("silicon", count=32, difficulty_level=4)

# 2. Validate a contributor's Parameter PR across all 4 verifiable domains via SPRT
gate = SPRTGate()
report = gate.validate_multi_domain_pr(
    target_domain="silicon",
    domain_wld={
        "silicon": (65, 28, 70),  # +Elo breakthrough in Verilog synthesis!
        "logic":   (35, 32, 85),  # Non-regressing in Lean 4
        "systems": (36, 33, 80),  # Non-regressing in eBPF
        "matter":  (34, 31, 90),  # Non-regressing in Chemistry/Physics
    },
)
assert report.accepted

# 3. Merge accepted Parameter PRs from multiple contributors via TIES-Merging
merger = TIESMerger(density=0.25)
master_delta = merger.merge([pr_logic, pr_silicon, pr_systems, pr_matter])
```

---

## Running Tests

```bash
python3 -m unittest discover -s tests -v
```
