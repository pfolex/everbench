"""
Stockfish Fishtest-style Sequential Probability Ratio Test (`SPRTGate`) and
Non-Regression Multi-Domain Validator.

Evaluates a contributor's candidate parameter update (`ParameterDelta`) against the
current master baseline model on procedural verifiable benchmarks:
- Tracks paired match outcomes: Wins (`W`), Losses (`L`), Draws (`D`)
- Computes sequential Log-Likelihood Ratio (`LLR`) against Wald bounds
  `[B, A] = [ln(beta / (1 - alpha)), ln((1 - beta) / alpha)]`
- Estimates Elo gain (`Delta Elo`) with 95% confidence interval
- Enforces cross-domain non-regression across all 4 verifiable pillars
  (`logic`, `silicon`, `systems`, `matter`) before accepting a parameter PR!
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .seed_protocol import BenchmarkInstanceSpec, EpochSeedProtocol, VERIFIABLE_DOMAINS


class SPRTStatus(str, Enum):
    """Decision state of a Sequential Probability Ratio Test."""

    ACCEPT_H1 = "ACCEPT_H1"
    REJECT_H0 = "REJECT_H0"
    CONTINUE = "CONTINUE"


@dataclass(frozen=True)
class SPRTConfig:
    """
    Configuration for Wald's Sequential Probability Ratio Test (SPRT).

    - `elo0`: Null-hypothesis Elo difference (e.g., `0.0` for non-regression or `0.5` for improvement)
    - `elo1`: Alternative-hypothesis target Elo gain (e.g., `15.0` Elo)
    - `alpha`: False positive rate (Type-I error probability, default `0.05`)
    - `beta`: False negative rate (Type-II error probability, default `0.05`)
    """

    elo0: float = 0.0
    elo1: float = 15.0
    alpha: float = 0.05
    beta: float = 0.05

    @property
    def lower_bound(self) -> float:
        """Wald lower stopping threshold `B = ln(beta / (1 - alpha))`."""
        return math.log(self.beta / (1.0 - self.alpha))

    @property
    def upper_bound(self) -> float:
        """Wald upper stopping threshold `A = ln((1 - beta) / alpha)`."""
        return math.log((1.0 - self.beta) / self.alpha)


@dataclass(frozen=True)
class SPRTResult:
    """Summary of an SPRT evaluation run on a benchmark domain."""

    domain: str
    wins: int
    losses: int
    draws: int
    llr: float
    lower_bound: float
    upper_bound: float
    elo_diff: float
    elo_ci_95: float
    status: SPRTStatus

    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws


@dataclass(frozen=True)
class MultiDomainGateReport:
    """
    Complete multi-domain Fishtest gate report across `logic`, `silicon`, `systems`, and `matter`.
    """

    target_domain: str
    domain_results: Dict[str, SPRTResult]
    aggregate_elo_gain: float
    accepted: bool
    rejection_reasons: List[str]


def _elo_to_expected_score(elo_diff: float) -> float:
    """Logistic expected score `s in (0, 1)` for a given Elo difference."""
    return 1.0 / (1.0 + math.pow(10.0, -elo_diff / 400.0))


def estimate_elo_difference(wins: int, losses: int, draws: int) -> Tuple[float, float]:
    """
    Estimate empirical Elo gain `Delta Elo` and its 95% confidence half-width (`+- ci_95`)
    from paired match counts `(wins, losses, draws)`.
    """
    n = wins + losses + draws
    if n == 0:
        return 0.0, 0.0

    score = (wins + 0.5 * draws) / n
    clamped = min(max(score, 1e-4), 1.0 - 1e-4)
    elo = -400.0 * math.log10(1.0 / clamped - 1.0)

    # Sample variance of game score X in {0, 0.5, 1}
    p_w = wins / n
    p_d = draws / n
    p_l = losses / n
    var_s = (
        p_w * (1.0 - score) ** 2
        + p_d * (0.5 - score) ** 2
        + p_l * (0.0 - score) ** 2
    )
    se_s = math.sqrt(max(var_s, 1e-9) / n)

    # Delta method derivative: d(Elo)/ds = 400 / (ln(10) * s * (1 - s))
    d_elo_ds = 400.0 / (math.log(10.0) * clamped * (1.0 - clamped))
    ci_95 = 1.96 * d_elo_ds * se_s
    return elo, ci_95


class SPRTGate:
    """
    Stockfish Fishtest-style Sequential Probability Ratio Test (SPRT) Gate.

    Computes trinomial Log-Likelihood Ratios (`LLR`) comparing hypothesis `H_1`
    (`Delta Elo = elo1`) against `H_0` (`Delta Elo = elo0`) as paired head-to-head
    verifier evaluations arrive.
    """

    def __init__(
        self,
        improvement_config: Optional[SPRTConfig] = None,
        non_regression_config: Optional[SPRTConfig] = None,
    ) -> None:
        self.improvement_config = improvement_config or SPRTConfig(
            elo0=0.0, elo1=20.0, alpha=0.05, beta=0.05
        )
        # Non-regression gate tests that a domain does not regress by more than -5 Elo
        # vs gaining/maintaining +5 Elo
        self.non_regression_config = non_regression_config or SPRTConfig(
            elo0=-10.0, elo1=5.0, alpha=0.05, beta=0.05
        )

    @staticmethod
    def compute_llr(
        wins: int,
        losses: int,
        draws: int,
        config: SPRTConfig,
    ) -> float:
        """
        Compute the trinomial Log-Likelihood Ratio (LLR) for `(wins, losses, draws)`
        under `H_1(elo1)` vs. `H_0(elo0)`.
        """
        n = wins + losses + draws
        if n == 0:
            return 0.0

        # Empirical draw ratio regularized slightly away from 0 and 1
        draw_rate = min(max(draws / n, 0.05), 0.90)
        s0 = _elo_to_expected_score(config.elo0)
        s1 = _elo_to_expected_score(config.elo1)

        # Trinomial probabilities (W, D, L) matching expected score s_i and draw_rate
        def _trinomial_probs(s: float, d_prob: float) -> Tuple[float, float, float]:
            w_prob = max(1e-6, min(1.0 - d_prob - 1e-6, s - 0.5 * d_prob))
            l_prob = max(1e-6, 1.0 - d_prob - w_prob)
            total = w_prob + d_prob + l_prob
            return w_prob / total, d_prob / total, l_prob / total

        w0, d0, l0 = _trinomial_probs(s0, draw_rate)
        w1, d1, l1 = _trinomial_probs(s1, draw_rate)

        llr = (
            wins * math.log(w1 / w0)
            + draws * math.log(d1 / d0)
            + losses * math.log(l1 / l0)
        )
        return llr

    def evaluate_counts(
        self,
        domain: str,
        wins: int,
        losses: int,
        draws: int,
        is_target_domain: bool = True,
    ) -> SPRTResult:
        """
        Evaluate W/L/D counts against the SPRT stopping boundaries.
        """
        cfg = self.improvement_config if is_target_domain else self.non_regression_config
        llr = self.compute_llr(wins, losses, draws, cfg)
        lower_b = cfg.lower_bound
        upper_a = cfg.upper_bound
        elo_diff, elo_ci = estimate_elo_difference(wins, losses, draws)

        if llr >= upper_a:
            status = SPRTStatus.ACCEPT_H1
        elif llr <= lower_b:
            status = SPRTStatus.REJECT_H0
        else:
            status = SPRTStatus.CONTINUE

        return SPRTResult(
            domain=domain,
            wins=wins,
            losses=losses,
            draws=draws,
            llr=llr,
            lower_bound=lower_b,
            upper_bound=upper_a,
            elo_diff=elo_diff,
            elo_ci_95=elo_ci,
            status=status,
        )

    def run_sequential_match(
        self,
        domain: str,
        epoch_protocol: EpochSeedProtocol,
        baseline_evaluator: Callable[[BenchmarkInstanceSpec], bool],
        candidate_evaluator: Callable[[BenchmarkInstanceSpec], bool],
        max_instances: int = 200,
        is_target_domain: bool = True,
    ) -> SPRTResult:
        """
        Run paired sequential evaluations on procedural `BenchmarkInstanceSpec` tasks,
        stopping early as soon as the Wald SPRT boundary `A` or `B` is crossed.
        """
        wins = 0
        losses = 0
        draws = 0
        cfg = self.improvement_config if is_target_domain else self.non_regression_config

        for idx in range(max_instances):
            spec = epoch_protocol.generate_instance_spec(domain, idx)
            base_solved = bool(baseline_evaluator(spec))
            cand_solved = bool(candidate_evaluator(spec))

            if cand_solved and not base_solved:
                wins += 1
            elif base_solved and not cand_solved:
                losses += 1
            else:
                draws += 1

            llr = self.compute_llr(wins, losses, draws, cfg)
            if llr >= cfg.upper_bound or llr <= cfg.lower_bound:
                break

        return self.evaluate_counts(
            domain=domain,
            wins=wins,
            losses=losses,
            draws=draws,
            is_target_domain=is_target_domain,
        )

    def validate_multi_domain_pr(
        self,
        target_domain: str,
        domain_wld: Mapping[str, Tuple[int, int, int]],
    ) -> MultiDomainGateReport:
        """
        Validate a contributor's Parameter PR across all 4 verifiable pillars
        (`logic`, `silicon`, `systems`, `matter`).

        A PR is accepted if and only if:
        1. `target_domain` achieves `SPRTStatus.ACCEPT_H1` (statistically significant Elo gain)
           or positive Elo with LLR > 0 when sample budget completes.
        2. None of the other 3 domains fail the non-regression gate (`REJECT_H0`) or exhibit
           negative Elo below the non-regression floor.
        """
        if target_domain not in VERIFIABLE_DOMAINS:
            raise ValueError(f"target_domain must be one of {VERIFIABLE_DOMAINS}")

        results: Dict[str, SPRTResult] = {}
        rejection_reasons: List[str] = []

        for dom in VERIFIABLE_DOMAINS:
            if dom not in domain_wld:
                rejection_reasons.append(f"Missing evaluation counts for domain '{dom}'.")
                continue
            w, l, d = domain_wld[dom]
            is_target = dom == target_domain
            res = self.evaluate_counts(dom, w, l, d, is_target_domain=is_target)
            results[dom] = res

            if is_target:
                if res.status != SPRTStatus.ACCEPT_H1:
                    rejection_reasons.append(
                        f"Target domain '{dom}' did not pass SPRT H1 gate "
                        f"(status={res.status.value}, LLR={res.llr:.2f}, Elo={res.elo_diff:+.1f})."
                    )
            else:
                if res.status == SPRTStatus.REJECT_H0 or res.elo_diff < -5.0:
                    rejection_reasons.append(
                        f"Non-target domain '{dom}' regressed "
                        f"(status={res.status.value}, LLR={res.llr:.2f}, Elo={res.elo_diff:+.1f})."
                    )

        avg_elo = (
            sum(r.elo_diff for r in results.values()) / len(results)
            if results
            else 0.0
        )
        return MultiDomainGateReport(
            target_domain=target_domain,
            domain_results=results,
            aggregate_elo_gain=avg_elo,
            accepted=(len(rejection_reasons) == 0 and len(results) == len(VERIFIABLE_DOMAINS)),
            rejection_reasons=rejection_reasons,
        )
