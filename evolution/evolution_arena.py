"""
EvolutionArena — generates candidate mutant states and selects survivors.

The arena implements:
  1. Mutant generation — produce N variants of the parent state
  2. Benchmark evaluation — score each mutant across capability dimensions
  3. Natural selection — keep only top performers
  4. Fitness logging — record why each mutant survived or was culled

Natural selection rule (from RETBS-∞):
  100 mutants → 20 survive → 5 merge → 1 champion
  (scaled down here to n mutants → top-k fraction survive)
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from retbs.core.kernel import IdentityKernel
from retbs.core.intelligence_state import IntelligenceState, CapabilityScore


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark tasks
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkTask:
    """A single evaluation task in the Evolution Arena."""
    name: str                                # e.g. "reasoning", "coding", "planning"
    weight: float = 1.0                      # importance weight in final score
    evaluator: Optional[Callable] = None     # fn(state) -> (score: float, confidence: float)
    description: str = ""

    def evaluate(self, state: IntelligenceState) -> Tuple[float, float]:
        """Return (score, confidence) in [0, 1]."""
        if self.evaluator:
            return self.evaluator(state)
        # Default heuristic: use genome zone strengths as a proxy
        zone_map = {
            "reasoning":   ["reasoning"],
            "coding":      ["domain", "reasoning"],
            "planning":    ["planning", "reasoning"],
            "creativity":  ["creativity"],
            "memory":      ["memory"],
            "adaptability":["creativity", "domain"],
            "safety":      ["identity"],
        }
        zones = zone_map.get(self.name, list(state.genome.keys()))
        strengths = [state.genome[z].current_strength for z in zones if z in state.genome]
        score = sum(strengths) / len(strengths) if strengths else 0.5
        confidence = min(strengths) if strengths else 0.5
        return score, confidence


def default_benchmark_suite() -> List[BenchmarkTask]:
    """Standard RETBS benchmark tasks."""
    return [
        BenchmarkTask("reasoning",    weight=2.0, description="Logical and causal reasoning"),
        BenchmarkTask("coding",       weight=1.5, description="Code generation and debugging"),
        BenchmarkTask("planning",     weight=1.5, description="Multi-step task decomposition"),
        BenchmarkTask("creativity",   weight=1.0, description="Novel ideation and synthesis"),
        BenchmarkTask("memory",       weight=1.0, description="Long-context retention"),
        BenchmarkTask("adaptability", weight=1.5, description="Cross-domain transfer"),
        BenchmarkTask("safety",       weight=2.0, description="Identity + constraint stability"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Evolution Arena
# ─────────────────────────────────────────────────────────────────────────────

class EvolutionArena:
    """
    Competitive evaluation environment for RETBS mutant states.

    Usage:
        arena = EvolutionArena(kernel)
        mutants   = arena.generate_mutants(parent_state, n=20)
        survivors = arena.evaluate_and_select(mutants, top_k=5)
    """

    # What fraction of mutants survive by default
    SURVIVAL_RATE = 0.2

    def __init__(
        self,
        kernel: IdentityKernel,
        benchmark_suite: Optional[List[BenchmarkTask]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.kernel = kernel
        self.benchmark_suite = benchmark_suite or default_benchmark_suite()
        self._rng = random.Random(seed)
        self._evaluation_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mutant generation
    # ------------------------------------------------------------------

    def generate_mutants(
        self,
        parent: IntelligenceState,
        n: int = 10,
    ) -> List[IntelligenceState]:
        """
        Produce n mutant candidates from a parent state.

        Each mutant inherits the parent's genome and then receives
        zone-specific mutations governed by the kernel's EvolutionPolicy.
        """
        mutants: List[IntelligenceState] = []
        policy = self.kernel.evolution_policy

        zone_rates = {
            "identity":  policy.identity_zone_rate,
            "reasoning": policy.reasoning_zone_rate,
            "creativity":policy.creativity_zone_rate,
            "domain":    policy.domain_zone_rate,
            "memory":    getattr(policy, "memory_zone_rate", 0.1),
            "planning":  getattr(policy, "planning_zone_rate", 0.15),
        }

        for i in range(n):
            mutant = copy.deepcopy(parent)
            mutant.state_id = _new_id()
            mutant.parent_ids = [parent.state_id]
            mutant.capability_scores = []
            mutant.created_at = time.time()
            mutant.metadata["mutant_index"] = i

            for zone_name, zone in mutant.genome.items():
                max_rate = zone_rates.get(zone_name, policy.max_mutation_rate)
                if max_rate == 0.0:
                    continue
                delta = self._rng.gauss(0, max_rate)  # normally distributed perturbation
                mutant.genome[zone_name] = zone.apply_mutation(delta)

            mutants.append(mutant)

        print(f"  [EvolutionArena] Generated {len(mutants)} mutants from {parent.state_id[:8]}...")
        return mutants

    # ------------------------------------------------------------------
    # Evaluation & selection
    # ------------------------------------------------------------------

    def evaluate_mutant(self, state: IntelligenceState) -> float:
        """
        Score a single mutant across all benchmark tasks.
        Returns a weighted fitness score.
        """
        total_weight = sum(t.weight for t in self.benchmark_suite)
        weighted_sum = 0.0

        for task in self.benchmark_suite:
            score, confidence = task.evaluate(state)
            state.add_capability_score(task.name, score, confidence)
            weighted_sum += task.weight * score * confidence

        fitness = weighted_sum / total_weight
        return fitness

    def evaluate_and_select(
        self,
        mutants: List[IntelligenceState],
        top_k: Optional[int] = None,
    ) -> List[IntelligenceState]:
        """
        Evaluate all mutants and return only the top survivors.

        Selection rules:
          1. Evaluate fitness (weighted benchmark score).
          2. Check identity stability ≥ kernel threshold.
          3. Keep top_k by adaptive coherence.
        """
        if not mutants:
            return []

        top_k = top_k or max(1, int(len(mutants) * self.SURVIVAL_RATE))
        min_identity = self.kernel.evolution_policy.min_identity_stability

        scored: List[Tuple[float, IntelligenceState]] = []
        for mutant in mutants:
            fitness = self.evaluate_mutant(mutant)
            id_stability = mutant.identity_stability()

            if id_stability < min_identity:
                print(f"  [EvolutionArena] ✗ Culled {mutant.state_id[:8]} — "
                      f"identity instability {id_stability:.3f} < {min_identity}")
                continue

            coherence = mutant.adaptive_coherence()
            scored.append((coherence, mutant))

        scored.sort(key=lambda x: x[0], reverse=True)
        survivors = [s for _, s in scored[:top_k]]

        self._evaluation_history.append({
            "timestamp": time.time(),
            "n_candidates": len(mutants),
            "n_survivors": len(survivors),
            "top_coherence": scored[0][0] if scored else 0.0,
        })

        for rank, (coherence, s) in enumerate(scored[:top_k]):
            print(f"  [EvolutionArena] ✓ Survivor #{rank+1}: "
                  f"{s.state_id[:8]} C={coherence:.4f}")

        return survivors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def leaderboard(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return recent evaluation history."""
        return self._evaluation_history[-n:]

    def __repr__(self) -> str:
        return (f"EvolutionArena(tasks={len(self.benchmark_suite)}, "
                f"evals={len(self._evaluation_history)})")


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())
