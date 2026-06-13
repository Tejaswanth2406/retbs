"""
BlendEngine — merge survivor states into a single hybrid intelligence.

Blending methods (from RETBS framework):
  weighted_average   — interpolate genome strengths by fitness rank
  task_vector        — additive capability transfer (like arithmetic on LoRA weights)
  best_of_zone       — for each zone, take the strongest value across survivors
  adaptive           — choose method per-zone based on zone type

In production fine-tuning these map to:
  - Weight interpolation (SLERP / LERP on model parameters)
  - Adapter fusion (merge multiple LoRA adapters)
  - Task arithmetic (add task vectors to a base model)
  - Model merging frameworks (e.g. MergeKit)

Blending respects the IdentityKernel: the identity zone is never blended
away — it is always inherited from the lineage root.
"""

from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from retbs.core.kernel import IdentityKernel
from retbs.core.intelligence_state import IntelligenceState, GenomeZone, CapabilityScore


class BlendMethod(str, Enum):
    WEIGHTED_AVERAGE = "weighted_average"
    TASK_VECTOR      = "task_vector"
    BEST_OF_ZONE     = "best_of_zone"
    ADAPTIVE         = "adaptive"


class BlendEngine:
    """
    Merges a list of survivor IntelligenceStates into a single hybrid state.

    Usage:
        engine = BlendEngine(kernel)
        hybrid = engine.blend(survivors)          # uses default adaptive method
        hybrid = engine.blend(survivors, method=BlendMethod.BEST_OF_ZONE)
    """

    # Zones that benefit from best-of selection (high-variance, competitive)
    _COMPETITIVE_ZONES = {"creativity", "domain", "planning"}
    # Zones that benefit from averaging (stable, consensus-driven)
    _CONSENSUS_ZONES   = {"reasoning", "memory"}

    def __init__(
        self,
        kernel: IdentityKernel,
        default_method: BlendMethod = BlendMethod.ADAPTIVE,
    ) -> None:
        self.kernel = kernel
        self.default_method = default_method
        self._blend_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def blend(
        self,
        states: List[IntelligenceState],
        method: Optional[BlendMethod] = None,
        weights: Optional[List[float]] = None,
    ) -> IntelligenceState:
        """
        Merge a list of states into one hybrid state.

        Args:
            states:  Survivor states from the Evolution Arena.
            method:  Blending strategy. Defaults to self.default_method.
            weights: Optional fitness weights (must match len(states)).
                     If None, weights are derived from adaptive_coherence().

        Returns:
            A new IntelligenceState representing the blended generation.
        """
        if not states:
            raise ValueError("[BlendEngine] Cannot blend an empty list of states.")

        if len(states) == 1:
            print("  [BlendEngine] Single survivor — no blending needed.")
            return copy.deepcopy(states[0])

        method = method or self.default_method
        weights = weights or self._coherence_weights(states)
        weights = _normalise(weights)

        print(f"  [BlendEngine] Blending {len(states)} states via '{method}'")

        if method == BlendMethod.WEIGHTED_AVERAGE:
            hybrid_genome = self._weighted_average(states, weights)
        elif method == BlendMethod.TASK_VECTOR:
            hybrid_genome = self._task_vector(states, weights)
        elif method == BlendMethod.BEST_OF_ZONE:
            hybrid_genome = self._best_of_zone(states)
        else:  # ADAPTIVE
            hybrid_genome = self._adaptive(states, weights)

        # Always restore the immutable identity zone from the first (highest-ranked) state
        if "identity" in states[0].genome:
            hybrid_genome["identity"] = copy.deepcopy(states[0].genome["identity"])

        # Build the blended state
        hybrid = IntelligenceState(
            state_id=str(uuid.uuid4()),
            generation=max(s.generation for s in states),
            parent_ids=[s.state_id for s in states],
            genome=hybrid_genome,
            metadata={
                "blend_method": method,
                "n_parents": len(states),
                "blend_weights": weights,
            },
        )

        # Inherit best capability scores
        hybrid.capability_scores = self._merge_capability_scores(states, weights)

        self._blend_history.append({
            "timestamp": time.time(),
            "method": method,
            "n_parents": len(states),
            "result_id": hybrid.state_id[:8],
            "coherence": hybrid.adaptive_coherence(),
        })

        print(f"  [BlendEngine] Hybrid C={hybrid.adaptive_coherence():.4f} "
              f"id={hybrid.state_id[:8]}...")
        return hybrid

    # ------------------------------------------------------------------
    # Blending strategies
    # ------------------------------------------------------------------

    def _weighted_average(
        self,
        states: List[IntelligenceState],
        weights: List[float],
    ) -> Dict[str, GenomeZone]:
        """Interpolate each zone's strength as a weighted average."""
        all_zones = set()
        for s in states:
            all_zones.update(s.genome.keys())

        genome: Dict[str, GenomeZone] = {}
        for zone_name in all_zones:
            strengths    = [s.genome[zone_name].current_strength if zone_name in s.genome else 0.0 for s in states]
            stabilities  = [s.genome[zone_name].stability        if zone_name in s.genome else 0.5 for s in states]
            mut_rate     = states[0].genome[zone_name].mutation_rate if zone_name in states[0].genome else 0.1

            blended_strength  = sum(w * v for w, v in zip(weights, strengths))
            blended_stability = sum(w * v for w, v in zip(weights, stabilities))

            genome[zone_name] = GenomeZone(
                name=zone_name,
                mutation_rate=mut_rate,
                current_strength=min(1.0, blended_strength),
                stability=min(1.0, blended_stability),
            )
        return genome

    def _task_vector(
        self,
        states: List[IntelligenceState],
        weights: List[float],
    ) -> Dict[str, GenomeZone]:
        """
        Task-vector blending: start from the highest-ranked state, then
        additively inject capability deltas from all others.

        Analogous to: base + Σ(weight_i * task_vector_i) in parameter space.
        """
        base = copy.deepcopy(states[0].genome)
        for i, state in enumerate(states[1:], start=1):
            w = weights[i]
            for zone_name, zone in state.genome.items():
                if zone_name not in base or zone_name == "identity":
                    continue
                base_zone = base[zone_name]
                delta = (zone.current_strength - base_zone.current_strength) * w
                base[zone_name] = base_zone.apply_mutation(delta)
        return base

    def _best_of_zone(
        self,
        states: List[IntelligenceState],
    ) -> Dict[str, GenomeZone]:
        """For every zone, pick the state with the highest zone strength."""
        all_zones = set()
        for s in states:
            all_zones.update(s.genome.keys())

        genome: Dict[str, GenomeZone] = {}
        for zone_name in all_zones:
            candidates = [(s.genome[zone_name].current_strength, s.genome[zone_name])
                          for s in states if zone_name in s.genome]
            if candidates:
                _, best_zone = max(candidates, key=lambda x: x[0])
                genome[zone_name] = copy.deepcopy(best_zone)
        return genome

    def _adaptive(
        self,
        states: List[IntelligenceState],
        weights: List[float],
    ) -> Dict[str, GenomeZone]:
        """
        Per-zone adaptive blending:
          - Competitive zones  → best_of_zone
          - Consensus zones    → weighted_average
          - Everything else    → task_vector
        """
        all_zones = set()
        for s in states:
            all_zones.update(s.genome.keys())

        genome: Dict[str, GenomeZone] = {}
        for zone_name in all_zones:
            if zone_name in self._COMPETITIVE_ZONES:
                candidates = [(s.genome[zone_name].current_strength, s.genome[zone_name])
                              for s in states if zone_name in s.genome]
                if candidates:
                    _, best = max(candidates, key=lambda x: x[0])
                    genome[zone_name] = copy.deepcopy(best)

            elif zone_name in self._CONSENSUS_ZONES:
                strengths   = [s.genome[zone_name].current_strength if zone_name in s.genome else 0.0 for s in states]
                stabilities = [s.genome[zone_name].stability        if zone_name in s.genome else 0.5 for s in states]
                mut_rate    = states[0].genome[zone_name].mutation_rate if zone_name in states[0].genome else 0.1
                genome[zone_name] = GenomeZone(
                    name=zone_name,
                    mutation_rate=mut_rate,
                    current_strength=min(1.0, sum(w * v for w, v in zip(weights, strengths))),
                    stability=min(1.0, sum(w * v for w, v in zip(weights, stabilities))),
                )
            else:
                # task_vector for remaining zones
                base_zone = states[0].genome.get(zone_name)
                if base_zone is None:
                    continue
                zone = copy.deepcopy(base_zone)
                for i, state in enumerate(states[1:], start=1):
                    if zone_name not in state.genome:
                        continue
                    delta = (state.genome[zone_name].current_strength - base_zone.current_strength) * weights[i]
                    zone = zone.apply_mutation(delta)
                genome[zone_name] = zone

        return genome

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coherence_weights(states: List[IntelligenceState]) -> List[float]:
        scores = [max(s.adaptive_coherence(), 0.001) for s in states]
        return scores

    @staticmethod
    def _merge_capability_scores(
        states: List[IntelligenceState],
        weights: List[float],
    ) -> List[CapabilityScore]:
        """Build merged capability scores: weighted average per task name."""
        from collections import defaultdict
        buckets: Dict[str, List] = defaultdict(list)
        for state, w in zip(states, weights):
            for cs in state.capability_scores:
                buckets[cs.name].append((cs.score * w, cs.confidence * w, w))

        merged = []
        for name, items in buckets.items():
            total_w = sum(i[2] for i in items)
            if total_w == 0:
                continue
            score = sum(i[0] for i in items) / total_w
            conf  = sum(i[1] for i in items) / total_w
            merged.append(CapabilityScore(name=name, score=score, confidence=conf))
        return merged

    def history(self) -> List[Dict]:
        return list(self._blend_history)

    def __repr__(self) -> str:
        return f"BlendEngine(method={self.default_method}, blends={len(self._blend_history)})"


def _normalise(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]
