"""
SustainEngine — stabilises a blended state and guards against identity drift.

Without sustainment the RETBS cycle collapses:
  Mutation → Instability → Collapse

Sustain creates:
  1. Identity stability check   — reject states that drift from the kernel's mission
  2. Knowledge retention check  — detect catastrophic forgetting
  3. Capability preservation    — ensure no critical benchmark regresses
  4. Evolution governor         — if checks fail, roll back to the previous state
  5. Stabilisation pass         — gently re-anchor volatile genome zones

Metrics produced:
  identity_stability_score    — how close the identity zone is to its baseline
  knowledge_retention_score   — mean zone stability vs previous generation
  capability_preservation_score — whether top capabilities were preserved
  drift_score                 — overall deviation (lower is better)

Rollback policy:
  If identity_stability < kernel.min_identity_stability → hard rollback
  If drift_score > MAX_DRIFT_THRESHOLD               → soft rollback (blend with parent)
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from retbs.core.kernel import IdentityKernel
from retbs.core.intelligence_state import IntelligenceState, GenomeZone


MAX_DRIFT_THRESHOLD = 0.4   # soft rollback if drift exceeds this


@dataclass
class SustainReport:
    """Full sustainability report for one generation transition."""
    generation: int
    identity_stability: float
    knowledge_retention: float
    capability_preservation: float
    drift_score: float
    action: str            # "accepted" | "soft_rollback" | "hard_rollback"
    notes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return self.action == "accepted"

    def summary(self) -> str:
        lines = [
            f"SustainReport(gen={self.generation}, action={self.action})",
            f"  identity_stability   : {self.identity_stability:.4f}",
            f"  knowledge_retention  : {self.knowledge_retention:.4f}",
            f"  capability_preserve  : {self.capability_preservation:.4f}",
            f"  drift_score          : {self.drift_score:.4f}",
        ]
        for note in self.notes:
            lines.append(f"  ⚠ {note}")
        return "\n".join(lines)


class SustainEngine:
    """
    Guards the RETBS lineage against instability and identity collapse.

    Usage:
        sustain = SustainEngine(kernel)
        stable_state = sustain.stabilise(blended_state, previous_state, generation=3)
    """

    def __init__(
        self,
        kernel: IdentityKernel,
        stabilisation_strength: float = 0.1,
    ) -> None:
        self.kernel = kernel
        self.stabilisation_strength = stabilisation_strength
        self.reports: List[SustainReport] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def stabilise(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
        generation: int,
    ) -> IntelligenceState:
        """
        Validate and stabilise a candidate state before it becomes the next generation.

        Steps:
          1. Measure stability metrics
          2. Decide: accept / soft-rollback / hard-rollback
          3. Apply stabilisation pass if accepted
          4. Record report

        Returns the final (possibly rolled-back / stabilised) state.
        """
        notes: List[str] = []
        min_identity = self.kernel.evolution_policy.min_identity_stability

        # ── 1. Measure ───────────────────────────────────────────────
        identity_stability      = self._identity_stability(candidate, previous)
        knowledge_retention     = self._knowledge_retention(candidate, previous)
        capability_preservation = self._capability_preservation(candidate, previous)
        drift_score             = self._drift_score(candidate, previous)

        # ── 2. Decide ─────────────────────────────────────────────────
        action = "accepted"

        if identity_stability < min_identity:
            action = "hard_rollback"
            notes.append(
                f"Identity stability {identity_stability:.3f} < minimum {min_identity:.3f}. "
                "Hard rollback to previous generation."
            )
        elif drift_score > MAX_DRIFT_THRESHOLD:
            action = "soft_rollback"
            notes.append(
                f"Drift score {drift_score:.3f} > threshold {MAX_DRIFT_THRESHOLD}. "
                "Soft rollback: blending candidate 50/50 with previous."
            )
        elif knowledge_retention < 0.5:
            notes.append(f"Knowledge retention low ({knowledge_retention:.3f}) — stabilisation pass will compensate.")
        elif capability_preservation < 0.6:
            notes.append(f"Capability preservation low ({capability_preservation:.3f}).")

        # ── 3. Apply rollback / stabilisation ────────────────────────
        if action == "hard_rollback":
            result = copy.deepcopy(previous)
            result.metadata["sustain_action"] = "hard_rollback"
        elif action == "soft_rollback":
            result = self._soft_blend(candidate, previous)
            result.metadata["sustain_action"] = "soft_rollback"
        else:
            result = self._stabilisation_pass(candidate, previous)
            result.metadata["sustain_action"] = "accepted"

        # ── 4. Record ─────────────────────────────────────────────────
        result.generation = generation
        report = SustainReport(
            generation=generation,
            identity_stability=identity_stability,
            knowledge_retention=knowledge_retention,
            capability_preservation=capability_preservation,
            drift_score=drift_score,
            action=action,
            notes=notes,
        )
        self.reports.append(report)
        print(report.summary())

        return result

    # ------------------------------------------------------------------
    # Stability metrics
    # ------------------------------------------------------------------

    def _identity_stability(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
    ) -> float:
        """How stable is the identity zone compared to the previous state."""
        prev_id   = previous.genome.get("identity")
        cand_id   = candidate.genome.get("identity")
        if prev_id is None or cand_id is None:
            return 1.0
        # Stability is measured as the candidate's raw identity stability score
        return cand_id.stability

    def _knowledge_retention(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
    ) -> float:
        """
        Mean zone stability across all zones.
        A drop in stability = risk of catastrophic forgetting.
        """
        if not candidate.genome:
            return 0.0
        return sum(z.stability for z in candidate.genome.values()) / len(candidate.genome)

    def _capability_preservation(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
    ) -> float:
        """
        How many capabilities in the candidate are at least as strong
        as in the previous state (as a fraction).
        """
        if not previous.genome:
            return 1.0
        preserved = 0
        total = 0
        for zone_name, prev_zone in previous.genome.items():
            cand_zone = candidate.genome.get(zone_name)
            if cand_zone is None:
                continue
            total += 1
            if cand_zone.current_strength >= prev_zone.current_strength - 0.05:  # 5% grace
                preserved += 1
        return preserved / total if total else 1.0

    def _drift_score(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
    ) -> float:
        """
        Overall drift: mean absolute change in zone strengths.
        Higher = more drift.
        """
        if not candidate.genome or not previous.genome:
            return 0.0
        deltas = []
        for zone_name, prev_zone in previous.genome.items():
            cand_zone = candidate.genome.get(zone_name)
            if cand_zone is not None:
                deltas.append(abs(cand_zone.current_strength - prev_zone.current_strength))
        return sum(deltas) / len(deltas) if deltas else 0.0

    # ------------------------------------------------------------------
    # Stabilisation pass
    # ------------------------------------------------------------------

    def _stabilisation_pass(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
    ) -> IntelligenceState:
        """
        Gently re-anchor each genome zone toward its previous value
        using the stabilisation_strength as the blending coefficient.

        new_strength = candidate + alpha * (previous - candidate)
        where alpha = stabilisation_strength
        """
        result = copy.deepcopy(candidate)
        alpha  = self.stabilisation_strength

        for zone_name, cand_zone in result.genome.items():
            prev_zone = previous.genome.get(zone_name)
            if prev_zone is None or zone_name == "identity":
                continue
            anchored_strength = (
                cand_zone.current_strength
                + alpha * (prev_zone.current_strength - cand_zone.current_strength)
            )
            anchored_stability = (
                cand_zone.stability
                + alpha * (prev_zone.stability - cand_zone.stability)
            )
            result.genome[zone_name] = GenomeZone(
                name=zone_name,
                mutation_rate=cand_zone.mutation_rate,
                current_strength=min(1.0, max(0.0, anchored_strength)),
                stability=min(1.0, max(0.0, anchored_stability)),
            )
        return result

    def _soft_blend(
        self,
        candidate: IntelligenceState,
        previous: IntelligenceState,
        ratio: float = 0.5,
    ) -> IntelligenceState:
        """Blend candidate and previous 50/50 to reduce excessive drift."""
        result = copy.deepcopy(candidate)
        for zone_name, cand_zone in result.genome.items():
            prev_zone = previous.genome.get(zone_name)
            if prev_zone is None or zone_name == "identity":
                continue
            blended_strength  = ratio * cand_zone.current_strength + (1 - ratio) * prev_zone.current_strength
            blended_stability = ratio * cand_zone.stability         + (1 - ratio) * prev_zone.stability
            result.genome[zone_name] = GenomeZone(
                name=zone_name,
                mutation_rate=cand_zone.mutation_rate,
                current_strength=min(1.0, blended_strength),
                stability=min(1.0, blended_stability),
            )
        return result

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_history(self) -> List[Dict[str, Any]]:
        return [
            {
                "generation": r.generation,
                "action": r.action,
                "identity_stability": round(r.identity_stability, 4),
                "knowledge_retention": round(r.knowledge_retention, 4),
                "capability_preservation": round(r.capability_preservation, 4),
                "drift_score": round(r.drift_score, 4),
                "notes": r.notes,
            }
            for r in self.reports
        ]

    def __repr__(self) -> str:
        return (f"SustainEngine(reports={len(self.reports)}, "
                f"stabilisation_strength={self.stabilisation_strength})")
