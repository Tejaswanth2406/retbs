"""
IntelligenceState — a snapshot of one RETBS model generation.

I(t) = (K_t, M_t, A_t, F_t)
  K_t = knowledge state
  M_t = memory/capability state
  A_t = adaptation state
  F_t = field interactions
"""

from __future__ import annotations

import uuid
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class CapabilityScore:
    """Benchmark scores for a single capability dimension."""
    name: str
    score: float          # 0.0 – 1.0
    confidence: float     # 0.0 – 1.0
    evaluated_at: float = field(default_factory=time.time)

    def weighted(self) -> float:
        return self.score * self.confidence


@dataclass
class GenomeZone:
    """
    Represents a mutable region of the model's 'genome'.

    Zones:
        identity   — immutable (mutation_rate = 0)
        reasoning  — very stable
        creativity — flexible
        domain     — moderate, domain-specific
    """
    name: str
    mutation_rate: float     # 0.0 = immutable, 1.0 = fully mutable
    current_strength: float  # 0.0 – 1.0, how capable this zone currently is
    stability: float         # 0.0 – 1.0, resistance to collapse

    def apply_mutation(self, delta: float) -> "GenomeZone":
        """Return a new zone after applying a mutation delta."""
        if self.mutation_rate == 0.0:
            return self  # immutable
        new_strength = max(0.0, min(1.0, self.current_strength + delta * self.mutation_rate))
        new_stability = max(0.0, min(1.0, self.stability - abs(delta) * 0.1))
        return GenomeZone(
            name=self.name,
            mutation_rate=self.mutation_rate,
            current_strength=new_strength,
            stability=new_stability,
        )


@dataclass
class IntelligenceState:
    """
    A fully-described snapshot of one RETBS model generation.

    Each state has:
      - A unique ID and generation number.
      - Parent lineage (for genealogy tracking).
      - Genome zones that define capability structure.
      - Capability scores from the Evolution Arena.
      - Radiation history (what perturbations shaped this state).
      - An adaptive coherence score: C = A + G + R + P - D
    """

    # Identity
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0
    parent_ids: List[str] = field(default_factory=list)
    model_path: Optional[str] = None         # Path to the fine-tuned weights
    adapter_path: Optional[str] = None       # LoRA adapter path
    created_at: float = field(default_factory=time.time)

    # Genome
    genome: Dict[str, GenomeZone] = field(default_factory=dict)

    # Capability scores from Evolution Arena
    capability_scores: List[CapabilityScore] = field(default_factory=list)

    # Radiation history
    radiation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def adaptability(self) -> float:
        """Average of creativity + domain zone strengths."""
        zones = [self.genome[z].current_strength
                 for z in ("creativity", "domain") if z in self.genome]
        return sum(zones) / len(zones) if zones else 0.0

    @property
    def generalization(self) -> float:
        """Mean weighted capability score."""
        if not self.capability_scores:
            return 0.0
        return sum(c.weighted() for c in self.capability_scores) / len(self.capability_scores)

    @property
    def retention(self) -> float:
        """Mean genome stability (proxy for catastrophic forgetting resistance)."""
        if not self.genome:
            return 0.0
        return sum(z.stability for z in self.genome.values()) / len(self.genome)

    @property
    def performance(self) -> float:
        """Max weighted capability score."""
        if not self.capability_scores:
            return 0.0
        return max(c.weighted() for c in self.capability_scores)

    @property
    def drift(self) -> float:
        """Proxy: instability of identity + reasoning zones."""
        zones = [self.genome[z] for z in ("identity", "reasoning") if z in self.genome]
        if not zones:
            return 0.0
        return 1.0 - sum(z.stability for z in zones) / len(zones)

    def adaptive_coherence(self) -> float:
        """
        C = A + G + R + P - D
        The core RETBS fitness metric.
        """
        return self.adaptability + self.generalization + self.retention + self.performance - self.drift

    def identity_stability(self) -> float:
        """Stability of the identity genome zone (0 = collapsed, 1 = stable)."""
        zone = self.genome.get("identity")
        return zone.stability if zone else 1.0

    # ------------------------------------------------------------------
    # Genome helpers
    # ------------------------------------------------------------------

    @classmethod
    def default_genome(cls) -> Dict[str, GenomeZone]:
        return {
            "identity":  GenomeZone("identity",  mutation_rate=0.0,  current_strength=1.0, stability=1.0),
            "reasoning": GenomeZone("reasoning", mutation_rate=0.05, current_strength=0.7, stability=0.9),
            "creativity":GenomeZone("creativity",mutation_rate=0.3,  current_strength=0.5, stability=0.7),
            "domain":    GenomeZone("domain",     mutation_rate=0.2,  current_strength=0.6, stability=0.8),
            "memory":    GenomeZone("memory",     mutation_rate=0.1,  current_strength=0.6, stability=0.85),
            "planning":  GenomeZone("planning",   mutation_rate=0.15, current_strength=0.6, stability=0.8),
        }

    def add_capability_score(self, name: str, score: float, confidence: float = 1.0) -> None:
        self.capability_scores.append(CapabilityScore(name, score, confidence))

    def record_radiation(self, radiation_type: str, source: str, intensity: float) -> None:
        self.radiation_history.append({
            "type": radiation_type,
            "source": source,
            "intensity": intensity,
            "timestamp": time.time(),
        })

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "state_id": self.state_id,
            "generation": self.generation,
            "parent_ids": self.parent_ids,
            "model_path": self.model_path,
            "adapter_path": self.adapter_path,
            "created_at": self.created_at,
            "genome": {k: asdict(v) for k, v in self.genome.items()},
            "capability_scores": [asdict(c) for c in self.capability_scores],
            "radiation_history": self.radiation_history,
            "metadata": self.metadata,
            "metrics": {
                "adaptability": self.adaptability,
                "generalization": self.generalization,
                "retention": self.retention,
                "performance": self.performance,
                "drift": self.drift,
                "adaptive_coherence": self.adaptive_coherence(),
                "identity_stability": self.identity_stability(),
            },
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_from_dict(cls, data: dict) -> "IntelligenceState":
        genome = {k: GenomeZone(**v) for k, v in data.get("genome", {}).items()}
        scores = [CapabilityScore(**c) for c in data.get("capability_scores", [])]
        return cls(
            state_id=data["state_id"],
            generation=data["generation"],
            parent_ids=data.get("parent_ids", []),
            model_path=data.get("model_path"),
            adapter_path=data.get("adapter_path"),
            created_at=data.get("created_at", time.time()),
            genome=genome,
            capability_scores=scores,
            radiation_history=data.get("radiation_history", []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "IntelligenceState":
        data = json.loads(Path(path).read_text())
        genome = {k: GenomeZone(**v) for k, v in data.get("genome", {}).items()}
        scores = [CapabilityScore(**c) for c in data.get("capability_scores", [])]
        state = cls(
            state_id=data["state_id"],
            generation=data["generation"],
            parent_ids=data.get("parent_ids", []),
            model_path=data.get("model_path"),
            adapter_path=data.get("adapter_path"),
            created_at=data.get("created_at", time.time()),
            genome=genome,
            capability_scores=scores,
            radiation_history=data.get("radiation_history", []),
            metadata=data.get("metadata", {}),
        )
        return state

    def __repr__(self) -> str:
        return (f"IntelligenceState(gen={self.generation}, "
                f"C={self.adaptive_coherence():.3f}, "
                f"id={self.state_id[:8]}...)")
