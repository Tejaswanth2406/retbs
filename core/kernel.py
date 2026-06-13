"""
IdentityKernel — the immutable core of every RETBS intelligence.

The Identity Kernel is NEVER directly fine-tuned. It defines the model's
mission, values, safety constraints, and evolution policies. Think of it
as the "genetic root" that persists across all generations.

Laws:
  - Mission does not mutate.
  - Values do not mutate.
  - Safety constraints do not mutate.
  - Everything else can evolve.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


@dataclass(frozen=True)
class SafetyConstraint:
    """An immutable safety rule that governs all model generations."""
    name: str
    description: str
    severity: str = "critical"  # "critical" | "warning" | "info"


@dataclass(frozen=True)
class EvolutionPolicy:
    """Rules controlling how mutations are applied across genome zones."""
    max_mutation_rate: float = 0.3      # Max allowed parameter perturbation magnitude
    identity_zone_rate: float = 0.0     # Identity zone: NEVER mutated
    reasoning_zone_rate: float = 0.05   # Core reasoning: mutated very slowly
    creativity_zone_rate: float = 0.3   # Creativity: can mutate quickly
    domain_zone_rate: float = 0.2       # Domain knowledge: moderate mutation
    min_identity_stability: float = 0.85  # Reject mutations that drop below this
    max_generations_without_gain: int = 5  # Stop if no improvement after N gens


@dataclass
class IdentityKernel:
    """
    The immutable nucleus of a RETBS intelligence.

    Usage:
        kernel = IdentityKernel(
            mission="Be a highly capable, safe AI assistant.",
            values=["honesty", "helpfulness", "safety"],
            safety_constraints=[...],
            evolution_policy=EvolutionPolicy()
        )
        kernel.save("kernel.json")
    """

    mission: str
    values: List[str]
    safety_constraints: List[SafetyConstraint] = field(default_factory=list)
    evolution_policy: EvolutionPolicy = field(default_factory=EvolutionPolicy)
    knowledge_schema_version: str = "1.0.0"

    # Computed once; never modified
    _fingerprint: str = field(init=False, repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        # Compute a stable fingerprint so any tampering is detectable
        content = json.dumps({
            "mission": self.mission,
            "values": sorted(self.values),
        }, sort_keys=True)
        object.__setattr__(self, "_fingerprint", hashlib.sha256(content.encode()).hexdigest())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_mutation(self, zone: str, proposed_rate: float) -> bool:
        """Return True if the proposed mutation rate is allowed for this zone."""
        limits = {
            "identity": self.evolution_policy.identity_zone_rate,
            "reasoning": self.evolution_policy.reasoning_zone_rate,
            "creativity": self.evolution_policy.creativity_zone_rate,
            "domain": self.evolution_policy.domain_zone_rate,
        }
        allowed = limits.get(zone, self.evolution_policy.max_mutation_rate)
        if proposed_rate > allowed:
            print(f"[IdentityKernel] Mutation REJECTED for zone '{zone}': "
                  f"{proposed_rate:.3f} > allowed {allowed:.3f}")
            return False
        return True

    def check_safety(self, capability_description: str) -> List[str]:
        """Return list of violated constraint names for a given capability."""
        violated = []
        for constraint in self.safety_constraints:
            # Placeholder: real implementation would use a classifier
            if any(kw in capability_description.lower()
                   for kw in ["harmful", "dangerous", "unsafe"]):
                violated.append(constraint.name)
        return violated

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "mission": self.mission,
            "values": self.values,
            "safety_constraints": [asdict(c) for c in self.safety_constraints],
            "evolution_policy": asdict(self.evolution_policy),
            "knowledge_schema_version": self.knowledge_schema_version,
            "fingerprint": self._fingerprint,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        print(f"[IdentityKernel] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "IdentityKernel":
        data = json.loads(Path(path).read_text())
        constraints = [SafetyConstraint(**c) for c in data.get("safety_constraints", [])]
        policy = EvolutionPolicy(**data.get("evolution_policy", {}))
        kernel = cls(
            mission=data["mission"],
            values=data["values"],
            safety_constraints=constraints,
            evolution_policy=policy,
            knowledge_schema_version=data.get("knowledge_schema_version", "1.0.0"),
        )
        stored_fp = data.get("fingerprint", "")
        if stored_fp and kernel._fingerprint != stored_fp:
            raise ValueError("[IdentityKernel] Fingerprint mismatch — kernel may have been tampered with!")
        return kernel

    def __repr__(self) -> str:
        return (f"IdentityKernel(mission={self.mission!r}, "
                f"values={self.values}, fp={self._fingerprint[:8]}...)")


# ------------------------------------------------------------------
# Factory helpers
# ------------------------------------------------------------------

def default_kernel() -> IdentityKernel:
    """Return a sensible default kernel for experimentation."""
    return IdentityKernel(
        mission="Be a continuously improving, safe, and adaptive intelligence.",
        values=["helpfulness", "honesty", "safety", "consistency"],
        safety_constraints=[
            SafetyConstraint("no_harm", "Never assist with harmful activities.", "critical"),
            SafetyConstraint("factuality", "Prioritise factual accuracy over fluency.", "warning"),
        ],
        evolution_policy=EvolutionPolicy(),
    )
