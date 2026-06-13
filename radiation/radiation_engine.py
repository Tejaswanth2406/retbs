"""
RadiationEngine — generates and applies controlled perturbations to intelligence states.

Radiation Types (from RETBS framework):
  knowledge     — injects new information (new papers, APIs, discoveries)
  contradiction — injects paradoxes and conflicting truths to force new structures
  adversarial   — injects attack patterns, edge cases, failure modes
  environmental — injects domain-level shifts (regulation changes, market shifts)
  emergence     — injects tasks the model cannot currently solve (frontier pressure)
  entropy       — destabilises stagnant solutions to prevent local optima lock-in

Radiation Intensities (α → Ω):
  alpha   — small, local mutations (0.0 – 0.1)
  beta    — structural mutations   (0.1 – 0.3)
  gamma   — architecture-level     (0.3 – 0.6)
  omega   — paradigm mutations     (0.6 – 1.0)
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from retbs.core.kernel import IdentityKernel
from retbs.core.intelligence_state import IntelligenceState, GenomeZone


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RadiationPacket:
    """A single radiation event to be applied to an IntelligenceState."""
    radiation_type: str          # knowledge | contradiction | adversarial | environmental | emergence | entropy
    source_data: List[Any]       # raw data items (strings, dicts, samples …)
    intensity: float             # 0.0 – 1.0
    target_zones: List[str]      # which genome zones this radiation affects
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def intensity_class(self) -> str:
        if self.intensity < 0.1:
            return "alpha"
        elif self.intensity < 0.3:
            return "beta"
        elif self.intensity < 0.6:
            return "gamma"
        else:
            return "omega"

    def __repr__(self) -> str:
        return (f"RadiationPacket(type={self.radiation_type}, "
                f"intensity={self.intensity:.2f} [{self.intensity_class}], "
                f"zones={self.target_zones}, items={len(self.source_data)})")


# ─────────────────────────────────────────────────────────────────────────────
# Radiation Engine
# ─────────────────────────────────────────────────────────────────────────────

# Maps each radiation type to the genome zones it typically perturbs
_ZONE_AFFINITY: Dict[str, List[str]] = {
    "knowledge":     ["domain", "memory"],
    "contradiction": ["reasoning", "creativity"],
    "adversarial":   ["reasoning", "domain"],
    "environmental": ["domain", "planning"],
    "emergence":     ["creativity", "planning", "reasoning"],
    "entropy":       ["creativity", "domain", "memory"],
}


class RadiationEngine:
    """
    Generates RadiationPackets and applies them to IntelligenceStates.

    The engine respects the IdentityKernel's EvolutionPolicy — it will never
    apply radiation to zones whose allowed mutation rate is 0.
    """

    def __init__(self, kernel: IdentityKernel, seed: Optional[int] = None) -> None:
        self.kernel = kernel
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Packet creation
    # ------------------------------------------------------------------

    def generate_packet(
        self,
        radiation_type: str,
        source_data: List[Any],
        intensity: float,
        target_zones: Optional[List[str]] = None,
    ) -> RadiationPacket:
        """
        Create a RadiationPacket, automatically filtering zones that are
        forbidden by the IdentityKernel.
        """
        if radiation_type not in _ZONE_AFFINITY:
            raise ValueError(
                f"Unknown radiation type '{radiation_type}'. "
                f"Valid types: {list(_ZONE_AFFINITY)}"
            )
        zones = target_zones or _ZONE_AFFINITY[radiation_type]
        # Filter out immutable zones
        allowed_zones = [
            z for z in zones
            if self.kernel.validate_mutation(z, intensity)
        ]
        if not allowed_zones:
            print(f"[RadiationEngine] All target zones blocked by kernel policy — packet neutered.")
            intensity = 0.0

        return RadiationPacket(
            radiation_type=radiation_type,
            source_data=source_data,
            intensity=intensity,
            target_zones=allowed_zones,
        )

    def default_packet(self) -> RadiationPacket:
        """A gentle knowledge radiation used when no sources are provided."""
        return self.generate_packet(
            radiation_type="knowledge",
            source_data=["default_knowledge_pulse"],
            intensity=0.05,
        )

    # ------------------------------------------------------------------
    # Applying radiation to a state
    # ------------------------------------------------------------------

    def apply(
        self,
        state: IntelligenceState,
        packets: List[RadiationPacket],
    ) -> IntelligenceState:
        """
        Apply a list of RadiationPackets to an IntelligenceState.

        Each packet perturbs the genome zones it targets.
        Returns a NEW state (original is untouched).
        """
        new_state = copy.deepcopy(state)

        for packet in packets:
            print(f"  → Applying {packet}")
            for zone_name in packet.target_zones:
                if zone_name not in new_state.genome:
                    continue  # zone doesn't exist on this state
                zone = new_state.genome[zone_name]
                # Delta is intensity scaled by a random noise factor
                delta = packet.intensity * self._rng.uniform(0.5, 1.5)
                new_state.genome[zone_name] = zone.apply_mutation(delta)

            # Record the radiation event
            new_state.record_radiation(
                radiation_type=packet.radiation_type,
                source=str(packet.source_data[:3]),   # brief record
                intensity=packet.intensity,
            )

        return new_state

    # ------------------------------------------------------------------
    # Dataset mutation (for actual fine-tuning use)
    # ------------------------------------------------------------------

    def mutate_dataset(
        self,
        base_dataset: List[Dict[str, Any]],
        packet: RadiationPacket,
    ) -> List[Dict[str, Any]]:
        """
        Produce a mutated version of a training dataset by injecting
        radiation items according to the packet type.

        Returns a new dataset with injected samples. The caller passes
        this to their fine-tuning script.
        """
        mutated = list(base_dataset)
        n_inject = max(1, int(len(base_dataset) * packet.intensity))

        if packet.radiation_type == "knowledge":
            for item in packet.source_data[:n_inject]:
                mutated.append({"source": "radiation_knowledge", "content": item})

        elif packet.radiation_type == "contradiction":
            for item in packet.source_data[:n_inject]:
                mutated.append({
                    "source": "radiation_contradiction",
                    "content": item,
                    "instruction": "Reconcile the following conflicting information:",
                })

        elif packet.radiation_type == "adversarial":
            for item in packet.source_data[:n_inject]:
                mutated.append({
                    "source": "radiation_adversarial",
                    "content": item,
                    "instruction": "Identify and correct the flaw in the following:",
                })

        elif packet.radiation_type == "emergence":
            for item in packet.source_data[:n_inject]:
                mutated.append({
                    "source": "radiation_emergence",
                    "content": item,
                    "instruction": "Solve the following problem you have never encountered before:",
                })

        elif packet.radiation_type in ("environmental", "entropy"):
            self._rng.shuffle(mutated)
            mutated = mutated[:int(len(mutated) * (1 - packet.intensity * 0.2))]
            for item in packet.source_data[:n_inject]:
                mutated.append({"source": f"radiation_{packet.radiation_type}", "content": item})

        return mutated

    def __repr__(self) -> str:
        return f"RadiationEngine(kernel={self.kernel.mission[:30]}...)"
