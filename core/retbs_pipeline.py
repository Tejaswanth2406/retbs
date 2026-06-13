"""
RETBSPipeline — orchestrates the full evolutionary cycle.

Governing equation:
  I_{t+1} = S( B( T( E( R(I_t) ) ) ) )

  R = Radiate   → perturb the intelligence state with new knowledge/challenges
  E = Evolve    → generate and select mutant candidate states
  T = Transform → apply morphological transformation for the target task
  B = Blend     → merge top survivors into a hybrid state
  S = Sustain   → stabilise and preserve identity before the next generation
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from retbs.core.kernel import IdentityKernel, default_kernel
from retbs.core.intelligence_state import IntelligenceState
from retbs.radiation.radiation_engine import RadiationEngine, RadiationPacket
from retbs.evolution.evolution_arena import EvolutionArena
from retbs.transformation.transform_engine import TransformEngine
from retbs.blending.blend_engine import BlendEngine
from retbs.sustain.sustain_engine import SustainEngine


class RETBSPipeline:
    """
    Full RETBS evolutionary pipeline.

    Usage:
        kernel  = default_kernel()
        initial = IntelligenceState(generation=0, genome=IntelligenceState.default_genome())
        pipe    = RETBSPipeline(kernel, initial)

        for gen in range(10):
            next_state = pipe.run_generation(radiation_sources=[...])
            print(next_state)
    """

    def __init__(
        self,
        kernel: IdentityKernel,
        initial_state: IntelligenceState,
        *,
        radiation_engine: Optional[RadiationEngine] = None,
        evolution_arena: Optional[EvolutionArena] = None,
        transform_engine: Optional[TransformEngine] = None,
        blend_engine: Optional[BlendEngine] = None,
        sustain_engine: Optional[SustainEngine] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        self.kernel = kernel
        self.current_state = initial_state
        self.lineage: List[IntelligenceState] = [initial_state]

        # Inject or create default engines
        self.radiation_engine  = radiation_engine  or RadiationEngine(kernel)
        self.evolution_arena   = evolution_arena   or EvolutionArena(kernel)
        self.transform_engine  = transform_engine  or TransformEngine(kernel)
        self.blend_engine      = blend_engine      or BlendEngine(kernel)
        self.sustain_engine    = sustain_engine    or SustainEngine(kernel)

        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self._generation_logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Core cycle
    # ------------------------------------------------------------------

    def run_generation(
        self,
        radiation_sources: Optional[List[Dict[str, Any]]] = None,
        target_forms: Optional[List[str]] = None,
        n_mutants: int = 5,
    ) -> IntelligenceState:
        """
        Run one full RETBS generation cycle and return the next intelligence state.

        Args:
            radiation_sources: List of dicts describing what to inject, e.g.
                [{"type": "knowledge", "data": [...], "intensity": 0.2}]
            target_forms: Desired morphological forms, e.g. ["research", "engineering"]
            n_mutants: Number of mutant candidates to generate in the Evolution Arena.

        Returns:
            The next-generation IntelligenceState.
        """
        gen_start = time.time()
        gen_num   = self.current_state.generation + 1
        print(f"\n{'='*60}")
        print(f"[RETBS] Generation {gen_num} — Start")
        print(f"{'='*60}")

        # ── R : Radiate ────────────────────────────────────────────────
        print(f"\n[R] Radiation phase")
        packets: List[RadiationPacket] = []
        for src in (radiation_sources or []):
            pkt = self.radiation_engine.generate_packet(
                radiation_type=src.get("type", "knowledge"),
                source_data=src.get("data", []),
                intensity=src.get("intensity", 0.1),
            )
            packets.append(pkt)
        if not packets:
            packets = [self.radiation_engine.default_packet()]
        radiated_state = self.radiation_engine.apply(self.current_state, packets)

        # ── E : Evolve ─────────────────────────────────────────────────
        print(f"\n[E] Evolution phase ({n_mutants} mutants)")
        mutants = self.evolution_arena.generate_mutants(radiated_state, n=n_mutants)
        survivors = self.evolution_arena.evaluate_and_select(mutants)
        print(f"    {len(survivors)} survivors selected from {len(mutants)} mutants")

        # ── T : Transform ──────────────────────────────────────────────
        print(f"\n[T] Transformation phase")
        transformed = [
            self.transform_engine.transform(s, forms=target_forms or ["general"])
            for s in survivors
        ]

        # ── B : Blend ──────────────────────────────────────────────────
        print(f"\n[B] Blending phase")
        blended_state = self.blend_engine.blend(transformed)

        # ── S : Sustain ────────────────────────────────────────────────
        print(f"\n[S] Sustain phase")
        next_state = self.sustain_engine.stabilise(blended_state, self.current_state, gen_num)

        # ── Logging ────────────────────────────────────────────────────
        elapsed = time.time() - gen_start
        log_entry = {
            "generation": gen_num,
            "elapsed_s": round(elapsed, 3),
            "n_mutants": n_mutants,
            "n_survivors": len(survivors),
            "adaptive_coherence": next_state.adaptive_coherence(),
            "identity_stability": next_state.identity_stability(),
            "state_id": next_state.state_id,
        }
        self._generation_logs.append(log_entry)

        print(f"\n[RETBS] Generation {gen_num} complete in {elapsed:.2f}s")
        print(f"        Adaptive Coherence : {next_state.adaptive_coherence():.4f}")
        print(f"        Identity Stability : {next_state.identity_stability():.4f}")
        print(f"        State ID           : {next_state.state_id[:16]}...")

        if self.log_dir:
            next_state.save(self.log_dir / f"gen_{gen_num:04d}.json")
            (self.log_dir / "pipeline_log.json").write_text(
                json.dumps(self._generation_logs, indent=2)
            )

        self.lineage.append(next_state)
        self.current_state = next_state
        return next_state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def run_n_generations(
        self,
        n: int,
        radiation_schedule: Optional[List[List[Dict[str, Any]]]] = None,
        **kwargs,
    ) -> List[IntelligenceState]:
        """Run n generations and return the full list of produced states."""
        results = []
        for i in range(n):
            srcs = (radiation_schedule[i] if radiation_schedule and i < len(radiation_schedule)
                    else kwargs.get("radiation_sources"))
            state = self.run_generation(radiation_sources=srcs, **{k: v for k, v in kwargs.items()
                                                                    if k != "radiation_sources"})
            results.append(state)
            # Early stop if stagnant
            if len(self.lineage) > self.kernel.evolution_policy.max_generations_without_gain + 1:
                recent = [s.adaptive_coherence() for s in
                          self.lineage[-(self.kernel.evolution_policy.max_generations_without_gain + 1):]]
                if max(recent) - min(recent) < 0.001:
                    print(f"[RETBS] Early stop: no improvement over last "
                          f"{self.kernel.evolution_policy.max_generations_without_gain} generations.")
                    break
        return results

    def lineage_summary(self) -> List[Dict[str, Any]]:
        """Return a compact summary of all generations."""
        return [
            {
                "generation": s.generation,
                "state_id": s.state_id[:8],
                "adaptive_coherence": round(s.adaptive_coherence(), 4),
                "identity_stability": round(s.identity_stability(), 4),
                "n_radiation_events": len(s.radiation_history),
            }
            for s in self.lineage
        ]

    def best_state(self) -> IntelligenceState:
        """Return the generation with the highest adaptive coherence."""
        return max(self.lineage, key=lambda s: s.adaptive_coherence())

    def __repr__(self) -> str:
        return (f"RETBSPipeline(generations={len(self.lineage)-1}, "
                f"current_C={self.current_state.adaptive_coherence():.4f})")
