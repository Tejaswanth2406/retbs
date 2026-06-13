"""
LineageTracker — records the full genealogy of a RETBS model lineage.

Every intelligence state has ancestry. The tracker provides:
  - A directed acyclic graph (DAG) of all states and their parent relationships
  - Generation-by-generation metric progression
  - Lineage export (JSON) for governance and audit
  - Best-path tracing: which evolutionary branch produced the champion?
  - Rollback snapshots: retrieve any past state by generation or ID
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from retbs.core.intelligence_state import IntelligenceState


class LineageTracker:
    """
    Records and queries the full evolutionary lineage of RETBS states.

    Usage:
        tracker = LineageTracker()
        tracker.register(gen0_state)
        tracker.register(gen1_state)
        tracker.register(gen2_state)

        best = tracker.best_state()
        path = tracker.best_path()
        tracker.save("lineage.json")
    """

    def __init__(self, lineage_name: str = "retbs_lineage") -> None:
        self.lineage_name = lineage_name
        self._states: Dict[str, IntelligenceState] = {}        # id → state
        self._by_generation: Dict[int, List[str]] = defaultdict(list)  # gen → [ids]
        self._registration_order: List[str] = []
        self._created_at: float = time.time()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, state: IntelligenceState) -> None:
        """Register a new state in the lineage."""
        self._states[state.state_id] = state
        self._by_generation[state.generation].append(state.state_id)
        self._registration_order.append(state.state_id)
        print(f"  [LineageTracker] Registered gen={state.generation} "
              f"id={state.state_id[:8]} C={state.adaptive_coherence():.4f}")

    def register_all(self, states: List[IntelligenceState]) -> None:
        for s in states:
            self.register(s)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, state_id: str) -> Optional[IntelligenceState]:
        return self._states.get(state_id)

    def generation(self, gen: int) -> List[IntelligenceState]:
        """Return all states from a specific generation."""
        return [self._states[sid] for sid in self._by_generation.get(gen, [])]

    def latest(self) -> Optional[IntelligenceState]:
        """Return the most recently registered state."""
        if not self._registration_order:
            return None
        return self._states[self._registration_order[-1]]

    def best_state(self) -> Optional[IntelligenceState]:
        """Return the state with the highest adaptive coherence."""
        if not self._states:
            return None
        return max(self._states.values(), key=lambda s: s.adaptive_coherence())

    def best_per_generation(self) -> List[IntelligenceState]:
        """Return the best state from each generation, sorted by generation."""
        result = []
        for gen in sorted(self._by_generation.keys()):
            states = self.generation(gen)
            if states:
                result.append(max(states, key=lambda s: s.adaptive_coherence()))
        return result

    def best_path(self) -> List[IntelligenceState]:
        """
        Trace the ancestry of the best state back to generation 0.
        Returns the list ordered from root → champion.
        """
        champion = self.best_state()
        if champion is None:
            return []

        path: List[IntelligenceState] = [champion]
        current = champion
        while current.parent_ids:
            parent_id = current.parent_ids[0]
            parent = self._states.get(parent_id)
            if parent is None:
                break
            path.append(parent)
            current = parent

        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def coherence_over_time(self) -> List[Tuple[int, float]]:
        """Return (generation, best_coherence) tuples for plotting."""
        return [
            (s.generation, s.adaptive_coherence())
            for s in self.best_per_generation()
        ]

    def summary(self) -> Dict[str, Any]:
        best = self.best_state()
        return {
            "lineage_name": self.lineage_name,
            "total_states": len(self._states),
            "generations": sorted(self._by_generation.keys()),
            "best_state_id": best.state_id[:8] if best else None,
            "best_coherence": round(best.adaptive_coherence(), 4) if best else None,
            "coherence_over_time": self.coherence_over_time(),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Export the full lineage as JSON."""
        data = {
            "lineage_name": self.lineage_name,
            "created_at": self._created_at,
            "exported_at": time.time(),
            "states": [s.to_dict() for s in self._states.values()],
            "summary": self.summary(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        print(f"[LineageTracker] Lineage saved → {path} ({len(self._states)} states)")

    @classmethod
    def load(cls, path: str | Path) -> "LineageTracker":
        data = json.loads(Path(path).read_text())
        tracker = cls(lineage_name=data.get("lineage_name", "retbs_lineage"))
        tracker._created_at = data.get("created_at", time.time())
        for state_dict in data.get("states", []):
            tracker.register(IntelligenceState.load_from_dict(state_dict))
        return tracker

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_tree(self) -> None:
        """Print a compact ASCII lineage tree."""
        print(f"\n{'─'*60}")
        print(f"  RETBS Lineage: {self.lineage_name}")
        print(f"{'─'*60}")
        for gen in sorted(self._by_generation.keys()):
            states = self.generation(gen)
            print(f"  Gen {gen:03d} ({len(states)} states)")
            for s in sorted(states, key=lambda x: -x.adaptive_coherence()):
                mark = "★" if s == self.best_state() else " "
                print(f"    {mark} {s.state_id[:8]}  C={s.adaptive_coherence():.4f}  "
                      f"ID={s.identity_stability():.4f}")
        print(f"{'─'*60}\n")

    def __len__(self) -> int:
        return len(self._states)

    def __repr__(self) -> str:
        return (f"LineageTracker(name={self.lineage_name!r}, "
                f"states={len(self._states)}, "
                f"generations={len(self._by_generation)})")
