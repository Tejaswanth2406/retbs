"""
RETBS Metrics — scoring, benchmarking, and comparison utilities.

Core metric: Adaptive Coherence
  C = A + G + R + P - D
  A = adaptability, G = generalization, R = retention, P = performance, D = drift

Additional metrics:
  - Evolutionary gain       : coherence improvement per generation
  - Identity stability rate : % of generations that passed identity check
  - Radiation efficiency    : coherence gain per unit of radiation intensity
  - Capability profile      : per-zone strength radar chart data
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from retbs.core.intelligence_state import IntelligenceState


# ─────────────────────────────────────────────────────────────────────────────
# Single-state metrics
# ─────────────────────────────────────────────────────────────────────────────

def adaptive_coherence(state: IntelligenceState) -> float:
    """C = A + G + R + P - D  (the primary RETBS fitness metric)."""
    return state.adaptive_coherence()


def capability_profile(state: IntelligenceState) -> Dict[str, float]:
    """Zone-by-zone strength values — useful for radar/spider chart visualisation."""
    return {name: zone.current_strength for name, zone in state.genome.items()}


def stability_profile(state: IntelligenceState) -> Dict[str, float]:
    """Zone-by-zone stability values."""
    return {name: zone.stability for name, zone in state.genome.items()}


def radiation_count(state: IntelligenceState) -> int:
    return len(state.radiation_history)


def total_radiation_intensity(state: IntelligenceState) -> float:
    return sum(r.get("intensity", 0.0) for r in state.radiation_history)


# ─────────────────────────────────────────────────────────────────────────────
# Lineage / multi-state metrics
# ─────────────────────────────────────────────────────────────────────────────

def evolutionary_gain(states: List[IntelligenceState]) -> List[float]:
    """
    Return per-generation coherence gain:
      gain[i] = C(gen_i) - C(gen_i-1)
    """
    if len(states) < 2:
        return []
    scores = [s.adaptive_coherence() for s in states]
    return [scores[i] - scores[i - 1] for i in range(1, len(scores))]


def cumulative_gain(states: List[IntelligenceState]) -> float:
    """Total coherence improvement from first to last state."""
    if not states:
        return 0.0
    return states[-1].adaptive_coherence() - states[0].adaptive_coherence()


def identity_stability_rate(states: List[IntelligenceState], threshold: float = 0.85) -> float:
    """Fraction of states whose identity stability is above the threshold."""
    if not states:
        return 0.0
    passing = sum(1 for s in states if s.identity_stability() >= threshold)
    return passing / len(states)


def radiation_efficiency(states: List[IntelligenceState]) -> Optional[float]:
    """
    Coherence gain per unit of total radiation intensity applied.
    Higher = more efficient use of radiation.
    Returns None if no radiation was applied.
    """
    total_intensity = sum(total_radiation_intensity(s) for s in states)
    if total_intensity == 0:
        return None
    gain = cumulative_gain(states)
    return gain / total_intensity


def zone_improvement(
    initial: IntelligenceState,
    final: IntelligenceState,
) -> Dict[str, float]:
    """Per-zone strength improvement from initial to final state."""
    result = {}
    for zone_name, final_zone in final.genome.items():
        init_zone = initial.genome.get(zone_name)
        if init_zone:
            result[zone_name] = final_zone.current_strength - init_zone.current_strength
    return result


def compare_states(
    a: IntelligenceState,
    b: IntelligenceState,
) -> Dict[str, Any]:
    """Side-by-side comparison of two intelligence states."""
    return {
        "state_a": a.state_id[:8],
        "state_b": b.state_id[:8],
        "coherence_delta":         round(b.adaptive_coherence() - a.adaptive_coherence(), 4),
        "identity_stability_delta": round(b.identity_stability() - a.identity_stability(), 4),
        "adaptability_delta":      round(b.adaptability - a.adaptability, 4),
        "performance_delta":       round(b.performance - a.performance, 4),
        "retention_delta":         round(b.retention - a.retention, 4),
        "drift_delta":             round(b.drift - a.drift, 4),
        "zone_improvement":        zone_improvement(a, b),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary report
# ─────────────────────────────────────────────────────────────────────────────

def lineage_report(states: List[IntelligenceState]) -> Dict[str, Any]:
    """
    Full metrics report for a RETBS lineage.

    Args:
        states: Ordered list from generation 0 → latest.

    Returns:
        Dict with summary metrics.
    """
    if not states:
        return {"error": "no states provided"}

    gains = evolutionary_gain(states)

    return {
        "n_generations": len(states),
        "initial_coherence":    round(states[0].adaptive_coherence(), 4),
        "final_coherence":      round(states[-1].adaptive_coherence(), 4),
        "cumulative_gain":      round(cumulative_gain(states), 4),
        "mean_gen_gain":        round(sum(gains) / len(gains), 4) if gains else 0.0,
        "max_gen_gain":         round(max(gains), 4) if gains else 0.0,
        "identity_stability_rate": round(identity_stability_rate(states), 4),
        "radiation_efficiency": radiation_efficiency(states),
        "best_generation":      max(states, key=lambda s: s.adaptive_coherence()).generation,
        "final_capability_profile": capability_profile(states[-1]),
        "zone_improvement_total":   zone_improvement(states[0], states[-1]),
    }


def print_report(states: List[IntelligenceState]) -> None:
    """Pretty-print a lineage metrics report to stdout."""
    r = lineage_report(states)
    print("\n" + "═" * 55)
    print("  RETBS LINEAGE METRICS REPORT")
    print("═" * 55)
    print(f"  Generations          : {r['n_generations']}")
    print(f"  Initial Coherence    : {r['initial_coherence']}")
    print(f"  Final Coherence      : {r['final_coherence']}")
    print(f"  Cumulative Gain      : {r['cumulative_gain']:+.4f}")
    print(f"  Mean Gen Gain        : {r['mean_gen_gain']:+.4f}")
    print(f"  Identity Stability % : {r['identity_stability_rate']*100:.1f}%")
    eff = r["radiation_efficiency"]
    print(f"  Radiation Efficiency : {eff:.4f}" if eff else "  Radiation Efficiency : n/a")
    print(f"  Best Generation      : {r['best_generation']}")
    print("\n  Final Capability Profile:")
    for zone, val in sorted(r["final_capability_profile"].items()):
        bar = "█" * int(val * 20)
        print(f"    {zone:12s} {val:.3f}  {bar}")
    print("═" * 55 + "\n")
