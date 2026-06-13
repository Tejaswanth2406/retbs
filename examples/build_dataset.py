"""
build_dataset.py — Construct a RETBS radiation-augmented training dataset.

Demonstrates:
  - Building a knowledge genome graph
  - Adding chameleon form samples
  - Adding radiation samples of each type
  - Adding quantum-inspired reasoning samples
  - Saving to JSONL for use with Axolotl / TRL

Run:
    python -m retbs.examples.build_dataset
"""

from pathlib import Path

from retbs.knowledge.atomic_knowledge import build_cs_starter_graph, KAtom, KnowledgeGenomeGraph
from retbs.knowledge.dataset_builder import DatasetBuilder


def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║        RETBS — Dataset Builder Example              ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. Build / extend the knowledge genome graph ──────────────────
    graph = build_cs_starter_graph()

    # Add a few cross-domain atoms (physics + cs → emergence)
    graph.add_atom(KAtom("entropy",       "physics", "Measure of disorder.", confidence=0.6, utility=0.5))
    graph.add_atom(KAtom("gradient",      "math",    "Rate of change.",       confidence=0.8, utility=0.85))
    graph.add_atom(KAtom("optimisation",  "math",    "Finding minima/maxima.",confidence=0.75, utility=0.8,
                          dependencies=["gradient"]))
    graph.add_atom(KAtom("backprop",      "ml",      "Gradient-based learning.",confidence=0.7, utility=0.8,
                          dependencies=["gradient", "optimisation", "recursion"]))
    graph.add_atom(KAtom("transformer",   "ml",      "Attention-based architecture.",confidence=0.65, utility=0.75,
                          dependencies=["backprop", "algorithm"]))

    print(f"Knowledge graph: {graph.stats()}\n")

    # ── 2. Build the dataset ───────────────────────────────────────────
    builder = DatasetBuilder(seed=42)

    # Chameleon samples — teach form switching
    builder.add_chameleon_samples(
        forms=["research", "engineering", "scientific", "creative", "strategic", "educator"],
        n_per_form=30,
        generation=1,
    )

    # Knowledge radiation — inject new concepts
    builder.add_radiation_samples(graph, radiation_type="knowledge",     n_samples=80, generation=1)

    # Contradiction radiation — force new reasoning structures
    builder.add_radiation_samples(graph, radiation_type="contradiction",  n_samples=40, generation=1)

    # Adversarial radiation — stress-test edge cases
    builder.add_radiation_samples(graph, radiation_type="adversarial",   n_samples=40, generation=1)

    # Emergence radiation — frontier unsolvable problems
    builder.add_radiation_samples(graph, radiation_type="emergence",     n_samples=30, generation=1)

    # Quantum-inspired multi-state reasoning
    builder.add_quantum_samples(n_samples=20, generation=1)

    # ── 3. Save ────────────────────────────────────────────────────────
    out_dir = Path("./retbs_datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = builder.save(out_dir / "gen1_training.jsonl")

    print(f"\nDataset stats:")
    stats = builder.stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print(f"\nTotal samples written: {n}")
    print(f"Output: {out_dir / 'gen1_training.jsonl'}")
    print("\nPass this file to Axolotl or TRL for fine-tuning.")


if __name__ == "__main__":
    main()
