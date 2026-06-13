"""
quickstart.py — Run a 5-generation RETBS cycle (no GPU required).

Demonstrates:
  1. Creating an IdentityKernel
  2. Initialising an IntelligenceState with a default genome
  3. Running 5 generations with varied radiation sources
  4. Printing a lineage metrics report

Run:
    python -m retbs.examples.quickstart
    # or
    python retbs/examples/quickstart.py
"""

from retbs.core.kernel import default_kernel
from retbs.core.intelligence_state import IntelligenceState
from retbs.core.retbs_pipeline import RETBSPipeline
from retbs.utils.lineage_tracker import LineageTracker
from retbs.utils.metrics import print_report


def main() -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print("║        RETBS — Quickstart (5 Generations)           ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. Build the identity kernel ──────────────────────────────────
    kernel = default_kernel()
    print(f"Kernel  : {kernel}\n")

    # ── 2. Create the initial (generation-0) state ────────────────────
    initial = IntelligenceState(
        generation=0,
        genome=IntelligenceState.default_genome(),
        metadata={"experiment": "quickstart"},
    )
    print(f"Initial : {initial}\n")

    # ── 3. Set up the pipeline ────────────────────────────────────────
    pipeline = RETBSPipeline(
        kernel=kernel,
        initial_state=initial,
        log_dir="./retbs_logs/quickstart",
    )

    # ── 4. Define a radiation schedule (one entry per generation) ────
    radiation_schedule = [
        # Gen 1 — gentle knowledge radiation
        [{"type": "knowledge", "data": ["new_technique_A", "paper_B"], "intensity": 0.08}],
        # Gen 2 — contradiction to force new reasoning
        [{"type": "contradiction", "data": ["claim_X_vs_Y"],           "intensity": 0.15}],
        # Gen 3 — adversarial stress test
        [{"type": "adversarial",   "data": ["edge_case_1", "attack_2"],"intensity": 0.20}],
        # Gen 4 — cross-domain emergence
        [{"type": "emergence",     "data": ["novel_problem_set"],       "intensity": 0.25}],
        # Gen 5 — entropy shake to escape local optima
        [{"type": "entropy",       "data": [],                          "intensity": 0.12}],
    ]

    # ── 5. Run 5 generations ──────────────────────────────────────────
    results = pipeline.run_n_generations(
        n=5,
        radiation_schedule=radiation_schedule,
        target_forms=["research", "engineering"],
        n_mutants=8,
    )

    # ── 6. Register lineage and print reports ────────────────────────
    tracker = LineageTracker("quickstart")
    tracker.register_all(pipeline.lineage)
    tracker.print_tree()

    print_report(pipeline.lineage)

    best = pipeline.best_state()
    print(f"Champion : {best}")
    print(f"Best path length : {len(tracker.best_path())} generations\n")

    print("Done. Check ./retbs_logs/quickstart for per-generation JSON snapshots.")


if __name__ == "__main__":
    main()
