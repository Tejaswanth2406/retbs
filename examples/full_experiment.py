"""
full_experiment.py — Config-driven RETBS experiment with lineage export.

Demonstrates the complete production workflow:
  1. Load / create a RETBSConfig
  2. Build an initial dataset
  3. Run N generations of R→E→T→B→S
  4. Track and export the full lineage
  5. Print metrics report

Run:
    python -m retbs.examples.full_experiment
    python -m retbs.examples.full_experiment --config configs/my_config.json
"""

import argparse
import json
from pathlib import Path

from retbs.core.kernel import IdentityKernel, SafetyConstraint, EvolutionPolicy
from retbs.core.intelligence_state import IntelligenceState
from retbs.core.retbs_pipeline import RETBSPipeline
from retbs.knowledge.atomic_knowledge import build_cs_starter_graph
from retbs.knowledge.dataset_builder import DatasetBuilder
from retbs.utils.lineage_tracker import LineageTracker
from retbs.utils.config import RETBSConfig, default_config
from retbs.utils.metrics import print_report


def build_kernel(cfg: RETBSConfig) -> IdentityKernel:
    policy = EvolutionPolicy(
        max_mutation_rate=0.3,
        identity_zone_rate=0.0,
        reasoning_zone_rate=0.05,
        creativity_zone_rate=0.3,
        domain_zone_rate=0.2,
        min_identity_stability=cfg.evolution.min_identity_stability,
        max_generations_without_gain=cfg.evolution.early_stop_patience,
    )
    return IdentityKernel(
        mission="Be a continuously improving, safe, and morphologically adaptive intelligence.",
        values=["helpfulness", "honesty", "safety", "consistency", "adaptability"],
        safety_constraints=[
            SafetyConstraint("no_harm",    "Never assist with harmful activities.", "critical"),
            SafetyConstraint("factuality", "Prioritise factual accuracy.",           "warning"),
            SafetyConstraint("coherence",  "Maintain logical consistency.",          "warning"),
        ],
        evolution_policy=policy,
    )


def build_radiation_schedule(cfg: RETBSConfig) -> list:
    """Build a radiation schedule for each generation."""
    schedule = []
    radiation_types = cfg.radiation.types
    for gen in range(cfg.evolution.max_generations):
        intensity = cfg.radiation_intensity(gen)
        rad_type  = radiation_types[gen % len(radiation_types)]
        schedule.append([{
            "type": rad_type,
            "data": [f"gen{gen+1}_radiation_sample_{i}" for i in range(3)],
            "intensity": intensity,
        }])
    return schedule


def run(cfg: RETBSConfig) -> None:
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  RETBS Experiment: {cfg.experiment_name[:32]:<32} ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── 1. Build knowledge graph + dataset ────────────────────────────
    print("[Step 1] Building knowledge graph and dataset...")
    graph   = build_cs_starter_graph()
    builder = DatasetBuilder(seed=cfg.seed)
    builder.add_chameleon_samples(cfg.target_forms, n_per_form=cfg.n_chameleon_samples_per_form)
    for rtype in cfg.radiation.types:
        builder.add_radiation_samples(graph, radiation_type=rtype,
                                      n_samples=cfg.n_radiation_samples // len(cfg.radiation.types))
    builder.add_quantum_samples(n_samples=20)

    out_dir = Path(cfg.dataset_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = builder.save(out_dir / "gen0_seed.jsonl")
    print(f"  Seed dataset: {n} samples → {out_dir / 'gen0_seed.jsonl'}\n")

    # ── 2. Kernel + initial state ──────────────────────────────────────
    print("[Step 2] Initialising kernel and pipeline...")
    kernel  = build_kernel(cfg)
    initial = IntelligenceState(
        generation=0,
        genome=IntelligenceState.default_genome(),
        metadata={"experiment": cfg.experiment_name, "config": cfg.to_dict()},
    )

    pipeline = RETBSPipeline(
        kernel=kernel,
        initial_state=initial,
        log_dir=cfg.log_dir,
    )

    # ── 3. Run generations ────────────────────────────────────────────
    print(f"[Step 3] Running {cfg.evolution.max_generations} generations...\n")
    schedule = build_radiation_schedule(cfg)
    pipeline.run_n_generations(
        n=cfg.evolution.max_generations,
        radiation_schedule=schedule,
        target_forms=cfg.target_forms,
        n_mutants=cfg.evolution.n_mutants,
    )

    # ── 4. Track lineage ──────────────────────────────────────────────
    print("\n[Step 4] Building lineage tracker...")
    tracker = LineageTracker(cfg.experiment_name)
    tracker.register_all(pipeline.lineage)
    tracker.print_tree()

    lineage_path = Path(cfg.log_dir) / "lineage.json"
    tracker.save(lineage_path)

    # ── 5. Report ─────────────────────────────────────────────────────
    print("[Step 5] Metrics report:")
    print_report(pipeline.lineage)

    best = pipeline.best_state()
    print(f"Champion state : {best}")
    print(f"Saved lineage  : {lineage_path}\n")

    # Save config used
    cfg_path = Path(cfg.log_dir) / "config_used.json"
    cfg.save(cfg_path)
    print(f"Config saved   : {cfg_path}")
    print("\nExperiment complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RETBS experiment.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a RETBSConfig JSON file.")
    args = parser.parse_args()

    if args.config:
        cfg = RETBSConfig.load(args.config)
    else:
        cfg = default_config()
        cfg.evolution.max_generations = 5
        cfg.evolution.n_mutants = 6
        cfg.experiment_name = "retbs_full_demo"

    run(cfg)


if __name__ == "__main__":
    main()
