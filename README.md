<img width="1983" height="793" alt="image" src="https://github.com/user-attachments/assets/0f7d333c-9a20-45b7-b4e5-61ce603651f5" />

# RETBS — Radiate → Evolve → Transform → Blend → Sustain

**A production-grade evolutionary fine-tuning framework that treats language models as living intelligence lineages.**

---

## Core Idea

Most fine-tuning today is a single shot:

```
Base Model → Fine-Tune → Final Model
```

RETBS treats a model as an **evolving species**:

```
Base Model → Radiation → Mutation States → Evolution → Transformation → Blending → Sustainment → Next Generation
```

Each generation is stronger than the last, while identity and safety constraints remain intact.

---

## Governing Equation

```
I_{t+1} = S( B( T( E( R(I_t) ) ) ) )
```

| Symbol | Layer | Purpose |
|--------|-------|---------|
| R | Radiate | Inject controlled perturbations (new knowledge, contradictions, adversarial inputs) |
| E | Evolve | Generate mutant candidates, evaluate, keep top performers |
| T | Transform | Apply morphological forms (Research, Engineering, Creative, …) |
| B | Blend | Merge survivors via weight interpolation, task vectors, or adaptive fusion |
| S | Sustain | Validate identity stability; roll back if drift exceeds threshold |

**Adaptive Coherence** is the primary fitness metric:

```
C = A + G + R + P - D
```

Where A = adaptability, G = generalization, R = retention, P = performance, D = drift.

---

## Project Structure

```
retbs/
├── core/
│   ├── kernel.py              # IdentityKernel — immutable mission/values/safety
│   ├── intelligence_state.py  # IntelligenceState — genome, scores, lineage
│   └── retbs_pipeline.py      # RETBSPipeline — full R→E→T→B→S orchestration
├── radiation/
│   └── radiation_engine.py    # Generates & applies perturbation packets
├── evolution/
│   └── evolution_arena.py     # Mutant generation, benchmark evaluation, selection
├── transformation/
│   └── transform_engine.py    # Morphological forms (chameleon principle)
├── blending/
│   └── blend_engine.py        # Weighted average, task vectors, adaptive merge
├── sustain/
│   └── sustain_engine.py      # Drift detection, identity guard, rollback
├── knowledge/
│   ├── atomic_knowledge.py    # K-Atom → Molecule → Cell → Organ hierarchy + genome graph
│   └── dataset_builder.py     # Build radiation-augmented JSONL training datasets
├── utils/
│   ├── lineage_tracker.py     # Genealogy DAG, best-path tracing, export
│   ├── metrics.py             # Coherence, gain, stability, radar profiles
│   └── config.py              # RETBSConfig — full experiment configuration
└── examples/
    ├── quickstart.py          # 5-generation demo (no GPU required)
    ├── build_dataset.py       # Dataset construction walkthrough
    └── full_experiment.py     # End-to-end experiment with config file
```

---

## Quickstart

```bash
git clone <repo>
cd retbs
pip install -r requirements.txt

python -m retbs.examples.quickstart
```

Expected output:

```
╔══════════════════════════════════════════════════════╗
║        RETBS — Quickstart (5 Generations)           ║
╚══════════════════════════════════════════════════════╝

[RETBS] Generation 1 — Start
  → Applying RadiationPacket(type=knowledge, intensity=0.08 [alpha], ...)
  [EvolutionArena] Generated 8 mutants ...
  [EvolutionArena] ✓ Survivor #1: a3f1bc20  C=2.1847
  ...
══════════════════════════════════════════
  RETBS LINEAGE METRICS REPORT
  Cumulative Gain      : +0.3142
  Identity Stability % : 100.0%
══════════════════════════════════════════
```

---

## Radiation Types

| Type | Purpose | Genome Zones Targeted |
|------|---------|----------------------|
| `knowledge` | New facts, papers, APIs | domain, memory |
| `contradiction` | Conflicting truths, paradoxes | reasoning, creativity |
| `adversarial` | Attacks, edge cases, failures | reasoning, domain |
| `environmental` | Market shifts, regulation changes | domain, planning |
| `emergence` | Novel unsolvable problems | creativity, planning, reasoning |
| `entropy` | Destabilise stagnation | creativity, domain, memory |

---

## Cognitive Forms (Morphological Intelligence)

The model learns to shift internal activation pathways per task — no prompt engineering needed.

| Form | Amplified Zones | Use Case |
|------|----------------|----------|
| `research` | reasoning×1.4, memory×1.3 | Literature synthesis, hypotheses |
| `engineering` | planning×1.4, domain×1.3 | System design, code, architecture |
| `scientific` | reasoning×1.4, domain×1.4 | Experiment design, data analysis |
| `creative` | creativity×1.5 | Ideation, narrative, invention |
| `strategic` | planning×1.5, reasoning×1.2 | Risk, roadmaps, decisions |
| `educator` | memory×1.3, creativity×1.2 | Explanation, analogies |

---

## Building a Training Dataset

```python
from retbs.knowledge.atomic_knowledge import build_cs_starter_graph
from retbs.knowledge.dataset_builder import DatasetBuilder

graph   = build_cs_starter_graph()
builder = DatasetBuilder(seed=42)

builder.add_chameleon_samples(["research", "engineering", "creative"], n_per_form=50)
builder.add_radiation_samples(graph, radiation_type="knowledge",     n_samples=100)
builder.add_radiation_samples(graph, radiation_type="contradiction",  n_samples=50)
builder.add_quantum_samples(n_samples=20)

n = builder.save("datasets/gen1.jsonl")
print(f"Saved {n} samples")
print(builder.stats())
```

---

## Configuration

```python
from retbs.utils.config import RETBSConfig, EvolutionConfig, RadiationConfig

cfg = RETBSConfig(
    experiment_name="retbs_v1",
    evolution=EvolutionConfig(n_mutants=20, max_generations=10),
    radiation=RadiationConfig(base_intensity=0.1, intensity_schedule="increasing"),
)
cfg.save("configs/experiment_v1.json")
```

---

## Research Contributions

This framework implements and makes testable:

1. **Radiation-Based Fine-Tuning** — controlled perturbation learning
2. **Evolutionary Model Lineages** — models as generations, not checkpoints
3. **Morphological Intelligence** — dynamic internal cognitive forms
4. **Identity Preservation Learning** — self-consistency through evolution
5. **Atomic Knowledge Evolution** — concept-level knowledge growth
6. **Adaptive Mutation Scheduling** — zone-specific mutation rates
7. **Dynamic Model Merging** — adaptive blending of survivor states
8. **Adaptive Coherence Maximisation** — fitness beyond static benchmarks

---

## The Strongest Research Hypothesis

> Intelligence is not a static neural network but a self-preserving evolutionary field composed of atomic knowledge structures that continuously radiate, transform, blend, and sustain themselves through successive generations of adaptation.

---

## License

MIT
