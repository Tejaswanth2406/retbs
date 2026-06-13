"""
RETBSConfig — centralised experiment configuration.

Load from YAML / JSON, validate, and pass to the pipeline.
All hyperparameters for a RETBS experiment live here so runs are
fully reproducible from a single config file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RadiationConfig:
    enabled: bool = True
    types: List[str] = field(default_factory=lambda: ["knowledge", "contradiction", "adversarial"])
    base_intensity: float = 0.1
    intensity_schedule: str = "constant"   # "constant" | "increasing" | "cosine"
    max_intensity: float = 0.4


@dataclass
class EvolutionConfig:
    n_mutants: int = 10
    survival_rate: float = 0.2
    min_identity_stability: float = 0.85
    max_generations: int = 20
    early_stop_patience: int = 5


@dataclass
class BlendConfig:
    method: str = "adaptive"              # "weighted_average" | "task_vector" | "best_of_zone" | "adaptive"


@dataclass
class SustainConfig:
    stabilisation_strength: float = 0.1
    max_drift_threshold: float = 0.4


@dataclass
class TrainingConfig:
    """Fine-tuning hyperparameters (passed to Axolotl / TRL)."""
    base_model: str = "meta-llama/Llama-3-8B"
    fine_tune_method: str = "qlora"         # "full" | "lora" | "qlora"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    output_dir: str = "./retbs_output"
    save_steps: int = 100
    logging_steps: int = 10


@dataclass
class RETBSConfig:
    """Master configuration for a RETBS experiment."""

    # Experiment identity
    experiment_name: str = "retbs_experiment_v1"
    seed: int = 42
    log_dir: str = "./retbs_logs"

    # Component configs
    radiation:  RadiationConfig  = field(default_factory=RadiationConfig)
    evolution:  EvolutionConfig  = field(default_factory=EvolutionConfig)
    blend:      BlendConfig      = field(default_factory=BlendConfig)
    sustain:    SustainConfig    = field(default_factory=SustainConfig)
    training:   TrainingConfig   = field(default_factory=TrainingConfig)

    # Cognitive forms to train
    target_forms: List[str] = field(default_factory=lambda: ["research", "engineering", "creative"])

    # Knowledge graph settings
    knowledge_graph_path: Optional[str] = None

    # Dataset settings
    base_dataset_path: Optional[str] = None
    dataset_output_dir: str = "./retbs_datasets"
    n_chameleon_samples_per_form: int = 50
    n_radiation_samples: int = 100

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        print(f"[RETBSConfig] Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RETBSConfig":
        raw = json.loads(Path(path).read_text())
        cfg = cls(
            experiment_name=raw.get("experiment_name", "retbs_experiment"),
            seed=raw.get("seed", 42),
            log_dir=raw.get("log_dir", "./retbs_logs"),
            radiation=RadiationConfig(**raw.get("radiation", {})),
            evolution=EvolutionConfig(**raw.get("evolution", {})),
            blend=BlendConfig(**raw.get("blend", {})),
            sustain=SustainConfig(**raw.get("sustain", {})),
            training=TrainingConfig(**raw.get("training", {})),
            target_forms=raw.get("target_forms", ["research", "engineering", "creative"]),
            knowledge_graph_path=raw.get("knowledge_graph_path"),
            base_dataset_path=raw.get("base_dataset_path"),
            dataset_output_dir=raw.get("dataset_output_dir", "./retbs_datasets"),
            n_chameleon_samples_per_form=raw.get("n_chameleon_samples_per_form", 50),
            n_radiation_samples=raw.get("n_radiation_samples", 100),
        )
        print(f"[RETBSConfig] Loaded from {path}")
        return cfg

    def radiation_intensity(self, generation: int) -> float:
        """Compute radiation intensity for a given generation using the schedule."""
        base = self.radiation.base_intensity
        max_i = self.radiation.max_intensity
        n = max(1, self.evolution.max_generations)
        schedule = self.radiation.intensity_schedule

        if schedule == "increasing":
            return base + (max_i - base) * (generation / n)
        elif schedule == "cosine":
            import math
            return base + (max_i - base) * 0.5 * (1 - math.cos(math.pi * generation / n))
        else:
            return base  # constant

    def __repr__(self) -> str:
        return (f"RETBSConfig(name={self.experiment_name!r}, "
                f"model={self.training.base_model!r}, "
                f"max_gen={self.evolution.max_generations})")


def default_config() -> RETBSConfig:
    return RETBSConfig()
