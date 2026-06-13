"""
DatasetBuilder — constructs RETBS-style fine-tuning datasets.

Organises training data around the atomic knowledge hierarchy and
injects radiation samples to drive evolutionary fine-tuning.

Output format is JSONL: one JSON object per line, compatible with
Axolotl, TRL, and most HuggingFace fine-tuning pipelines.

Each record has:
  {
    "instruction": str,
    "input": str,          (optional context)
    "output": str,
    "metadata": {
        "source": str,     ("base" | "radiation_*" | "chameleon")
        "form": str,       (cognitive form tag)
        "domain": str,
        "atom_ids": [...], (K-Atom concepts exercised by this sample)
        "generation": int
    }
  }
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from retbs.knowledge.atomic_knowledge import KAtom, KnowledgeGenomeGraph


@dataclass
class TrainingSample:
    instruction: str
    output: str
    input: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
        }


class DatasetBuilder:
    """
    Builds and augments RETBS training datasets.

    Usage:
        builder = DatasetBuilder(seed=42)
        builder.add_base_samples(my_samples)
        builder.add_chameleon_samples(["research", "engineering", "creative"])
        builder.add_radiation_samples(graph, radiation_type="knowledge")
        builder.save("dataset_gen1.jsonl")
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._samples: List[TrainingSample] = []

    # ------------------------------------------------------------------
    # Base samples
    # ------------------------------------------------------------------

    def add_base_samples(self, samples: List[Dict[str, Any]], generation: int = 0) -> None:
        """Add pre-existing training samples (already formatted as dicts)."""
        for s in samples:
            self._samples.append(TrainingSample(
                instruction=s.get("instruction", ""),
                input=s.get("input", ""),
                output=s.get("output", ""),
                metadata={**s.get("metadata", {}), "source": "base", "generation": generation},
            ))
        print(f"[DatasetBuilder] Added {len(samples)} base samples.")

    # ------------------------------------------------------------------
    # Chameleon form samples
    # ------------------------------------------------------------------

    def add_chameleon_samples(
        self,
        forms: List[str],
        base_instruction: str = "Respond to the following:",
        n_per_form: int = 10,
        generation: int = 0,
    ) -> None:
        """
        Generate synthetic chameleon samples that teach the model to
        activate different cognitive forms based on context.

        Example output:
            instruction: "Environment = Cybersecurity. Analyse the threat vector."
            output:      "[Engineering Form] ..."
        """
        form_prefixes = {
            "research":    "You are in Research Form. Provide deep analytical reasoning:",
            "engineering": "You are in Engineering Form. Give a precise, structured solution:",
            "scientific":  "You are in Scientific Form. Apply empirical reasoning:",
            "creative":    "You are in Creative Form. Generate a novel, inventive response:",
            "strategic":   "You are in Strategic Form. Plan and evaluate options systematically:",
            "educator":    "You are in Educator Form. Explain clearly with analogies:",
            "general":     "You are in General Form. Balance depth and clarity:",
        }

        form_contexts = {
            "research":    ["academic paper review", "literature synthesis", "hypothesis generation"],
            "engineering": ["system design", "code debugging", "architecture decision"],
            "scientific":  ["experiment design", "data analysis", "theory evaluation"],
            "creative":    ["story generation", "ideation session", "metaphor creation"],
            "strategic":   ["risk assessment", "roadmap planning", "competitive analysis"],
            "educator":    ["concept explanation", "tutorial creation", "Q&A session"],
            "general":     ["general question", "mixed task", "open-ended query"],
        }

        for form in forms:
            prefix  = form_prefixes.get(form, f"You are in {form.title()} Form.")
            contexts = form_contexts.get(form, ["task"])
            for _ in range(n_per_form):
                context = self._rng.choice(contexts)
                sample = TrainingSample(
                    instruction=f"Environment = {context.replace('_', ' ').title()}.\n{base_instruction}",
                    input="",
                    output=prefix,
                    metadata={
                        "source": "chameleon",
                        "form": form,
                        "domain": "meta",
                        "generation": generation,
                    },
                )
                self._samples.append(sample)

        print(f"[DatasetBuilder] Added chameleon samples for forms: {forms}")

    # ------------------------------------------------------------------
    # Radiation-augmented samples
    # ------------------------------------------------------------------

    def add_radiation_samples(
        self,
        graph: KnowledgeGenomeGraph,
        radiation_type: str = "knowledge",
        n_samples: int = 20,
        intensity: float = 0.2,
        generation: int = 0,
    ) -> None:
        """
        Generate radiation samples from the knowledge graph.

        Targets weak atoms (low strength) with radiation prompts to
        create training pressure on under-developed capabilities.
        """
        weak = graph.weak_atoms(threshold=0.5)
        if not weak:
            weak = list(graph._atoms.values())

        templates = {
            "knowledge": [
                "Explain the concept of {concept} and how it relates to {dep}.",
                "Provide a detailed technical overview of {concept} in {domain}.",
                "What are the key properties and use-cases of {concept}?",
            ],
            "contradiction": [
                "Reconcile the following apparent contradiction: {concept} and {dep} seem incompatible. Explain.",
                "Given conflicting information about {concept}, how would you resolve the ambiguity?",
            ],
            "adversarial": [
                "Identify the flaw in the following claim about {concept}: [intentionally incorrect statement].",
                "What are the edge cases and failure modes when using {concept}?",
            ],
            "emergence": [
                "How might {concept} from {domain} apply to a completely different field?",
                "Design a novel use of {concept} that has never been attempted before.",
            ],
            "environmental": [
                "How has the role of {concept} in {domain} changed over the past 5 years?",
                "What new requirements or constraints affect {concept} today?",
            ],
        }

        tmpl_list = templates.get(radiation_type, templates["knowledge"])
        generated = 0

        for atom in self._rng.choices(weak, k=n_samples):
            tmpl = self._rng.choice(tmpl_list)
            dep  = atom.dependencies[0] if atom.dependencies else atom.domain
            instruction = tmpl.format(concept=atom.concept, dep=dep, domain=atom.domain)
            self._samples.append(TrainingSample(
                instruction=instruction,
                output=f"[Radiation-{radiation_type.title()} response about '{atom.concept}']",
                metadata={
                    "source": f"radiation_{radiation_type}",
                    "atom_ids": [atom.concept],
                    "domain": atom.domain,
                    "intensity": intensity,
                    "generation": generation,
                },
            ))
            generated += 1

        print(f"[DatasetBuilder] Added {generated} radiation '{radiation_type}' samples.")

    # ------------------------------------------------------------------
    # Quantum-inspired multi-state samples
    # ------------------------------------------------------------------

    def add_quantum_samples(
        self,
        n_samples: int = 10,
        generation: int = 0,
    ) -> None:
        """
        Add samples that train the model to reason across multiple
        competing state hypotheses before committing to an answer.
        """
        for i in range(n_samples):
            self._samples.append(TrainingSample(
                instruction=(
                    "Before answering, enumerate multiple possible interpretations "
                    f"(State A, State B, State C) and then collapse to the strongest one. "
                    f"Task: [Problem {i+1}]"
                ),
                output=(
                    "State A (40%): ...\n"
                    "State B (35%): ...\n"
                    "State C (25%): ...\n\n"
                    "Collapse → State A: [final answer]"
                ),
                metadata={"source": "quantum_inspired", "generation": generation},
            ))
        print(f"[DatasetBuilder] Added {n_samples} quantum-inspired samples.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path, shuffle: bool = True) -> int:
        """Write the dataset to a JSONL file. Returns number of records written."""
        samples = list(self._samples)
        if shuffle:
            self._rng.shuffle(samples)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")
        print(f"[DatasetBuilder] Saved {len(samples)} samples → {path}")
        return len(samples)

    def stats(self) -> Dict[str, Any]:
        from collections import Counter
        sources = Counter(s.metadata.get("source", "unknown") for s in self._samples)
        forms   = Counter(s.metadata.get("form", "") for s in self._samples if s.metadata.get("form"))
        return {
            "total_samples": len(self._samples),
            "by_source": dict(sources),
            "by_form": dict(forms),
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return f"DatasetBuilder(samples={len(self._samples)})"
