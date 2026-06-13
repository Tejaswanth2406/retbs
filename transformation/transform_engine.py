"""
TransformEngine — morphological intelligence.

The chameleon principle: the model physically shifts its internal activation
pathways to match the shape of the task, without changing its identity.

Forms (learned latent modes):
  general       — balanced default
  research      — deep analytical reasoning, hypothesis generation
  engineering   — systematic problem decomposition, code, architecture
  scientific    — empirical reasoning, experiment design, evidence synthesis
  creative      — divergent thinking, narrative, ideation
  strategic     — planning, risk assessment, multi-step foresight
  educator      — explanation, analogies, progressive disclosure

Shape Theory:
  Shape(Task) == Shape(Intelligence) → maximum alignment → best output

Each form adjusts genome zone activation weights. In a real fine-tuning
scenario, forms correspond to task-conditioned LoRA adapters or expert
routing tables. Here we represent them as zone weight multipliers that
modify how each genome zone contributes to the final state's performance.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from retbs.core.kernel import IdentityKernel
from retbs.core.intelligence_state import IntelligenceState


# ─────────────────────────────────────────────────────────────────────────────
# Form definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CognitivForm:
    """
    A named morphological mode with zone activation multipliers.

    multipliers > 1.0 → zone is amplified in this form
    multipliers < 1.0 → zone is dampened
    multipliers = 1.0 → no change
    """
    name: str
    description: str
    zone_multipliers: Dict[str, float]
    prompt_prefix: str = ""      # System-prompt prefix for fine-tuned models
    adapter_tag: str = ""        # LoRA adapter tag (used in real training pipelines)


# Built-in form library
FORM_LIBRARY: Dict[str, CognitivForm] = {
    "general": CognitivForm(
        name="general",
        description="Balanced default mode — no zone amplification.",
        zone_multipliers={"reasoning": 1.0, "creativity": 1.0, "domain": 1.0,
                          "planning": 1.0, "memory": 1.0},
        prompt_prefix="You are a highly capable, balanced intelligence.",
        adapter_tag="general",
    ),
    "research": CognitivForm(
        name="research",
        description="Deep analytical reasoning, hypothesis generation, literature synthesis.",
        zone_multipliers={"reasoning": 1.4, "memory": 1.3, "creativity": 1.1,
                          "planning": 1.0, "domain": 1.2},
        prompt_prefix="You are in Research Form. Prioritise rigorous analysis, hypothesis generation, and evidence synthesis.",
        adapter_tag="research",
    ),
    "engineering": CognitivForm(
        name="engineering",
        description="Systematic problem decomposition, code, system architecture.",
        zone_multipliers={"reasoning": 1.3, "planning": 1.4, "domain": 1.3,
                          "creativity": 0.9, "memory": 1.1},
        prompt_prefix="You are in Engineering Form. Prioritise precision, systems thinking, and implementation correctness.",
        adapter_tag="engineering",
    ),
    "scientific": CognitivForm(
        name="scientific",
        description="Empirical reasoning, experiment design, evidence evaluation.",
        zone_multipliers={"reasoning": 1.4, "domain": 1.4, "memory": 1.2,
                          "creativity": 1.0, "planning": 1.1},
        prompt_prefix="You are in Scientific Form. Prioritise empirical accuracy, experimental rigour, and uncertainty quantification.",
        adapter_tag="scientific",
    ),
    "creative": CognitivForm(
        name="creative",
        description="Divergent thinking, narrative construction, novel ideation.",
        zone_multipliers={"creativity": 1.5, "reasoning": 0.9, "domain": 1.0,
                          "planning": 0.9, "memory": 1.1},
        prompt_prefix="You are in Creative Form. Prioritise originality, metaphor, and novel synthesis.",
        adapter_tag="creative",
    ),
    "strategic": CognitivForm(
        name="strategic",
        description="Long-horizon planning, risk modelling, decision frameworks.",
        zone_multipliers={"planning": 1.5, "reasoning": 1.2, "domain": 1.1,
                          "creativity": 1.0, "memory": 1.2},
        prompt_prefix="You are in Strategic Form. Prioritise multi-step planning, risk assessment, and outcome modelling.",
        adapter_tag="strategic",
    ),
    "educator": CognitivForm(
        name="educator",
        description="Explanation, analogy, progressive concept disclosure.",
        zone_multipliers={"reasoning": 1.1, "memory": 1.3, "creativity": 1.2,
                          "planning": 1.1, "domain": 1.2},
        prompt_prefix="You are in Educator Form. Prioritise clarity, analogy, and progressive scaffolding.",
        adapter_tag="educator",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Transform Engine
# ─────────────────────────────────────────────────────────────────────────────

class TransformEngine:
    """
    Applies morphological transformations to intelligence states.

    For each requested form, the engine adjusts genome zone effective strengths
    using the form's multipliers, then records the transformation.

    In a production fine-tuning pipeline this corresponds to:
      - Selecting the task-conditioned LoRA adapter
      - Switching expert routing weights
      - Adjusting the system prompt prefix
    """

    def __init__(
        self,
        kernel: IdentityKernel,
        custom_forms: Optional[Dict[str, CognitivForm]] = None,
    ) -> None:
        self.kernel = kernel
        self.forms: Dict[str, CognitivForm] = {**FORM_LIBRARY, **(custom_forms or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(
        self,
        state: IntelligenceState,
        forms: List[str],
    ) -> IntelligenceState:
        """
        Apply one or more forms to a state sequentially.

        If multiple forms are given, they are blended by averaging multipliers.
        Returns a new IntelligenceState — original is untouched.
        """
        resolved = self._resolve_forms(forms)
        blended_multipliers = self._blend_multipliers(resolved)

        new_state = copy.deepcopy(state)
        for zone_name, zone in new_state.genome.items():
            if zone_name == "identity":
                continue  # identity zone is never transformed
            mult = blended_multipliers.get(zone_name, 1.0)
            # Apply multiplier as a bounded strength adjustment
            new_strength = min(1.0, zone.current_strength * mult)
            from retbs.core.intelligence_state import GenomeZone
            new_state.genome[zone_name] = GenomeZone(
                name=zone.name,
                mutation_rate=zone.mutation_rate,
                current_strength=new_strength,
                stability=zone.stability,
            )

        form_names = [f.name for f in resolved]
        new_state.metadata["active_forms"] = form_names
        new_state.metadata["prompt_prefixes"] = [f.prompt_prefix for f in resolved]
        new_state.metadata["adapter_tags"] = [f.adapter_tag for f in resolved]

        print(f"  [TransformEngine] Applied forms: {form_names}")
        return new_state

    def get_system_prompt(self, forms: List[str]) -> str:
        """Return a combined system prompt prefix for the given forms."""
        resolved = self._resolve_forms(forms)
        prefixes = [f.prompt_prefix for f in resolved if f.prompt_prefix]
        return " ".join(prefixes)

    def register_form(self, form: CognitivForm) -> None:
        """Register a custom cognitive form."""
        self.forms[form.name] = form
        print(f"  [TransformEngine] Registered custom form: '{form.name}'")

    def available_forms(self) -> List[str]:
        return list(self.forms.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_forms(self, form_names: List[str]) -> List[CognitivForm]:
        resolved = []
        for name in form_names:
            if name in self.forms:
                resolved.append(self.forms[name])
            else:
                print(f"  [TransformEngine] Unknown form '{name}', falling back to 'general'.")
                resolved.append(self.forms["general"])
        return resolved or [self.forms["general"]]

    @staticmethod
    def _blend_multipliers(forms: List[CognitivForm]) -> Dict[str, float]:
        """Average the multipliers across all requested forms."""
        if not forms:
            return {}
        all_zones: set = set()
        for f in forms:
            all_zones.update(f.zone_multipliers.keys())
        blended = {}
        for zone in all_zones:
            vals = [f.zone_multipliers.get(zone, 1.0) for f in forms]
            blended[zone] = sum(vals) / len(vals)
        return blended

    def __repr__(self) -> str:
        return f"TransformEngine(forms={list(self.forms.keys())})"
