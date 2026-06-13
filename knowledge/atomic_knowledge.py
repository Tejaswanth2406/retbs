"""
Atomic Knowledge Model — hierarchical knowledge representation for RETBS.

Hierarchy (bottom-up):
  K-Atom     — smallest indivisible concept unit
  K-Molecule — connected atoms forming a functional concept
  K-Cell     — molecules forming a capability cluster
  K-Organ    — cells forming a major knowledge domain
  K-Organism — all organs together = the full intelligence

Each atom tracks:
  - Concept name and description
  - Dependencies on other atoms
  - Confidence (0–1): how well the model knows this concept
  - Utility (0–1): how often this atom is activated
  - Evolution history: which generation introduced / mutated this atom

The genome graph connects atoms via dependency edges, enabling:
  - Targeted radiation (perturb a specific sub-graph)
  - Atomic mutation scheduling (mutate weak atoms, preserve strong ones)
  - Knowledge decay detection (low-utility atoms can be pruned)
  - Cross-domain emergence (connect atoms across domains)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy levels
# ─────────────────────────────────────────────────────────────────────────────

class KLevel(str, Enum):
    ATOM     = "atom"
    MOLECULE = "molecule"
    CELL     = "cell"
    ORGAN    = "organ"
    ORGANISM = "organism"


# ─────────────────────────────────────────────────────────────────────────────
# K-Atom
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KAtom:
    """
    The smallest indivisible knowledge unit.

    Example:
        KAtom(concept="recursion", domain="computer_science",
              dependencies=["function", "stack", "base_case"])
    """
    concept: str
    domain: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)   # concept names this atom depends on
    confidence: float = 0.5     # 0 = unknown, 1 = mastered
    utility: float = 0.5        # 0 = never activated, 1 = constantly used
    mutation_rate: float = 0.1  # how freely this atom can be mutated
    level: KLevel = KLevel.ATOM
    atom_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    evolution_history: List[Dict] = field(default_factory=list)

    def update(self, confidence_delta: float = 0.0, utility_delta: float = 0.0, generation: int = 0) -> None:
        """Apply learning signal to this atom."""
        self.confidence = max(0.0, min(1.0, self.confidence + confidence_delta))
        self.utility    = max(0.0, min(1.0, self.utility    + utility_delta))
        self.evolution_history.append({
            "generation": generation,
            "confidence": self.confidence,
            "utility": self.utility,
            "timestamp": time.time(),
        })

    @property
    def strength(self) -> float:
        """Combined measure: confidence * utility."""
        return self.confidence * self.utility

    def to_dict(self) -> dict:
        return {
            "atom_id": self.atom_id, "concept": self.concept, "domain": self.domain,
            "description": self.description, "dependencies": self.dependencies,
            "confidence": self.confidence, "utility": self.utility,
            "mutation_rate": self.mutation_rate, "level": self.level,
            "evolution_history": self.evolution_history,
        }

    def __repr__(self) -> str:
        return f"KAtom({self.concept!r}, conf={self.confidence:.2f}, util={self.utility:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# K-Molecule (connected atoms)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KMolecule:
    """A group of related K-Atoms forming a functional concept."""
    name: str
    atoms: List[KAtom] = field(default_factory=list)
    level: KLevel = KLevel.MOLECULE
    mol_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    @property
    def strength(self) -> float:
        if not self.atoms:
            return 0.0
        return sum(a.strength for a in self.atoms) / len(self.atoms)

    def add_atom(self, atom: KAtom) -> None:
        self.atoms.append(atom)

    def weak_atoms(self, threshold: float = 0.3) -> List[KAtom]:
        return [a for a in self.atoms if a.strength < threshold]

    def __repr__(self) -> str:
        return f"KMolecule({self.name!r}, atoms={len(self.atoms)}, str={self.strength:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# K-Cell (molecules forming a capability cluster)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KCell:
    """A capability cluster composed of K-Molecules."""
    name: str
    molecules: List[KMolecule] = field(default_factory=list)
    level: KLevel = KLevel.CELL

    @property
    def strength(self) -> float:
        if not self.molecules:
            return 0.0
        return sum(m.strength for m in self.molecules) / len(self.molecules)

    def add_molecule(self, mol: KMolecule) -> None:
        self.molecules.append(mol)

    def __repr__(self) -> str:
        return f"KCell({self.name!r}, molecules={len(self.molecules)}, str={self.strength:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# K-Organ (cells forming a major domain)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KOrgan:
    """A major knowledge domain composed of K-Cells."""
    name: str
    cells: List[KCell] = field(default_factory=list)
    level: KLevel = KLevel.ORGAN

    @property
    def strength(self) -> float:
        if not self.cells:
            return 0.0
        return sum(c.strength for c in self.cells) / len(self.cells)

    def add_cell(self, cell: KCell) -> None:
        self.cells.append(cell)

    def __repr__(self) -> str:
        return f"KOrgan({self.name!r}, cells={len(self.cells)}, str={self.strength:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Genome Graph
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeGenomeGraph:
    """
    A directed graph of K-Atoms connected by dependency edges.

    Node  = KAtom  (concept)
    Edge  = dependency relationship

    This graph is the RETBS representation of the model's knowledge structure.
    Mutations in the RETBS cycle operate on this graph — perturbing nodes,
    adding new edges (cross-domain fusion), or pruning dead atoms.

    Usage:
        graph = KnowledgeGenomeGraph()
        graph.add_atom(KAtom("recursion", "cs"))
        graph.add_atom(KAtom("tree", "cs", dependencies=["recursion"]))
        graph.add_edge("recursion", "tree")
        subgraph = graph.subgraph("cs")
    """

    def __init__(self) -> None:
        self._atoms: Dict[str, KAtom] = {}       # concept → KAtom
        self._edges: Dict[str, Set[str]] = {}    # concept → {dependent concepts}

    # ------------------------------------------------------------------
    # Mutation / construction
    # ------------------------------------------------------------------

    def add_atom(self, atom: KAtom) -> None:
        self._atoms[atom.concept] = atom
        if atom.concept not in self._edges:
            self._edges[atom.concept] = set()
        for dep in atom.dependencies:
            self.add_edge(dep, atom.concept)

    def add_edge(self, source: str, target: str) -> None:
        if source not in self._edges:
            self._edges[source] = set()
        self._edges[source].add(target)

    def remove_atom(self, concept: str) -> None:
        self._atoms.pop(concept, None)
        self._edges.pop(concept, None)
        for deps in self._edges.values():
            deps.discard(concept)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, concept: str) -> Optional[KAtom]:
        return self._atoms.get(concept)

    def dependents(self, concept: str) -> List[str]:
        """Return all concepts that depend on this one."""
        return list(self._edges.get(concept, set()))

    def dependencies_of(self, concept: str) -> List[str]:
        """Return direct dependency concepts for this atom."""
        atom = self._atoms.get(concept)
        return atom.dependencies if atom else []

    def subgraph(self, domain: str) -> "KnowledgeGenomeGraph":
        """Return a sub-graph containing only atoms from a specific domain."""
        sub = KnowledgeGenomeGraph()
        for atom in self._atoms.values():
            if atom.domain == domain:
                sub.add_atom(atom)
        return sub

    def weak_atoms(self, threshold: float = 0.3) -> List[KAtom]:
        """Return atoms below a strength threshold — candidates for radiation."""
        return [a for a in self._atoms.values() if a.strength < threshold]

    def strong_atoms(self, threshold: float = 0.7) -> List[KAtom]:
        return [a for a in self._atoms.values() if a.strength >= threshold]

    def domains(self) -> List[str]:
        return list({a.domain for a in self._atoms.values()})

    def stats(self) -> Dict:
        atoms = list(self._atoms.values())
        return {
            "n_atoms": len(atoms),
            "n_edges": sum(len(v) for v in self._edges.values()),
            "domains": self.domains(),
            "mean_confidence": sum(a.confidence for a in atoms) / len(atoms) if atoms else 0,
            "mean_utility":    sum(a.utility    for a in atoms) / len(atoms) if atoms else 0,
            "mean_strength":   sum(a.strength   for a in atoms) / len(atoms) if atoms else 0,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        data = {
            "atoms": {k: v.to_dict() for k, v in self._atoms.items()},
            "edges": {k: list(v) for k, v in self._edges.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "KnowledgeGenomeGraph":
        data = json.loads(Path(path).read_text())
        graph = cls()
        for concept, atom_dict in data.get("atoms", {}).items():
            atom_dict.pop("atom_id", None)
            level = atom_dict.pop("level", KLevel.ATOM)
            atom  = KAtom(**atom_dict)
            atom.level = level
            graph._atoms[concept] = atom
        graph._edges = {k: set(v) for k, v in data.get("edges", {}).items()}
        return graph

    def __len__(self) -> int:
        return len(self._atoms)

    def __repr__(self) -> str:
        return f"KnowledgeGenomeGraph(atoms={len(self._atoms)}, edges={sum(len(v) for v in self._edges.values())})"


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build a starter CS knowledge graph
# ─────────────────────────────────────────────────────────────────────────────

def build_cs_starter_graph() -> KnowledgeGenomeGraph:
    """A minimal computer-science knowledge graph to bootstrap RETBS."""
    graph = KnowledgeGenomeGraph()

    atoms = [
        KAtom("variable",    "cs", "A named storage location.", confidence=0.9, utility=0.9),
        KAtom("function",    "cs", "A reusable code block.",    confidence=0.85, utility=0.85, dependencies=["variable"]),
        KAtom("loop",        "cs", "Iteration construct.",      confidence=0.85, utility=0.8,  dependencies=["variable"]),
        KAtom("recursion",   "cs", "Self-referential function.",confidence=0.7,  utility=0.7,  dependencies=["function", "loop"]),
        KAtom("array",       "cs", "Ordered sequence.",         confidence=0.85, utility=0.85, dependencies=["variable"]),
        KAtom("stack",       "cs", "LIFO data structure.",      confidence=0.75, utility=0.7,  dependencies=["array"]),
        KAtom("tree",        "cs", "Hierarchical data struct.", confidence=0.7,  utility=0.75, dependencies=["recursion", "stack"]),
        KAtom("algorithm",   "cs", "Step-by-step procedure.",   confidence=0.8,  utility=0.8,  dependencies=["loop", "recursion"]),
        KAtom("complexity",  "cs", "Algorithm efficiency.",     confidence=0.65, utility=0.7,  dependencies=["algorithm"]),
        KAtom("oop",         "cs", "Object-oriented paradigm.", confidence=0.8,  utility=0.8,  dependencies=["function", "variable"]),
        KAtom("api",         "cs", "Application interface.",    confidence=0.8,  utility=0.85, dependencies=["function"]),
        KAtom("concurrency", "cs", "Parallel execution.",       confidence=0.6,  utility=0.65, dependencies=["function", "algorithm"]),
    ]

    for atom in atoms:
        graph.add_atom(atom)

    return graph
