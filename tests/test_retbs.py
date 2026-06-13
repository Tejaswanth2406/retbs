"""
tests/test_retbs.py — Unit and integration tests for the RETBS framework.

Run:
    pytest retbs/tests/test_retbs.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Kernel
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityKernel:
    def setup_method(self):
        from retbs.core.kernel import default_kernel
        self.kernel = default_kernel()

    def test_fingerprint_stable(self):
        from retbs.core.kernel import default_kernel
        k1 = default_kernel()
        k2 = default_kernel()
        assert k1._fingerprint == k2._fingerprint

    def test_identity_zone_not_mutable(self):
        assert not self.kernel.validate_mutation("identity", 0.01)

    def test_creativity_zone_mutable(self):
        assert self.kernel.validate_mutation("creativity", 0.2)

    def test_creativity_zone_capped(self):
        assert not self.kernel.validate_mutation("creativity", 0.99)

    def test_save_load(self, tmp_path):
        p = tmp_path / "kernel.json"
        self.kernel.save(p)
        from retbs.core.kernel import IdentityKernel
        loaded = IdentityKernel.load(p)
        assert loaded.mission == self.kernel.mission
        assert loaded._fingerprint == self.kernel._fingerprint


# ─────────────────────────────────────────────────────────────────────────────
# IntelligenceState
# ─────────────────────────────────────────────────────────────────────────────

class TestIntelligenceState:
    def setup_method(self):
        from retbs.core.intelligence_state import IntelligenceState
        self.state = IntelligenceState(
            generation=0,
            genome=IntelligenceState.default_genome(),
        )

    def test_default_genome_has_identity(self):
        assert "identity" in self.state.genome

    def test_identity_zone_immutable(self):
        zone = self.state.genome["identity"]
        mutated = zone.apply_mutation(1.0)
        assert mutated is zone  # returns self unchanged

    def test_adaptive_coherence_positive(self):
        c = self.state.adaptive_coherence()
        assert c >= 0.0

    def test_identity_stability_is_one(self):
        assert self.state.identity_stability() == 1.0

    def test_save_load_roundtrip(self, tmp_path):
        from retbs.core.intelligence_state import IntelligenceState
        p = tmp_path / "state.json"
        self.state.save(p)
        loaded = IntelligenceState.load(p)
        assert loaded.state_id == self.state.state_id
        assert loaded.generation == 0


# ─────────────────────────────────────────────────────────────────────────────
# Radiation Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestRadiationEngine:
    def setup_method(self):
        from retbs.core.kernel import default_kernel
        from retbs.radiation.radiation_engine import RadiationEngine
        from retbs.core.intelligence_state import IntelligenceState
        self.kernel  = default_kernel()
        self.engine  = RadiationEngine(self.kernel, seed=0)
        self.state   = IntelligenceState(generation=0, genome=IntelligenceState.default_genome())

    def test_knowledge_packet_creation(self):
        pkt = self.engine.generate_packet("knowledge", ["data_a"], intensity=0.1)
        assert pkt.radiation_type == "knowledge"
        assert pkt.intensity == 0.1

    def test_identity_zone_blocked(self):
        pkt = self.engine.generate_packet("knowledge", ["x"], intensity=0.0,
                                          target_zones=["identity"])
        assert "identity" not in pkt.target_zones or pkt.intensity == 0.0

    def test_apply_returns_new_state(self):
        pkt = self.engine.default_packet()
        new_state = self.engine.apply(self.state, [pkt])
        assert new_state is not self.state

    def test_radiation_recorded(self):
        pkt = self.engine.default_packet()
        new_state = self.engine.apply(self.state, [pkt])
        assert len(new_state.radiation_history) == 1

    def test_intensity_class_labels(self):
        from retbs.radiation.radiation_engine import RadiationPacket
        p = RadiationPacket("knowledge", [], 0.05, [])
        assert p.intensity_class == "alpha"
        p2 = RadiationPacket("knowledge", [], 0.7, [])
        assert p2.intensity_class == "omega"


# ─────────────────────────────────────────────────────────────────────────────
# Evolution Arena
# ─────────────────────────────────────────────────────────────────────────────

class TestEvolutionArena:
    def setup_method(self):
        from retbs.core.kernel import default_kernel
        from retbs.evolution.evolution_arena import EvolutionArena
        from retbs.core.intelligence_state import IntelligenceState
        self.kernel = default_kernel()
        self.arena  = EvolutionArena(self.kernel, seed=0)
        self.parent = IntelligenceState(generation=1, genome=IntelligenceState.default_genome())

    def test_generate_mutants_count(self):
        mutants = self.arena.generate_mutants(self.parent, n=5)
        assert len(mutants) == 5

    def test_mutants_have_parent_id(self):
        mutants = self.arena.generate_mutants(self.parent, n=3)
        for m in mutants:
            assert self.parent.state_id in m.parent_ids

    def test_identity_zone_unchanged(self):
        mutants = self.arena.generate_mutants(self.parent, n=10)
        for m in mutants:
            assert m.genome["identity"].current_strength == \
                   self.parent.genome["identity"].current_strength

    def test_evaluate_and_select_returns_subset(self):
        mutants   = self.arena.generate_mutants(self.parent, n=10)
        survivors = self.arena.evaluate_and_select(mutants)
        assert 0 < len(survivors) <= len(mutants)

    def test_survivors_have_capability_scores(self):
        mutants   = self.arena.generate_mutants(self.parent, n=5)
        survivors = self.arena.evaluate_and_select(mutants)
        for s in survivors:
            assert len(s.capability_scores) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Transform Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformEngine:
    def setup_method(self):
        from retbs.core.kernel import default_kernel
        from retbs.transformation.transform_engine import TransformEngine
        from retbs.core.intelligence_state import IntelligenceState
        self.kernel = default_kernel()
        self.engine = TransformEngine(self.kernel)
        self.state  = IntelligenceState(generation=1, genome=IntelligenceState.default_genome())

    def test_all_default_forms_available(self):
        forms = self.engine.available_forms()
        for f in ["general", "research", "engineering", "scientific", "creative", "strategic", "educator"]:
            assert f in forms

    def test_transform_returns_new_state(self):
        new = self.engine.transform(self.state, ["research"])
        assert new is not self.state

    def test_identity_zone_preserved(self):
        new = self.engine.transform(self.state, ["creative"])
        assert new.genome["identity"].current_strength == \
               self.state.genome["identity"].current_strength

    def test_creativity_amplified_in_creative_form(self):
        new = self.engine.transform(self.state, ["creative"])
        assert new.genome["creativity"].current_strength >= \
               self.state.genome["creativity"].current_strength

    def test_multi_form_transform(self):
        new = self.engine.transform(self.state, ["research", "engineering"])
        assert new.metadata.get("active_forms") == ["research", "engineering"]

    def test_unknown_form_falls_back_to_general(self):
        new = self.engine.transform(self.state, ["nonexistent_form_xyz"])
        assert new is not None


# ─────────────────────────────────────────────────────────────────────────────
# Blend Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestBlendEngine:
    def setup_method(self):
        from retbs.core.kernel import default_kernel
        from retbs.blending.blend_engine import BlendEngine
        from retbs.core.intelligence_state import IntelligenceState
        self.kernel = default_kernel()
        self.engine = BlendEngine(self.kernel)

        def make_state(strength_offset=0.0):
            s = IntelligenceState(generation=1, genome=IntelligenceState.default_genome())
            for zone in s.genome.values():
                if zone.name != "identity":
                    zone.current_strength = min(1.0, zone.current_strength + strength_offset)
            return s

        self.states = [make_state(0.0), make_state(0.1), make_state(-0.1)]

    def test_blend_returns_single_state(self):
        result = self.engine.blend(self.states)
        assert result is not None

    def test_blend_single_state_no_op(self):
        result = self.engine.blend([self.states[0]])
        assert result.state_id == self.states[0].state_id

    def test_blend_has_all_parents(self):
        result = self.engine.blend(self.states)
        for s in self.states:
            assert s.state_id in result.parent_ids

    def test_identity_zone_from_best_parent(self):
        result = self.engine.blend(self.states)
        best_id_strength = self.states[0].genome["identity"].current_strength
        assert result.genome["identity"].current_strength == best_id_strength

    def test_all_blend_methods(self):
        from retbs.blending.blend_engine import BlendMethod
        for method in BlendMethod:
            result = self.engine.blend(self.states, method=method)
            assert result is not None


# ─────────────────────────────────────────────────────────────────────────────
# Sustain Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestSustainEngine:
    def setup_method(self):
        from retbs.core.kernel import default_kernel
        from retbs.sustain.sustain_engine import SustainEngine
        from retbs.core.intelligence_state import IntelligenceState
        self.kernel  = default_kernel()
        self.sustain = SustainEngine(self.kernel)
        self.prev    = IntelligenceState(generation=1, genome=IntelligenceState.default_genome())
        self.cand    = IntelligenceState(generation=2, genome=IntelligenceState.default_genome())

    def test_stable_candidate_accepted(self):
        result = self.sustain.stabilise(self.cand, self.prev, generation=2)
        report = self.sustain.reports[-1]
        assert report.action == "accepted"

    def test_collapsed_identity_triggers_hard_rollback(self):
        from retbs.core.intelligence_state import IntelligenceState, GenomeZone
        bad = IntelligenceState(generation=2, genome=IntelligenceState.default_genome())
        bad.genome["identity"] = GenomeZone("identity", 0.0, 1.0, stability=0.1)  # very unstable
        result = self.sustain.stabilise(bad, self.prev, generation=2)
        report = self.sustain.reports[-1]
        assert report.action in ("hard_rollback", "soft_rollback")

    def test_report_history_grows(self):
        self.sustain.stabilise(self.cand, self.prev, generation=2)
        self.sustain.stabilise(self.cand, self.prev, generation=3)
        assert len(self.sustain.reports) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Knowledge — Atomic Knowledge Model
# ─────────────────────────────────────────────────────────────────────────────

class TestAtomicKnowledge:
    def test_starter_graph_builds(self):
        from retbs.knowledge.atomic_knowledge import build_cs_starter_graph
        g = build_cs_starter_graph()
        assert len(g) > 0

    def test_atom_strength(self):
        from retbs.knowledge.atomic_knowledge import KAtom
        a = KAtom("loop", "cs", confidence=0.8, utility=0.5)
        assert abs(a.strength - 0.4) < 1e-9

    def test_graph_subgraph(self):
        from retbs.knowledge.atomic_knowledge import build_cs_starter_graph
        g = build_cs_starter_graph()
        sub = g.subgraph("cs")
        assert len(sub) > 0

    def test_graph_save_load(self, tmp_path):
        from retbs.knowledge.atomic_knowledge import build_cs_starter_graph, KnowledgeGenomeGraph
        g = build_cs_starter_graph()
        p = tmp_path / "graph.json"
        g.save(p)
        loaded = KnowledgeGenomeGraph.load(p)
        assert len(loaded) == len(g)

    def test_weak_atoms_detection(self):
        from retbs.knowledge.atomic_knowledge import build_cs_starter_graph
        g = build_cs_starter_graph()
        weak = g.weak_atoms(threshold=0.9)
        assert isinstance(weak, list)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetBuilder:
    def setup_method(self):
        from retbs.knowledge.dataset_builder import DatasetBuilder
        from retbs.knowledge.atomic_knowledge import build_cs_starter_graph
        self.builder = DatasetBuilder(seed=0)
        self.graph   = build_cs_starter_graph()

    def test_chameleon_samples_added(self):
        self.builder.add_chameleon_samples(["research", "creative"], n_per_form=5)
        assert len(self.builder) == 10

    def test_radiation_samples_added(self):
        self.builder.add_radiation_samples(self.graph, n_samples=10)
        assert len(self.builder) >= 10

    def test_quantum_samples_added(self):
        self.builder.add_quantum_samples(n_samples=5)
        assert len(self.builder) >= 5

    def test_save_creates_jsonl(self, tmp_path):
        self.builder.add_chameleon_samples(["general"], n_per_form=3)
        p = tmp_path / "out.jsonl"
        n = self.builder.save(p)
        assert n == 3
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_stats_returns_dict(self):
        self.builder.add_chameleon_samples(["general"], n_per_form=2)
        stats = self.builder.stats()
        assert "total_samples" in stats


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def setup_method(self):
        from retbs.core.intelligence_state import IntelligenceState
        self.states = [
            IntelligenceState(generation=i, genome=IntelligenceState.default_genome())
            for i in range(5)
        ]

    def test_evolutionary_gain_length(self):
        from retbs.utils.metrics import evolutionary_gain
        gains = evolutionary_gain(self.states)
        assert len(gains) == 4

    def test_lineage_report_keys(self):
        from retbs.utils.metrics import lineage_report
        r = lineage_report(self.states)
        for key in ("n_generations", "initial_coherence", "final_coherence", "cumulative_gain"):
            assert key in r

    def test_compare_states(self):
        from retbs.utils.metrics import compare_states
        result = compare_states(self.states[0], self.states[-1])
        assert "coherence_delta" in result


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_loads(self):
        from retbs.utils.config import default_config
        cfg = default_config()
        assert cfg.evolution.max_generations > 0

    def test_save_load_roundtrip(self, tmp_path):
        from retbs.utils.config import RETBSConfig
        cfg = RETBSConfig(experiment_name="test_exp")
        p = tmp_path / "cfg.json"
        cfg.save(p)
        loaded = RETBSConfig.load(p)
        assert loaded.experiment_name == "test_exp"

    def test_radiation_intensity_schedule(self):
        from retbs.utils.config import RETBSConfig, RadiationConfig
        cfg = RETBSConfig(radiation=RadiationConfig(
            base_intensity=0.1, max_intensity=0.4, intensity_schedule="increasing"
        ), evolution=__import__("retbs.utils.config", fromlist=["EvolutionConfig"]).EvolutionConfig(max_generations=10))
        i0 = cfg.radiation_intensity(0)
        i9 = cfg.radiation_intensity(9)
        assert i9 > i0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: full pipeline smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_two_generation_cycle(self):
        from retbs.core.kernel import default_kernel
        from retbs.core.intelligence_state import IntelligenceState
        from retbs.core.retbs_pipeline import RETBSPipeline

        kernel  = default_kernel()
        initial = IntelligenceState(generation=0, genome=IntelligenceState.default_genome())
        pipe    = RETBSPipeline(kernel, initial)

        results = pipe.run_n_generations(
            n=2,
            radiation_sources=[{"type": "knowledge", "data": ["x"], "intensity": 0.05}],
            n_mutants=4,
        )

        assert len(results) == 2
        assert results[-1].generation == 2
        assert pipe.best_state() is not None

    def test_lineage_summary_length(self):
        from retbs.core.kernel import default_kernel
        from retbs.core.intelligence_state import IntelligenceState
        from retbs.core.retbs_pipeline import RETBSPipeline

        kernel  = default_kernel()
        initial = IntelligenceState(generation=0, genome=IntelligenceState.default_genome())
        pipe    = RETBSPipeline(kernel, initial)
        pipe.run_n_generations(n=3, n_mutants=3)

        summary = pipe.lineage_summary()
        assert len(summary) == 4  # gen 0 + 3 new


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
