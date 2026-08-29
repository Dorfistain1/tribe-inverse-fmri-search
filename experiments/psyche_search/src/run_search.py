"""
psyche_search's only job: define what "psychedelic-like" means as a
NeuralTarget, then hand it to the shared inverse_search engine. All
generic search machinery (selection loop, fitness scoring, TRIBE calls)
lives in inverse_search/, not here -- see mainStructure.md "Experiment
Contract".

Not runnable yet: inverse_search's generators (generators/audio.py,
generators/text.py) are stubs until a stimulus mutation strategy is
chosen. This file exists to pin down the intended usage shape.
"""

from inverse_search import EvolutionarySearch, NeuralTarget, SearchConfig
from inverse_search.generators.audio import AudioGenerator
from tribe_core import TribeRuntime


def build_psychedelic_target() -> NeuralTarget:
    """Literature-encoded placeholder (see restructure.md's "target
    source" discussion): hand-specified from the cross-drug psychedelic
    connectivity literature's most consistently reported finding --
    greater coupling between higher-order networks (Default Mode,
    Frontoparietal) and sensory/motor networks (restructure.md sections
    1-2). Not derived from real psychedelic fMRI data -- a real
    PsiConnect-derived psilocybin+music minus baseline+music contrast is
    the planned upgrade, same NeuralTarget shape, see section 4."""
    return NeuralTarget(
        name="literature_psychedelic_v1",
        network_deltas={
            ("Default Mode", "Visual"): +1,
            ("Frontoparietal", "Visual"): +1,
            ("Frontoparietal", "Somatomotor"): +1,
        },
    )


def main():
    runtime = TribeRuntime()
    target = build_psychedelic_target()
    search = EvolutionarySearch(
        runtime=runtime,
        generator=AudioGenerator(),
        target=target,
        config=SearchConfig(population_size=20, n_generations=10),
    )
    best = search.run()
    print(best[0])


if __name__ == "__main__":
    main()
