from inverse_search.candidate import Candidate
from inverse_search.generators.base import StimulusGenerator
from tribe_core import Stimulus


class TextGenerator(StimulusGenerator):
    """Not yet implemented -- mutation strategy for text is undecided.

    Candidates considered but not chosen yet: token-level edits (swap/
    insert/delete words), sentence-level rewriting via an LLM guided by
    fitness feedback, or latent-space mutation in an embedding space.
    Each gives a completely different search space and needs its own
    design pass.
    """

    modality = "text"

    def initial_population(self, n: int) -> list[Stimulus]:
        raise NotImplementedError("Text mutation strategy not yet designed.")

    def mutate(self, parent: Candidate) -> Stimulus:
        raise NotImplementedError("Text mutation strategy not yet designed.")
