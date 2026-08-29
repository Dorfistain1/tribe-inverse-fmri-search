from inverse_search.candidate import Candidate
from inverse_search.generators.base import StimulusGenerator
from tribe_core import Stimulus


class AudioGenerator(StimulusGenerator):
    """Not yet implemented -- mutation strategy for audio is undecided.

    Candidates considered but not chosen yet: perturbing a raw waveform
    directly, mutating parameters of a synthesizer/generative audio
    model, or interpolating in a learned latent space. Each gives a
    completely different search space and needs its own design pass.
    """

    modality = "audio"

    def initial_population(self, n: int) -> list[Stimulus]:
        raise NotImplementedError("Audio mutation strategy not yet designed.")

    def mutate(self, parent: Candidate) -> Stimulus:
        raise NotImplementedError("Audio mutation strategy not yet designed.")
