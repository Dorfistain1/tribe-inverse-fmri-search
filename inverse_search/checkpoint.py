"""
Checkpoint save/resume for EvolutionarySearch.run().

A candidate's Stimulus.metadata can hold a raw tensor (AudioGenerator
stashes its latent there for mutate() to perturb later) -- not JSON-
serializable, so each candidate's latent is saved as its own .pt file
in a sidecar folder next to the checkpoint JSON, referenced by path.
Everything else in metadata is assumed JSON-safe (true for every
generator that exists today); a future generator needing something
else non-serializable there would need this extending.

Only search *progress* is persisted (generation index, population,
fitness, lineage) -- not SearchConfig itself. Resuming uses whatever
config the EvolutionarySearch was constructed with; changing
population_size/etc between a pause and its resume is your call, not
validated against what's in the checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from inverse_search.candidate import Candidate
from tribe_core import Stimulus


def save(path: str | Path, generation: int, population: list[Candidate]) -> None:
    path = Path(path)
    latents_dir = path.parent / f"{path.stem}_latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    candidates_data = []
    for candidate in population:
        metadata = dict(candidate.stimulus.metadata)
        latent = metadata.pop("latent", None)
        latent_file = None
        if latent is not None:
            latent_file = str(latents_dir / f"{candidate.stimulus.identifier}.pt")
            torch.save(latent, latent_file)

        candidates_data.append({
            "identifier": candidate.stimulus.identifier,
            "modality": candidate.stimulus.modality,
            "source": candidate.stimulus.source,
            "metadata": metadata,
            "latent_file": latent_file,
            "generation": candidate.generation,
            "parent_id": candidate.parent_id,
            "fitness": candidate.fitness,
        })

    data = {"generation": generation, "population": candidates_data}
    # Write to a temp file then rename over the real path -- avoids a
    # half-written, unreadable checkpoint if the process dies mid-write
    # (rename is effectively atomic on the same filesystem).
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.replace(path)


def load(path: str | Path) -> tuple[int, list[Candidate]]:
    data = json.loads(Path(path).read_text())

    population = []
    for c in data["population"]:
        metadata = dict(c["metadata"])
        if c["latent_file"]:
            # weights_only=True: this file only ever holds a plain
            # tensor (see save() above) -- no need for full pickle
            # deserialization, which would allow arbitrary code
            # execution from a malicious/corrupted checkpoint file.
            metadata["latent"] = torch.load(c["latent_file"], weights_only=True)
        stimulus = Stimulus(
            identifier=c["identifier"],
            modality=c["modality"],
            source=c["source"],
            metadata=metadata,
        )
        population.append(Candidate(
            stimulus=stimulus,
            generation=c["generation"],
            parent_id=c["parent_id"],
            fitness=c["fitness"],
        ))

    return data["generation"], population


def exists(path: str | Path) -> bool:
    return Path(path).exists()
