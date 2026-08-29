from __future__ import annotations

import os

from tribe_core import _windows_patches
from tribe_core.cache import PredictionCache
from tribe_core.config import TribeConfig
from tribe_core.prediction import BrainPrediction
from tribe_core.stimulus import Stimulus


def _normalize_timing(segments) -> list[dict]:
    """model.predict()'s second return value has been observed as both a
    pandas DataFrame and a plain list depending on modality/path taken."""
    if segments is None:
        return []
    if hasattr(segments, "to_dict"):
        return segments.to_dict("records")
    return list(segments)


class TribeRuntime:
    """The only thing an experiment should need to talk to. See
    mainStructure.md "Shared TRIBE Runtime" / "Experiment Contract" --
    experiments call predict(stimulus) and never touch TRIBE, tribev2,
    torch, or the model checkpoint directly.

    Loading and inference happen exactly as in scripts/check_vram.py,
    which is the source of truth for the Windows workarounds; see
    _windows_patches.py for what each one fixes and why.
    """

    def __init__(self, config: TribeConfig | None = None):
        self.config = config or TribeConfig.from_yaml()
        self._configure_environment()
        _windows_patches.apply(self.config)
        self._model = None
        self.cache = PredictionCache(self.config.prediction_cache)

    def _configure_environment(self) -> None:
        os.environ.setdefault("HF_HOME", self.config.hf_cache_dir())
        os.environ.setdefault("TRANSFORMERS_CACHE", self.config.hf_cache_dir())
        os.environ.setdefault("TORCH_HOME", self.config.torch_cache_dir())
        os.environ.setdefault("UV_CACHE_DIR", self.config.uv_cache_dir())
        os.environ.setdefault("NILEARN_DATA", self.config.nilearn_data_dir())

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        from tribev2 import TribeModel

        checkpoint_dir = _windows_patches.resolve_checkpoint_dir(
            self.config.repo_id, self.config.tribev2_cache_dir()
        )
        self._model = TribeModel.from_pretrained(
            checkpoint_dir,
            checkpoint_name=self.config.checkpoint_name,
            cache_folder=self.config.tribev2_cache_dir(),
            device=self.config.device,
            # num_workers lives on the nested Data config (tribev2.main.Data),
            # not on TribeModel itself -- a flat "num_workers" key is
            # rejected as an unknown field (TribeModel's pydantic model
            # forbids extras). Dotted path matches the convention
            # demo_utils.from_pretrained() already uses for other nested
            # Data fields (e.g. "data.audio_feature.infra.folder").
            config_update={"data.num_workers": self.config.num_workers},
        )
        _windows_patches.patch_transcription_compute_type()

    def predict(
        self,
        stimulus: Stimulus,
        use_cache: bool = True,
        on_stage: "Callable[[str], None] | None" = None,
    ) -> BrainPrediction:
        """on_stage, if given, is called with a short human-readable label
        before each slow step -- callers (e.g. tools/brain_viewer) can use
        it to show real progress instead of a bare spinner. Purely a UI
        hook, changes no behavior."""
        if stimulus.modality not in self.config.supported_modalities:
            raise ValueError(
                f"Modality '{stimulus.modality}' not enabled in "
                f"config.supported_modalities ({self.config.supported_modalities})"
            )

        if use_cache:
            cached = self.cache.get(stimulus, self.config.model_identity)
            if cached is not None:
                if on_stage:
                    on_stage("cache hit")
                return cached

        if on_stage:
            on_stage("loading model" if self._model is None else "model ready")
        model = self.model
        source_kwarg = "audio_path" if stimulus.modality == "audio" else "text_path"

        if on_stage:
            on_stage("building events (transcription/TTS)")
        with _windows_patches.force_cpu_transcription():
            events = model.get_events_dataframe(**{source_kwarg: stimulus.source})

        if on_stage:
            on_stage("predicting brain activity")
        activity, segments = model.predict(events=events)

        prediction = BrainPrediction(
            stimulus_id=stimulus.identifier,
            modality=stimulus.modality,
            activity=activity,
            model_identity=self.config.model_identity,
            timing=_normalize_timing(segments),
            metadata=dict(stimulus.metadata),
        )

        if use_cache:
            self.cache.put(prediction, stimulus)

        return prediction
