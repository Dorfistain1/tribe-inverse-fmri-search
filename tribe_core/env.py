from __future__ import annotations

import os
from pathlib import Path


def configure_hf_cache(model_root: str) -> None:
    """Point HF_HOME/TRANSFORMERS_CACHE at model_root, preserving any
    existing `hf auth login` token via HF_TOKEN first.

    Two stacked gotchas here, both real, both discovered the hard way:

    1. Some huggingface_hub versions only check $HF_HOME/token, not the
       real default ~/.cache/huggingface/token, once HF_HOME points
       elsewhere -- silently breaking gated-model access (Llama, Stable
       Audio Open) with no error until the download itself 401s.
       HF_TOKEN is a stable, version-independent lookup path, so this
       captures the token that way instead.

    2. Naively fixing that by calling `huggingface_hub.get_token()` is
       itself broken: importing huggingface_hub at all bakes HF_HOME
       into an internal constant *at that moment* -- setting the
       HF_HOME env var afterward has no effect for the rest of the
       process, since the constant was already computed. So the token
       file is read directly here (plain file I/O, no huggingface_hub
       import) *before* HF_HOME is touched at all.

    Shared by TribeRuntime and inverse_search's generators, which both
    need this same fix independently (they don't share a process
    lifecycle -- see inverse_search/DESIGN.md's sequential load/unload
    plan).
    """
    if "HF_TOKEN" not in os.environ and "HF_HOME" not in os.environ:
        default_token_file = Path.home() / ".cache" / "huggingface" / "token"
        if default_token_file.exists():
            os.environ["HF_TOKEN"] = default_token_file.read_text().strip()

    os.environ.setdefault("HF_HOME", os.path.join(model_root, "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(model_root, "huggingface"))
