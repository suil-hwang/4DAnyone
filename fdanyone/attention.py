"""Pure validation and selection of the inference attention backend."""

from __future__ import annotations

from collections.abc import Mapping

from fdanyone.errors import ConfigurationError

ATTENTION_BACKEND_PRIORITY = ("sageattention", "sdpa")
ATTENTION_BACKENDS = ("auto", *ATTENTION_BACKEND_PRIORITY)


def validate_attention_backend(backend: str) -> None:
    if not isinstance(backend, str) or backend not in ATTENTION_BACKENDS:
        raise ConfigurationError(f"Unknown attention backend: {backend!r}.")


def resolve_attention_backend(backend: str, availability: Mapping[str, bool]) -> str:
    """Honor explicit selection, otherwise use the installed-backend priority."""

    validate_attention_backend(backend)
    if backend == "auto":
        return next(candidate for candidate in ATTENTION_BACKEND_PRIORITY if availability[candidate])
    if not availability[backend]:
        raise ConfigurationError(f"Attention backend unavailable: {backend}; use sdpa.")
    return backend
