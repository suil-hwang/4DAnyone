"""Normalize Space arguments through the CLI contract."""

from __future__ import annotations

import inspect

from fdanyone.errors import ConfigurationError
from fdanyone.video import validate_clip_options
from fdanyone.views import resolve_view_plan


def complete_options(options: dict) -> dict:
    from inference import inference

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(inference).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    return {**defaults, **options}


def validate_options(options: dict) -> None:
    """Apply the CLI camera grouping and clip contracts before submission."""

    options = complete_options(options)
    resolve_view_plan(
        **{
            key: options[key]
            for key in (
                "views_per_layer",
                "layer_pitches",
                "start_yaw",
                "yaw_span",
                "enable_rcp",
                "enable_tcr",
            )
        }
    )
    fps = options["target_fps"]
    validate_clip_options(start_time=options["start_time"], fps=None if str(fps).lower() == "auto" else fps)


def layout_options(views, pitches, start_yaw, yaw_span) -> dict:
    values = {"views_per_layer": views, "start_yaw": start_yaw, "yaw_span": yaw_span}
    for name, value in values.items():
        if int(value) != float(value):
            raise ConfigurationError(f"{name} must be an integer.")
    values = {name: int(value) for name, value in values.items()}
    values["layer_pitches"] = pitches
    resolve_view_plan(**values)
    return values
