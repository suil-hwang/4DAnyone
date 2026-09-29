"""Resolve the reader-facing target-view layout and inference grouping."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from operator import index

from fdanyone.config import CAMERA
from fdanyone.errors import ConfigurationError

VIEWS_PER_GROUP = 6
MIN_PITCH = -15
MAX_PITCH = 45

# RCP generates six canonical proposal cameras; the first four become target references.
RCP_CAMERA_ORDER = (4, 9, 14, 19, 0, 12)


@dataclass(frozen=True)
class TargetView:
    """One requested camera in the layer-major public view order."""

    camera_id: int
    layer_index: int
    pitch: int
    yaw: float


@dataclass(frozen=True)
class ViewPlan:
    """Validated target cameras and method components for one run."""

    views_per_layer: int
    layer_pitches: tuple[int, ...]
    start_yaw: int
    yaw_span: int
    enable_rcp: bool
    enable_tcr: bool

    @property
    def num_layers(self) -> int:
        return len(self.layer_pitches)

    @property
    def num_target_views(self) -> int:
        return self.views_per_layer * self.num_layers

    @property
    def num_groups(self) -> int:
        return self.num_target_views // VIEWS_PER_GROUP

    @property
    def target_views(self) -> tuple[TargetView, ...]:
        step = self.yaw_span / self.views_per_layer
        return tuple(
            TargetView(
                camera_id=layer_index * self.views_per_layer + view_index,
                layer_index=layer_index,
                pitch=pitch,
                yaw=self.start_yaw + view_index * step,
            )
            for layer_index, pitch in enumerate(self.layer_pitches)
            for view_index in range(self.views_per_layer)
        )

    @property
    def front_camera_ids(self) -> tuple[int, ...]:
        return tuple(view.camera_id for view in self.target_views if abs(view.yaw % 360.0) < 1e-8)

    @property
    def rcp_camera_ids(self) -> tuple[int, ...]:
        return RCP_CAMERA_ORDER if self.enable_rcp else ()

    @property
    def is_canonical_target_ring(self) -> bool:
        return (
            self.views_per_layer == CAMERA.count
            and self.layer_pitches == (int(CAMERA.pitch_degrees),)
            and self.start_yaw == 0
            and self.yaw_span == 360
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "views_per_layer": self.views_per_layer,
            "layer_pitches": list(self.layer_pitches),
            "start_yaw": self.start_yaw,
            "yaw_span": self.yaw_span,
            "enable_rcp": self.enable_rcp,
            "enable_tcr": self.enable_tcr,
        }

    @classmethod
    def from_dict(cls, value: dict) -> ViewPlan:
        return resolve_view_plan(
            views_per_layer=value["views_per_layer"],
            layer_pitches=value["layer_pitches"],
            start_yaw=value["start_yaw"],
            yaw_span=value["yaw_span"],
            enable_rcp=value["enable_rcp"],
            enable_tcr=value["enable_tcr"],
        )


def resolve_view_plan(
    *,
    views_per_layer: int = 24,
    layer_pitches: Sequence[int] = (15,),
    start_yaw: int = 0,
    yaw_span: int = 360,
    enable_rcp: bool = True,
    enable_tcr: bool = True,
) -> ViewPlan:
    """Validate the compact CLI settings before expensive work starts."""

    views_per_layer = index(views_per_layer)
    if views_per_layer <= 0:
        raise ConfigurationError(f"views_per_layer must be positive: {views_per_layer}.")
    pitches = tuple(map(index, layer_pitches))
    if not pitches or len(set(pitches)) != len(pitches):
        raise ConfigurationError(f"layer_pitches must be non-empty and distinct: {pitches}.")
    if any(not MIN_PITCH <= pitch <= MAX_PITCH for pitch in pitches):
        raise ConfigurationError(f"Layer pitches outside [{MIN_PITCH}, {MAX_PITCH}]: {pitches}.")
    start_yaw = (index(start_yaw) + 180) % 360 - 180
    yaw_span = index(yaw_span)
    if not 0 < yaw_span <= 360:
        raise ConfigurationError(f"yaw_span must be in [1, 360]: {yaw_span}.")
    total_views = views_per_layer * len(pitches)
    if total_views % VIEWS_PER_GROUP:
        raise ConfigurationError(f"Target view count must be divisible by {VIEWS_PER_GROUP}: {total_views}.")
    if not isinstance(enable_rcp, bool) or not isinstance(enable_tcr, bool):
        raise ConfigurationError("enable_rcp and enable_tcr must be booleans.")

    # One target group is generated directly, without proposal views.
    rcp_active = enable_rcp and total_views > VIEWS_PER_GROUP
    return ViewPlan(
        views_per_layer=views_per_layer,
        layer_pitches=pitches,
        start_yaw=start_yaw,
        yaw_span=yaw_span,
        enable_rcp=rcp_active,
        enable_tcr=enable_tcr,
    )
