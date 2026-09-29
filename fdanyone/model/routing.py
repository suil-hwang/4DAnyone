"""Map canonical target cameras onto cyclic denoising groups."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from fdanyone.views import VIEWS_PER_GROUP

if TYPE_CHECKING:
    from fdanyone.views import ViewPlan

CameraOrder = tuple[int, ...]
CameraGroup = tuple[int, ...]
StepGroups = tuple[CameraGroup, ...]
Routes = tuple[StepGroups, ...]


def denoising_camera_order(view_plan: ViewPlan) -> CameraOrder:
    """Return one deterministic cycle over the plan's canonical camera IDs.

    A single row or column keeps its angular order. A two-dimensional layout
    forms a Hamiltonian cycle in the open pitch-by-yaw grid, so partial yaw
    layouts do not need a false edge between their horizontal endpoints.
    """

    views_per_layer = view_plan.views_per_layer
    num_layers = view_plan.num_layers
    if num_layers == 1:
        return tuple(range(views_per_layer))

    # Public IDs follow input layer order. Physical vertical neighbors follow
    # pitch order, so this mapping changes traversal without changing identity.
    layers_by_pitch = tuple(sorted(range(num_layers), key=view_plan.layer_pitches.__getitem__))
    if views_per_layer == 1:
        return layers_by_pitch

    # Keep the released traversal when yaw is even. Odd yaw counts have an even
    # layer count, so transposing the grid gives the same adjacent-cycle construction.
    transpose = views_per_layer % 2 == 1
    rows, columns = (views_per_layer, num_layers) if transpose else (num_layers, views_per_layer)

    def camera_id(row: int, column: int) -> int:
        pitch_rank, yaw_index = (column, row) if transpose else (row, column)
        return layers_by_pitch[pitch_rank] * views_per_layer + yaw_index

    # Reserve the first row as the return lane, then snake through the other rows.
    order = [camera_id(row, 0) for row in range(rows)]
    for column in range(1, columns):
        row_order = range(rows - 1, 0, -1) if column % 2 else range(1, rows)
        order.extend(camera_id(row, column) for row in row_order)
    order.extend(camera_id(0, column) for column in range(columns - 1, 0, -1))

    return tuple(order)


def cyclic_groups(
    camera_order: Sequence[int],
    group_size: int,
    offset: int = 0,
) -> StepGroups:
    """Partition a cyclic camera order into equally sized consecutive groups."""

    order = tuple(camera_order)
    num_views = len(order)
    return tuple(
        tuple(order[(group_start + offset + local_index) % num_views] for local_index in range(group_size))
        for group_start in range(0, num_views, group_size)
    )


def routing_steps(
    *,
    view_plan: ViewPlan,
    num_steps: int,
    tcr_stride: int = 1,
    freeze_after_one_cycle: bool = False,
) -> Routes:
    """Return fixed or TCR-shifted partitions of one global camera ring."""

    camera_order = denoising_camera_order(view_plan)

    def step_offset(step_index: int) -> int:
        if not view_plan.enable_tcr:
            return 0
        offset = step_index * tcr_stride
        if freeze_after_one_cycle and offset >= VIEWS_PER_GROUP:
            return 0
        return offset

    return tuple(
        cyclic_groups(
            camera_order,
            VIEWS_PER_GROUP,
            step_offset(step_index),
        )
        for step_index in range(num_steps)
    )
