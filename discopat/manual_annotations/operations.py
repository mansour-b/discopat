from __future__ import annotations

from discopat.core import Box, Keypoint


def keypoint_to_box(
    keypoint: Keypoint, w_padding: float, h_padding: float
) -> Box:
    xmin, xmax, ymin, ymax = get_bounding_box(keypoint.point_list)

    bbox_width = xmax - xmin
    bbox_height = ymax - ymin

    output_width = bbox_width * (1 + w_padding)
    output_height = bbox_height * (1 + h_padding)

    x = xmin - bbox_width * w_padding / 2
    y = ymin - bbox_height * h_padding / 2

    return Box(
        label=keypoint.label,
        x=x,
        y=y,
        width=output_width,
        height=output_height,
        score=keypoint.score,
    )


def get_bounding_box(
    point_list: list[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xmin = min(point[0] for point in point_list)
    xmax = max(point[0] for point in point_list)
    ymin = min(point[1] for point in point_list)
    ymax = max(point[1] for point in point_list)

    return xmin, xmax, ymin, ymax
