from __future__ import annotations

import time
from typing import TYPE_CHECKING

import imageio
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm

from discopat.repositories.local import DISCOPATH

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

    from discopat.core import Annotation, Box, Frame, Keypoint, Movie


def to_int(image_array: np.array, eps: float = 1e-10) -> np.array:
    """Convert array to int [0, 255].

    Warning: values will be scaled even if the input array is already of type int.

    """
    min_val = image_array.min()
    max_val = image_array.max()
    return (
        np.round((image_array - min_val) / (max_val - min_val + eps) * 255)
    ).astype(np.uint8)


def get_center_path(track: np.array) -> np.array:
    """Get the trajectory of the center of the boxes corresponding to one track.

    Args:
        track: Array of shape (num_frames, 5) where each line corresponds to a box:
            - track[:, 0] = id of the frame,
            - track[:, 1] = xmin,
            - track[:, 2] = ymin,
            - track[:, 3] = xmax,
            - track[:, 4] = ymax.

    Returns:
        Array of shape (num_frame, 2), where each line corresponds to the (x, y) coordinates of the center of the box.

    """
    xmin_array = track[:, 1]
    ymin_array = track[:, 2]
    xmax_array = track[:, 3]
    ymax_array = track[:, 4]

    id_col = np.expand_dims(track[:, 0], axis=1)

    x_col = np.expand_dims((xmin_array + xmax_array) / 2, axis=1)
    y_col = np.expand_dims((ymin_array + ymax_array) / 2, axis=1)

    return np.hstack([id_col, x_col, y_col])


def frame_to_pil(
    frame: Frame,
    tracks: np.array,
    max_track_length: int = 0,
    persistence: int = 10,
    cmap: str = "gray",
    track_color: str = "red",
) -> Image:
    """Make a PIL image from a frame."""
    i = int(frame.name)

    color_map = mpl.colormaps.get_cmap(cmap)
    image_array = to_int(color_map(to_int(frame.image_array)))
    pil_image = Image.fromarray(image_array)
    pil_image = pil_image.convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for track in tracks.values():
        if int(track[-1, 0]) < i - persistence:
            continue
        current_track = track[track[:, 0] <= i]
        if len(current_track) <= 1:
            continue

        center_path = get_center_path(current_track)
        draw.line(
            xy=[(pos[1], pos[2]) for pos in center_path[-max_track_length:]],
            fill=track_color,
            width=3,
        )

    for box in frame.annotations:
        draw.rectangle(
            [box.xmin, box.ymin, box.xmax, box.ymax], outline=track_color
        )

    return pil_image


def make_movie(
    movie: Movie,
    persistence: int,
    cmap: str,
    track_color: str,
    fps: int,
    output_format: str,
):
    mpl.use("agg")

    time_str = time.strftime("%y%m%d_%H%M%S")
    movie_path = DISCOPATH / f"misc/{movie.name}_{time_str}.{output_format}"

    with imageio.get_writer(movie_path, fps=fps) as writer:
        for frame in tqdm(movie.frames):
            writer.append_data(
                np.array(
                    frame_to_pil(
                        frame,
                        movie.tracks,
                        persistence=persistence,
                        cmap=cmap,
                        track_color=track_color,
                    )
                )
            )


def plot_frame(  # noqa: RET503
    frame: Frame,
    cmap: str = "gray",
    annotation_color: str = "tab:red",
    show_figure: bool = True,
    return_figure: bool = False,
    figure_size: tuple[float, float] or None = None,
    figure_dpi: int or None = None,
):
    mpl.use("inline")
    image_array = frame.image_array
    fig, ax = plt.subplots(1, 1, figsize=figure_size, dpi=figure_dpi)
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.imshow(image_array, cmap=cmap)
    ax.axis("off")
    for annotation in frame.annotations:
        plot_annotation(ax, annotation, color=annotation_color)
    if show_figure:
        plt.show()
    if return_figure:
        return fig


def plot_annotation(ax: Axes, annotation: Annotation, color: str):
    annotation_type_dict = {"box": plot_box, "keypoint": plot_keypoint}
    plot_function = annotation_type_dict[annotation.type]
    plot_function(ax, annotation, color)


def plot_box(ax: Axes, box: Box, color: str):
    ax.add_patch(
        plt.Rectangle(
            xy=(box.xmin, box.ymin),
            width=box.width,
            height=box.height,
            edgecolor=color,
            facecolor="none",
        )
    )


def plot_keypoint(ax: Axes, keypoint: Keypoint, color: str):
    point_list = keypoint.point_list
    for i, point_1 in enumerate(point_list[:-1]):
        point_2 = point_list[i + 1]
        x1, y1 = point_1
        x2, y2 = point_2
        ax.plot([x1, x2], [y1, y2], color=color)
