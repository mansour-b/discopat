from __future__ import annotations

import time
from typing import TYPE_CHECKING

import imageio
import matplotlib as mpl
from matplotlib import pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage

if TYPE_CHECKING:
    from matplotlib.axes._axes import Axes

    from patrick.core import Annotation, Box, Frame, Keypoint


def make_movie(
    image_list: list[Frame],
    image_dir_name: str,
    fps: int,
    output_format: str,
    **plot_image_kwargs,
):
    mpl.use("agg")

    time_str = time.strftime("%y%m%d_%H%M%S")
    movie_path = f"misc/{image_dir_name}_{time_str}.{output_format}"

    with imageio.get_writer(movie_path, fps=fps) as writer:
        for image in image_list:
            fig = plot_frame(
                image,
                image_dir_name,
                show_figure=False,
                return_figure=True,
                **plot_image_kwargs,
            )
            writer.append_data(mplfig_to_npimage(fig))
            plt.close(fig)


def plot_frame(
    frame: Frame,
    cmap: str = "gray",
    annotation_color: str = "tab:red",
    show_figure: bool = True,
    return_figure: bool = False,
):
    image_array = frame.image_array
    fig, ax = plt.subplots(1, 1)
    ax.imshow(image_array, cmap=cmap)
    ax.axis("off")
    for annotation in frame.annotations:
        plot_annotation(ax, annotation, color=annotation_color)
    if show_figure:
        fig.show()
    if return_figure:
        return fig


def plot_annotation(ax: Axes, annotation: Annotation, color: str):
    annotation_type_dict = {"box": plot_box, "polyline": plot_keypoint}
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
        ax.plot([x1, x2], [y1, y2], color=color)
        ax.plot([x1, x2], [y1, y2], color=color)
