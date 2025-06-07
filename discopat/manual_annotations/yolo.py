# %%
# Imports
from __future__ import annotations

from pathlib import Path

from discopat.core import Box, Frame, Movie
from discopat.repositories.local import DISCOPATH

# %%
# Function definitions


class YoloAnnotationImporter:
    def __init__(
        self,
        simulation: str,
        label_map: dict[int, str],
        data_dir: Path = DISCOPATH,
    ):
        self.data_dir = data_dir
        self.label_map = label_map
        self.simulation = simulation

        self.image_width = None
        self.image_height = None
        self.get_image_dimensions()

    def get_image_dimensions():
        pass

    def txt_to_box(self, txt_box: str) -> Box:
        """Convert txt box annotation to discopat's Box format.

        Args:
            txt_box: line of text encoding a box in the following format:
                "class_id x_center y_center width height",
                where the last 4 values are expressed relatively to the image size.

        Returns:
            discopat.core.Box

        """  # noqa: E501
        class_id, x_center, y_center, width, height = txt_box.split()

        label = self.label_map[int(class_id)]

        x_center = float(x_center) * self.image_width
        y_center = float(y_center) * self.image_height
        width = float(width) * self.image_width
        height = float(height) * self.image_height

        return Box(
            label=label,
            x=x_center - width / 2,
            y=y_center - height / 2,
            width=width,
            height=height,
        )

    def txt_to_frame(self, txt_frame: str, frame_name: str) -> Frame:
        return Frame(
            name=frame_name,
            width=self.image_width,
            height=self.image_height,
            annotations=[
                self.txt_to_box(txt_box) for txt_box in txt_frame.splitlines()
            ],
        )

    def path_to_frame(self, path: Path) -> Frame:
        frame_name = path.stem.split("_")[-1]
        with Path.open(path) as f:
            return self.txt_to_frame(f.read(), frame_name=frame_name)

    def make_movie(self) -> Movie:
        path_list = sorted(
            (self.data_dir / "labels").glob(f"{self.simulation}_*.txt"),
            key=lambda x: x.stem,
        )
        return Movie(
            name=self.simulation,
            frames=[self.path_to_frame(path) for path in path_list],
            tracks=[],
        )
