from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from discopat.core import Box, Frame, Movie
from discopat.repositories.local import DATA_DIR_PATH, LocalRepository


class MOTRepository(LocalRepository):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._input_directory_path = DATA_DIR_PATH / "MOT17"

    def read(self, content_path: str or Path) -> Movie:
        # TODO: maybe add options none, default, and all for detection models
        movie_info = self._parse_content_path(content_path)
        data_folder = (
            content_path
            if movie_info["detection_model"] != ""
            else f"{content_path}-DPM"
        )
        data_path = self._input_directory_path / movie_info["set"] / data_folder
        image_path = data_path / "img1"
        annotation_path = data_path / "det" / "det.txt"

        annotation_dict = self._load_annotations(annotation_path)

        movie = Movie(name=str(content_path), frames=[], tracks=[])
        for path in image_path.glob("*.jpg"):
            frame_name = path.stem
            image = Image.open(path)
            image_array = np.array(image)
            height, width = image_array.shape[:2]
            movie.frames.append(
                Frame(
                    name=frame_name,
                    width=width,
                    height=height,
                    image_array=image_array,
                    annotations=annotation_dict[int(frame_name)],
                )
            )
        return movie

    @staticmethod
    def _parse_content_path(content_path: str or Path):
        elements = str(content_path).split("-")
        movie_index = int(elements[1])
        dataset = "train" if movie_index in {2, 4, 5, 9, 10, 11, 13} else "test"
        detection_model = "" if len(elements) == 2 else elements[2]
        return {
            "index": movie_index,
            "set": dataset,
            "detection_model": detection_model,
        }

    @staticmethod
    def _load_annotations(annotation_path: Path) -> defaultdict[int, list[Box]]:
        res = defaultdict()
        with Path.open(annotation_path) as f:
            for line in f:
                print(line)
                frame_id, _, x, y, w, h, _, _, _ = line.split(",")
                res[int(frame_id)].append(
                    Box(
                        label="pedestrian",
                        x=float(x),
                        y=float(y),
                        width=float(w),
                        height=float(h),
                        score=1.0,
                    )
                )
        return res

    def write(self, content_path: str or Path, content: Movie) -> None:
        pass
