from pathlib import Path

import numpy as np
from PIL import Image

from discopat.core import Frame, Movie
from discopat.repositories.local import DATA_DIR_PATH, LocalRepository


class MOTRepository(LocalRepository):
    def __init__(self, name: str = ""):
        super().__init__(name)
        self._input_directory_path = DATA_DIR_PATH / "MOT17"

    def read(self, content_path: str or Path) -> Movie:
        # TODO: maybe add options none, default, and all for detection models
        movie_info = self._parse_content_path(content_path)
        data_folder = (
            self.input_directory_path / content_path
            if movie_info["detection_model"] == ""
            else self._input_directory_path / f"{content_path}-DPM"
        )
        image_folder = data_folder / "img1"
        annotation_folder = data_folder / "det"

        movie = Movie(name=str(content_path), frames=[], tracks=[])
        for path in image_folder.glob("*.jpg"):
            image = Image.open("your_image.jpg")
            image_array = np.array(image)
            height, width = image_array.shape[:2]
            movie.frames.append(
                Frame(
                    name=path.stem,
                    width=width,
                    height=height,
                    image_array=image_array,
                    annotations=[],
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

    def write(self, content_path: str or Path, content: Movie) -> None:
        pass
