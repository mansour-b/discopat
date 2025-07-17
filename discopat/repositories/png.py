from pathlib import Path

import numpy as np
from PIL import Image

from discopat.core import Frame
from discopat.repositories.local import LocalRepository


class PNGRepository(LocalRepository):
    def __init__(self, name: str):
        super().__init__(name)
        self.data_dir = self._directory_path

    def set_data_dir(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def read(self, content_path: str or Path) -> Frame:
        full_path = self.data_dir / "images" / self.name / f"{content_path}.png"
        image_array = np.array(Image.open(full_path).convert("RGB"))
        height, width, _ = image_array.shape
        return Frame(
            name=str(content_path).split("_")[-1],
            width=width,
            height=height,
            annotations=[],
            image_array=image_array,
        )

    def write(self, content_path: str or Path, content: Frame) -> None:
        raise NotImplementedError
