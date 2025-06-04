from pathlib import Path

import h5py
import numpy as np

from discopat.core import Frame, Movie
from discopat.repositories.local import DATA_DIR_PATH, LocalRepository


class HDF5Repository(LocalRepository):
    def __init__(self, name: str):
        super().__init__(name)
        self._input_directory_path = DATA_DIR_PATH / self.name

    def read(self, content_path: str or Path) -> Movie:
        file_path = (
            self._input_directory_path / content_path / "data_TOKAM_run_00.h5"
        )
        with h5py.File(file_path) as f:
            density = np.array(f["density"])

        num_frames = density.shape[0]
        width = density.shape[2]
        height = density.shape[1]

        return Movie(
            name=f"{content_path}_density",
            frames=[
                Frame(
                    name=str(i),
                    width=width,
                    height=height,
                    annotations=[],
                    image_array=density[i],
                )
                for i in range(num_frames)
            ],
            tracks=[],
        )

    def write(self, content_path, content) -> None:
        pass
