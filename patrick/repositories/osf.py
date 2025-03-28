from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from osfclient import OSF

from patrick.core import Frame, Movie
from patrick.repositories.repository import Repository

PATRICK_OSF_PROJECT_ID = "jtp4z"

osf = OSF()
project = osf.project(PATRICK_OSF_PROJECT_ID)
storage = project.storage("osfstorage")

STORAGE_DICT = {file.path[1:]: file for file in storage.files}


class OSFRepository(Repository):
    data_source = "osf"

    def __init__(self, name: str):
        super().__init__(name)
        self._path = Path(name)
        self._storage_dict = {
            k: v for k, v in STORAGE_DICT.items() if k.split("/")[0] == name
        }

    def write(self, content_path: str or Path, content: Any) -> None:
        msg = "OSFRepository is intended to be read-only."
        raise NotImplementedError(msg)

    @staticmethod
    def _open_mock_file() -> BytesIO:
        buffer = BytesIO()
        buffer.mode = "b"
        return buffer


class OSFNNModelRepository(OSFRepository):
    def read(self, content_path: str or Path) -> dict[str, dict or BytesIO]:
        return {
            "label_map": self._load_yaml_file(content_path, "label_map.yaml"),
            "model_parameters": self._load_yaml_file(
                content_path, "model_parameters.yaml"
            ),
            "raw_net": self._load_raw_net(content_path),
        }

    def _load_yaml_file(
        self, content_path: str or Path, file_name: str
    ) -> dict[str, int or dict]:
        yaml_file = self._storage_dict[
            str(self._path / content_path / file_name)
        ]
        buffer = self._open_mock_file()
        yaml_file.write_to(buffer)
        return yaml.safe_load(buffer.getvalue())

    def _load_raw_net(self, content_path: str or Path):
        net_file = self._storage_dict[
            str(self._path / content_path / "net.pth")
        ]
        buffer = self._open_mock_file()
        net_file.write_to(buffer)
        buffer.seek(0)
        return buffer


class OSFMovieRepository(OSFRepository):

    def __init__(self, name: str):
        self.name = name

        folder_name = {"input_movies": "input"}[name]
        self._path = Path(folder_name)
        self._storage_dict = {
            k: v
            for k, v in STORAGE_DICT.items()
            if k.split("/")[0] == folder_name
        }

    def read(self, content_path: str or Path) -> Movie:
        experiment, field = self._parse_movie_name(movie_name=str(content_path))
        movie_file = self._storage_dict[
            str(self._path / experiment / f"{field}_movie.json")
        ]
        buffer = self._open_mock_file()
        movie_file.write_to(buffer)
        buffer.seek(0)

        movie_as_dict = json.load(buffer)
        movie = Movie.from_dict(movie_as_dict)
        self._load_image_arrays(movie)

        return movie

    @staticmethod
    def _parse_movie_name(movie_name: str) -> tuple[str, str]:
        return tuple(movie_name.split("/"))

    def _load_image_array(self, movie: Movie, frame: Frame) -> None:
        experiment, field = self._parse_movie_name(movie_name=movie.name)
        frame_id = int(frame.name)
        image_array_file = self._storage_dict[
            str(self._path / experiment / f"{field}_frame_{frame_id}.txt")
        ]
        buffer = self._open_mock_file()
        image_array_file.write_to(buffer)
        buffer.seek(0)
        text_data = buffer.read().decode("utf-8")
        frame.image_array = np.genfromtxt(
            text_data.splitlines(), delimiter=" ", dtype=np.float64
        )

    def _load_image_arrays(self, movie: Movie) -> None:
        for frame in movie.frames:
            self._load_image_array(movie, frame)
