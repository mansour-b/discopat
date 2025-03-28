from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import yaml

from patrick.core import Frame, Movie, NNModel
from patrick.repositories.repository import Repository

DATA_DIR_PATH = Path.home() / "data"
PATRICK_DIR_PATH = DATA_DIR_PATH / "pattern_discovery"


class LocalRepository(Repository):
    data_source = "local"

    def __init__(self, name: str):
        super().__init__(name)
        self._directory_path = PATRICK_DIR_PATH / self.name
        self._directory_path.mkdir(parents=True, exist_ok=True)


class LocalFrameRepository(LocalRepository):

    def __init__(self, name: str):
        self.name = name
        self._directory_path = {
            "input_frames": PATRICK_DIR_PATH / "input",
            "output_frames": PATRICK_DIR_PATH / "output",
        }[name]

    def read(self, content_path: str or Path) -> Frame:
        experiment, field, frame_id = self._parse_frame_name(str(content_path))
        file_stem = f"{field}_frame_{frame_id}"
        metadata_path = self._directory_path / experiment / f"{file_stem}.json"
        image_array_path = (
            self._directory_path / experiment / f"{file_stem}.txt"
        )

        if not metadata_path.exists() and not image_array_path.exists():
            msg = f"""
                Could not find any information on frame '{content_path}'.
                Please make sure that either:
                - a JSON file with metadata or
                - a txt file containing the image array
                exists in the folder.
            """.strip()
            raise FileNotFoundError(msg)

        if not metadata_path.exists():
            image_array = np.loadtxt(image_array_path)
            height, width = image_array.shape
            return Frame(
                name=file_stem,
                width=width,
                height=height,
                annotations=[],
                image_array=image_array,
            )

        with Path.open(metadata_path) as f:
            frame = Frame.from_dict(json.load(f))

        if not image_array_path.exists():
            return frame

        frame.image_array = np.loadtxt(image_array_path)
        return frame

    def write(self, content_path: str or Path, content: Frame) -> None:
        raise NotImplementedError

    @staticmethod
    def _parse_frame_name(frame_name: str) -> tuple[str, str, int]:
        experiment = frame_name.split("/")[0]
        field, _, frame_id = frame_name.split("/")[1].split("_")
        return experiment, field, int(frame_id)


class LocalNNModelRepository(LocalRepository):
    def read(self, content_path: str or Path) -> dict[str, dict or BytesIO]:

        return {
            "label_map": self._load_label_map(content_path),
            "model_parameters": self._load_model_parameters(content_path),
            "raw_net": self._load_raw_net(content_path),
        }

    def write(self, content_path: str or Path, content: NNModel) -> None:
        raise NotImplementedError

    def _load_label_map(self, content_path: str or Path) -> dict[str, int]:
        full_content_path = (
            self._directory_path / content_path / "label_map.yaml"
        )
        with Path.open(full_content_path) as f:
            return yaml.safe_load(f)

    def _load_model_parameters(
        self, content_path: str or Path
    ) -> dict[str, dict]:
        full_content_path = (
            self._directory_path / content_path / "model_parameters.yaml"
        )
        with Path.open(full_content_path) as f:
            return yaml.safe_load(f)

    def _load_raw_net(self, content_path: str or Path) -> BytesIO:
        full_content_path = self._directory_path / content_path / "net.pth"
        with Path.open(full_content_path, "rb") as f:
            return BytesIO(f.read())


class LocalMovieRepository(LocalRepository):

    def __init__(self, name: str):
        self.name = name
        self._directory_path = {
            "input_movies": PATRICK_DIR_PATH / "input",
            "output_movies": PATRICK_DIR_PATH / "output",
        }[name]

    def read(self, content_path: str or Path) -> Movie:
        experiment, field = self._parse_movie_name(movie_name=str(content_path))
        full_content_path = (
            self._directory_path / experiment / f"{field}_movie.json"
        )

        with Path.open(full_content_path) as f:
            movie = Movie.from_dict(json.load(f))
        self._load_image_arrays(movie)

        return movie

    def write(self, content_path: str or Path, content: Movie) -> None:
        experiment, field = self._parse_movie_name(movie_name=str(content_path))
        full_content_path = (
            self._directory_path / experiment / f"{field}_movie.json"
        )
        full_content_path.parent.mkdir(exist_ok=True)

        with Path.open(full_content_path, "w") as f:
            json.dump(content.to_dict(), f, indent=2)

    @staticmethod
    def _parse_movie_name(movie_name: str) -> tuple[str, str]:
        return tuple(movie_name.split("/"))

    def _load_image_array(self, movie: Movie, frame: Frame) -> None:
        experiment, field = self._parse_movie_name(movie_name=movie.name)
        frame_id = int(frame.name)
        image_array_path = (
            self._directory_path / f"{experiment}/{field}_frame_{frame_id}.txt"
        )
        frame.image_array = np.loadtxt(image_array_path)

    def _load_image_arrays(self, movie: Movie) -> None:
        for frame in movie.frames:
            self._load_image_array(movie, frame)
