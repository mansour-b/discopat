from patryk.core import DataSource
from patryk.repositories.local import (
    LocalFrameRepository,
    LocalMovieRepository,
    LocalNNModelRepository,
)
from patryk.repositories.osf import OSFMovieRepository, OSFNNModelRepository
from patryk.repositories.repository import Repository


def repository_factory(data_source: DataSource, name: str) -> Repository:
    repo_class_dict = {
        "local": {
            "input_frames": LocalFrameRepository,
            "output_frames": LocalFrameRepository,
            "input_movies": LocalMovieRepository,
            "output_movies": LocalMovieRepository,
            "models": LocalNNModelRepository,
        },
        "osf": {
            "input_movies": OSFMovieRepository,
            "models": OSFNNModelRepository,
        },
    }
    repo_class = repo_class_dict[data_source][name]
    return repo_class(name)
