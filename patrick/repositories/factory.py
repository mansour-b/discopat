from patrick.core import DataSource
from patrick.repositories.local import (
    LocalFrameRepository,
    LocalMovieRepository,
    LocalNNModelRepository,
)
from patrick.repositories.osf import OSFMovieRepository, OSFNNModelRepository
from patrick.repositories.repository import Repository


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
