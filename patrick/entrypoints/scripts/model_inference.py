from __future__ import annotations

from patrick.core import Model, Movie
from patrick.core.value_objects import ComputingDevice, DataSource, Framework
from patrick.interfaces.cnn import NetBuilder, TorchNetBuilder
from patrick.interfaces.model import ModelBuilder
from patrick.interfaces.repository import Repository
from patrick.repositories.local import (
    LocalModelRepository,
    LocalMovieRepository,
    LocalTorchNetRepository,
)


def model_repository_factory(data_source: DataSource) -> Repository:
    class_dict = {"local": LocalModelRepository}
    return class_dict[data_source]()


def net_repository_factory(
    data_source: DataSource, framework: Framework, device: ComputingDevice
) -> Repository:
    class_dict = {"local": {"torch": LocalTorchNetRepository}}
    concrete_class = class_dict[data_source][framework]
    return concrete_class(device)


def movie_repository_factory(data_source: DataSource, name: str) -> Repository:
    class_dict = {"local": LocalMovieRepository}
    repository = class_dict[data_source]()
    repository.name = name
    return repository


def net_builder_factory(
    framework: Framework,
    device: ComputingDevice,
    net_repository: Repository,
) -> NetBuilder:
    class_dict = {"torch": TorchNetBuilder}
    return class_dict[framework](device, net_repository)


def load_model(
    model_name: str,
    data_source: DataSource,
    framework: Framework,
    device: ComputingDevice,
) -> Model:

    net_repository = net_repository_factory(data_source, framework, device)
    net_builder = net_builder_factory(framework, device, net_repository)

    model_repository = model_repository_factory(data_source)
    model_repository._net_builder = net_builder

    model_builder = ModelBuilder(model_name, model_repository)

    return model_builder.build()


def load_movie(movie_name: str, data_source: DataSource) -> Movie:
    movie_repository = movie_repository_factory(data_source, "input")
    return movie_repository.read(movie_name)


def compute_predictions(model: Model, movie: Movie) -> Movie:
    predicted_frames = [model.predict(frame) for frame in movie.frames]
    return Movie(name=movie.name, frames=predicted_frames, tracks=[])


def save_movie(movie: Movie, data_source: DataSource) -> None:
    movie_repository = movie_repository_factory(data_source, "output")
    movie_repository.write(movie_name, movie)


if __name__ == "__main__":

    movie_name = "blob"
    model_name = "faster_rcnn_241113_131447"
    data_source = "local"
    framework = "torch"
    computing_device = "cpu"

    model = load_model(
        model_name=model_name,
        data_source=data_source,
        framework=framework,
        device=computing_device,
    )

    movie = load_movie(movie_name=movie_name, data_source=data_source)

    analysed_movie = compute_predictions(model, movie)

    save_movie(analysed_movie, data_source=data_source)
