import json
from pathlib import Path

from discopat.core import Model, Movie
from discopat.nn_models import FasterRCNNModel
from discopat.repositories.hdf5 import HDF5Repository
from discopat.repositories.local import DISCOPATH, LocalNNModelRepository
from discopat.utils import get_device

MOVIE_TABLE = {
    "blob_dwi_512": "250610_103200",
    "blob_i_512": "250605_164500",
    "turb_dwi_256": "250603_111000",
    "turb_dwi_512": "250610_110800",
    "turb_i_256": "250603_105600",
    "turb_i_512": "250715_150500",
}

SET_TABLE = {
    "train": {"movie_name": "blob_i_512", "annotation_task": "250606_110200"},
    "val": {"movie_name": "blob_dwi_512", "annotation_task": "250610_211600"},
    "test": {"movie_name": "turb_dwi_512", "annotation_task": "250610_220000"},
}

MOVIE_REPO = HDF5Repository("tokam2d")
MODEL_REPO = LocalNNModelRepository("models")

COMPUTING_DEVICE = get_device()
print("Computing device:", COMPUTING_DEVICE)


def load_movie(movie_name: str) -> Movie:
    if movie_name not in MOVIE_TABLE:
        msg = (
            f"Unkown movie name: {movie_name}. "
            f"Allowed names: {sorted(MOVIE_TABLE)}"
        )
        raise ValueError(msg)
    movie = MOVIE_REPO.read(MOVIE_TABLE[movie_name])
    print(f"Loaded {len(movie)} frames.")
    return movie


def load_set(set_name: str) -> Movie:
    if set_name not in SET_TABLE:
        msg = f"Unknown set name: {set_name}. Allowed names: {list(SET_TABLE)}"
        raise ValueError(msg)

    movie_name = SET_TABLE[set_name]["movie_name"]
    annotation_task = SET_TABLE[set_name]["annotation_task"]

    # Load movie
    movie = load_movie(movie_name)

    # Load annotations
    annotation_path = (
        DISCOPATH / "annotations" / annotation_task / "annotated_movie.json"
    )
    with Path.open(annotation_path) as f:
        annotation_dict = {
            frame.name: frame.annotations
            for frame in Movie.from_dict(json.load(f)).frames
        }

    # Filter out non-annotated frames
    movie.frames = [
        frame for frame in movie.frames if frame.name in annotation_dict
    ]

    # Add annotations to frames
    for frame in movie.frames:
        frame.annotations = annotation_dict[frame.name]

    print(f"Loaded {len(movie)} annotated frames.")
    return movie


def load_model(model_name: str) -> Model:
    raw_model = MODEL_REPO.read(model_name)
    model = FasterRCNNModel.from_dict(raw_model)
    model.set_device(COMPUTING_DEVICE)

    return model
