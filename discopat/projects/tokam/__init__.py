from discopat.core import Movie
from discopat.repositories.hdf5 import HDF5Repository

MOVIE_TABLE = {
    "blob_dwi_512": "250610_103200",
    "blob_i_512": "250605_164500",
    "turb_dwi_256": "250603_111000",
    "turb_dwi_512": "250610_110800",
    "turb_i_256": "250603_105600",
    "turb_i_512": "250715_150500",
}

MOVIE_REPO = HDF5Repository("tokam2d")


def load_movie(movie_name: str) -> Movie:
    if movie_name not in MOVIE_TABLE:
        msg = (
            f"Unkown movie name: {movie_name}. "
            f"Allowed names: {sorted(MOVIE_TABLE)}"
        )
        raise ValueError(msg)
    return MOVIE_REPO.read(MOVIE_TABLE[movie_name])
