# %%
# Imports
from discopat.core import Movie
from discopat.display import plot_frame
from discopat.repositories.hdf5 import HDF5Repository
from discopat.repositories.local import DISCOPATH


# %%
# Function definitions
def prepare_task(movie: Movie, annotation_task: str):
    output_path = DISCOPATH / "annotations" / annotation_task
    (output_path / "images").mkdir(parents=True, exist_ok=True)

    for frame in movie.frames:
        w, h = (frame.width, frame.height)
        print((w / 100, h / 100))
        fig = plot_frame(
            frame,
            figure_size=(w / 100, h / 100),
            figure_dpi=100,
            return_figure=True,
        )
        print(fig.get_size_inches())
        fig.savefig(
            output_path / "images" / frame.name,
            bbox_inches="tight",
            pad_inches=0,
            dpi=fig.dpi,
        )


# %%
# Definitions
simulation = "250610_103200"
annotation_task = "250610_211600"

# %% Load movie
movie_repo = HDF5Repository("tokam2d")
movie = movie_repo.read(simulation)
movie.frames = [frame for frame in movie.frames if int(frame.name) % 100 == 50]
print(len(movie.frames))

# %% Prepare task
prepare_task(movie, annotation_task)

# %%
