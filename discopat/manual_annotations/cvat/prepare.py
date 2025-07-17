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
        fig = plot_frame(
            frame,
            figure_size=(w / 100, h / 100),
            figure_dpi=100,
            return_figure=True,
        )
        fig.savefig(
            output_path / "images" / frame.name,
            bbox_inches="tight",
            pad_inches=0,
            dpi=fig.dpi,
        )


# %%
# Definitions
simulation = "250610_110800"
annotation_task = "250610_220000"

# %% Load movie
movie_repo = HDF5Repository("tokam2d")
movie = movie_repo.read(simulation)
movie.frames = [
    frame
    for frame in movie.frames
    if (int(frame.name) >= 1200) & (int(frame.name) % 100 == 50)
]
print(len(movie.frames))

# %%
# Prepare task
prepare_task(movie, annotation_task)

# %%
for frame in movie.frames:
    plot_frame(frame, cmap="inferno")

# %%
