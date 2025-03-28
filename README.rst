.. image:: https://github.com/mansour-b/pattern-discovery-kit/blob/main/assets/patrick_hollywood.png
   :align: center

PATteRn dIsCovery Kit (PATRICK)
-------------------------------

|Build Status| |Code Coverage|

Welcome to the Pattern Discovery Kit!

This library provides tools to discover, detect, and track meaningful patterns
in physical signals. These signals can be of various forms:

1. Time series,
2. Images,
3. Movies,
4. Any other type of n-dimensional data.

Installation
------------

You can install the ``pattern-discovery-kit`` from the source code by doing the following::

    pip install pattern-discovery-kit

You can then try running `this notebook
<https://github.com/mansour-b/pattern-discovery-kit/blob/main/examples/model_inference.ipynb>`_
on your computer to verify that the installation was succesful.

Quickstart
----------

Here is an example to briefly present the API:

.. code:: python

    import numpy as np

    from patrick.core import Box, Frame, Model, Movie, Tracker
    from patrick.display import plot_frame

    # Define the dimensions of the problem
    frame_width = 5
    frame_height = 5
    movie_length = 3
    gif_frames_per_second = 2

    # Define a concrete model class, just for the example
    class DumbModel(Model):
        def predict(self, frame: Frame) -> Frame:
            frame_id = int(frame.name)
            frame.annotations.append(
                Box(label="noise_in_a_square", x=frame_id, y=frame_id, width=1, height=1)
            )
            return frame

    model = DumbModel()

    # Our data for this short tutorial
    frames = [
        Frame(
            name=str(10 * i),
            width=frame_width,
            height=frame_height,
            annotations=[],
            image_array=np.random.random(frame_height, frame_width),
        )
        for i in range(movie_length)
    ]

    movie = Movie(name="some_noise", frames=frames, tracks=[])

    # Run the detection model on individual frames
    analysed_frames = [model(frame) for frame in movie.frames]
    analysed_movie = Movie(
        name="some_noise_with_boxes", frames=analysed_frames, tracks=[]
    )

    # TBD: run tracker on detections
    analysed_movie = tracker.make_tracks(analysed_movie)
    analysed_movie.name = "some_noise_with_boxes_and_tracks"

    # Plot individual frames with detections
    for frame in analysed_movie:
        plot_frame(frame)

    # TBD: make a GIF to show the tracks
    export_to_gif(analysed_movie, fps=gif_frames_per_seconds)

.. |Build Status| image:: https://github.com/mansour-b/patrick/actions/workflows/pytest.yaml/badge.svg
   :target: https://github.com/mansour-b/patrick/actions/workflows/pytest.yaml

.. |Code Coverage| image:: https://codecov.io/github/mansour-b/patrick/graph/badge.svg?token=E37XYKWFWT
   :target: https://codecov.io/github/mansour-b/patrick