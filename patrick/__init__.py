from pathlib import Path

from patrick.entities.annotation import Annotation, Box, Keypoint, Track
from patrick.entities.detection import Model, NNModel
from patrick.entities.frame import Frame
from patrick.entities.movie import Movie
from patrick.entities.tracking import SORTTracker, Tracker

DATA_DIR_PATH = Path.home() / "data"
PATRICK_DIR_PATH = DATA_DIR_PATH / "pattern_detection_tokam"
TOKAM_DIR_PATH = DATA_DIR_PATH / "tokam2d"

__all__ = [
    "Annotation",
    "Box",
    "Frame",
    "Keypoint",
    "Model",
    "Movie",
    "NNModel",
    "SORTTracker",
    "Track",
    "Tracker",
]
