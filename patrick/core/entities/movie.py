from __future__ import annotations

from typing_extensions import Self

from patrick.core.entities.annotation import Track
from patrick.core.entities.frame import Frame
from patrick.core.entities.metadata import Metadata


class Movie(Metadata):
    def __init__(self, name: str, frames: list[Frame], tracks: list[Track]):
        self.name = name
        self.frames = frames
        self.tracks = tracks

    @classmethod
    def printable_fields(cls) -> list[str]:
        return ["name", "frames", "tracks"]

    def to_dict(self) -> dict:
        output = super().to_dict()
        output["frames"] = [frame.to_dict() for frame in self.frames]
        output["tracks"] = [track.to_dict() for track in self.tracks]
        return output

    @classmethod
    def from_dict(cls, data_as_dict: dict) -> Self:
        return cls(
            name=data_as_dict["name"],
            frames=[
                Frame.from_dict(frame_as_dict)
                for frame_as_dict in data_as_dict["frames"]
            ],
            tracks=[
                Track.from_dict(track_as_dict)
                for track_as_dict in data_as_dict["tracks"]
            ],
        )
