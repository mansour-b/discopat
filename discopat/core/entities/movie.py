from __future__ import annotations

from typing_extensions import Self

from discopat.core.entities.annotation import Track
from discopat.core.entities.frame import Frame
from discopat.core.entities.metadata import Metadata


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

    def __len__(self):
        return len(self.frames)

    def to_coco(self, image_file_extension: str = "png") -> dict:
        """Convert movie to COCO annotation format."""
        images = []
        annotations = []
        categories = {}

        annotation_counter = 1
        category_counter = 1

        for i, frame in enumerate(self.frames):
            image_counter = i + 1
            images.append(
                {
                    "id": image_counter,
                    "file_name": f"{frame.name}.{image_file_extension}",
                    "width": frame.width,
                    "height": frame.height,
                }
            )
            for box in frame.annotations:
                if box.type != "box":
                    continue
                if box.label not in categories:
                    categories[box.label] = category_counter
                    category_counter += 1
                annotations.append(
                    {
                        "id": annotation_counter,
                        "image_id": image_counter,
                        "category_id": categories[box.label],
                        "bbox": [box.x, box.y, box.width, box.height],
                        "area": box.width * box.height,
                        "iscrowd": 0,
                    }
                )
                annotation_counter += 1
        return {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": v, "name": k} for k, v in categories.items()],
        }
