from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Self

from patrick.core.entities.annotation import Annotation, annotation_dict_factory
from patrick.core.entities.metadata import Metadata

if TYPE_CHECKING:
    from patrick.core.entities.array import Array


class Frame(Metadata):
    """Class to model movie frames or images."""

    def __init__(
        self,
        name: str,
        width: int,
        height: int,
        annotations: list[Annotation],
        image_array: Array = None,
    ):
        """Initialise the frame object.

        Args:
            name (str): Identifier of the frame. Can be an index, a file path, etc.
            width (int): Width of the image.
            height (int): Height of the image.
            annotations (list): Annotations or detections associated with the image.
            image_array (Array): 2D image corresponding to the frame

        """
        self.name = name
        self.width = width
        self.height = height
        self.annotations = annotations
        self.image_array = image_array

    @classmethod
    def printable_fields(cls) -> list[str]:
        """List of the relevant fields to serialise the object."""
        return ["name", "width", "height", "annotations"]

    @classmethod
    def from_dict(cls, data_as_dict: dict) -> Self:
        """Make object from a dictionary."""
        return cls(
            name=data_as_dict["name"],
            width=data_as_dict["width"],
            height=data_as_dict["height"],
            annotations=[
                annotation_dict_factory(annotation_as_dict)
                for annotation_as_dict in data_as_dict["annotations"]
            ],
        )

    def to_dict(self) -> dict:
        """Serialise object to a dictionary."""
        output = super().to_dict()
        output["annotations"] = [
            annotation.to_dict() for annotation in self.annotations
        ]
        return output

    def resize(self, target_width: int, target_height: int) -> None:
        """Resize the image to (target_width, target_height)."""
        w_ratio = target_width / self.width
        h_ratio = target_height / self.height

        self.width = target_width
        self.height = target_height

        for annotation in self.annotations:
            annotation.rescale(w_ratio, h_ratio)
