from __future__ import annotations

from abc import ABC, abstractmethod

from typing_extensions import Self


class Metadata(ABC):
    """Abstract class that represents metadata (frames, annotations, etc.)."""

    @classmethod
    @abstractmethod
    def printable_fields(cls) -> list[str]:
        """List of the relevant fields to serialise the object."""

    @classmethod
    @abstractmethod
    def from_dict(cls, input_dict: dict) -> Self:
        """Make object from a dictionary."""

    def to_dict(self) -> dict:
        """Serialise object to a dictionary."""
        return {
            attribute: getattr(self, attribute)
            for attribute in self.printable_fields()
        }

    def __repr__(self):
        printable_dict = {
            attr: getattr(self, attr) for attr in self.printable_fields()
        }
        attribute_str = ", ".join(
            [f"{k}={v}" for k, v in printable_dict.items()]
        )
        return f"{type(self).__name__}({attribute_str})"

    def __str__(self):
        printable_dict = {
            attr: getattr(self, attr) for attr in self.printable_fields()
        }
        attribute_str = ",\n    ".join(
            [f"{k}={v}" for k, v in printable_dict.items()]
        )
        attribute_str += ","
        return f"{type(self).__name__}(\n    {attribute_str}\n)"

    def __eq__(self, other: Self) -> bool:
        return self.to_dict() == other.to_dict()
