from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Protocol,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

X_co = TypeVar("X_co", covariant=True)
Y_co = TypeVar("Y_co", covariant=True)


@runtime_checkable
class Dataset(Protocol[X_co, Y_co]):
    """A dataset providing indexed access to samples."""

    def __getitem__(self, index: int) -> tuple[X_co, Y_co]:
        """Return a single sample (X, y) at the given index."""
        ...

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        ...


@runtime_checkable
class DataLoader(Protocol[X_co, Y_co]):
    """Generic interface for objects yielding (X, y) pairs."""

    def __iter__(self) -> Iterator[tuple[X_co, Y_co]]: ...

    def __len__(self) -> int:
        """Return the number of batches."""
        ...

    @property
    def batch_size(self) -> int | None:
        """Return the number of samples per batch, if known."""
        ...

    @property
    def dataset(self) -> Dataset[X_co, Y_co]:
        """Return the underlying dataset from which batches are drawn."""
        ...
