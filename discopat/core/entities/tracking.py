from __future__ import annotations

from abc import ABC, abstractmethod

from discopat.core import Movie, Track


class Tracker(ABC):
    @abstractmethod
    def make_tracks(self, movie: Movie) -> list[Track]:
        pass


class SORTTracker(ABC):
    pass
