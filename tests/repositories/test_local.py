import pytest
from discopat.repositories.local import (
    DISCOPATH,
    LocalFrameRepository,
    LocalMovieRepository,
)


class TestLocalFrameRepository:
    def test_init(self):
        repo = LocalFrameRepository("input_frames")
        assert repo.name == "input_frames"
        assert repo._directory_path == DISCOPATH / "input"
        with pytest.raises(KeyError):
            LocalFrameRepository("input")

    def test_parse_frame_name(self):
        repo = LocalFrameRepository("input_frames")
        assert repo._parse_frame_name("blob/density_frame_20") == (
            "blob",
            "density",
            20,
        )


class TestLocalMovieRepository:
    def test_init(self):
        repo = LocalMovieRepository("input_movies")
        assert repo.name == "input_movies"
        assert repo._directory_path == DISCOPATH / "input"
        with pytest.raises(KeyError):
            LocalMovieRepository("input")

    def test_parse_movie_name(self):
        repo = LocalMovieRepository("input_movies")
        assert repo._parse_movie_name("blob/density") == ("blob", "density")
