from patrick.core import Box, Frame, Movie, Track


class TestMovie:
    @staticmethod
    def make_movie():
        return Movie(
            name="blob_movie",
            frames=[
                Frame(
                    name="0",
                    width=512,
                    height=512,
                    annotations=[
                        Box(label="blob", x=0, y=0, width=1, height=1, score=1.0)
                    ],
                ),
            ],
            tracks=[],
        )

    def test_init(self):
        movie = Movie(name="blob_movie", frames=[], tracks=[])
        assert movie.name == "blob_movie"
        assert movie.frames == []
        assert movie.tracks == []

    def test_to_dict(self):
        movie = self.make_movie()

        assert movie.to_dict() == {
            "frames": [
                {
                    "annotations": [
                        {
                            "height": 1.0,
                            "label": "blob",
                            "score": 1.0,
                            "type": "box",
                            "width": 1.0,
                            "x": 0.0,
                            "y": 0.0,
                        },
                    ],
                    "height": 512,
                    "name": "0",
                    "width": 512,
                },
            ],
            "name": "blob_movie",
            "tracks": [],
        }

    def test_from_dict(self):
        movie_as_dict = {
            "frames": [
                {
                    "annotations": [
                        {
                            "height": 1.0,
                            "label": "blob",
                            "score": 1.0,
                            "type": "box",
                            "width": 1.0,
                            "x": 0.0,
                            "y": 0.0,
                        },
                    ],
                    "height": 512,
                    "name": "0",
                    "width": 512,
                },
            ],
            "name": "blob_movie",
            "tracks": [],
        }
        assert Movie.from_dict(movie_as_dict) == self.make_movie()
