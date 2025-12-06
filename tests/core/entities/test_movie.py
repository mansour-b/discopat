from discopat.core import Box, Frame, Movie, Track


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
                        Box(
                            label="blob", x=0, y=0, width=1, height=1, score=1.0
                        )
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

    def test_to_coco(self):
        movie = self.make_movie()
        expected = {
            "annotations": [
                {
                    "area": 1.0,
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "category_id": 1,
                    "id": 1,
                    "image_id": 1,
                    "iscrowd": 0,
                },
            ],
            "categories": [{"id": 1, "name": "blob"}],
            "images": [
                {"file_name": "0.png", "height": 512, "id": 1, "width": 512}
            ],
        }
        assert movie.to_coco() == expected
