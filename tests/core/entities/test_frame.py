from patrick.core import Box, Frame, Keypoint


class TestFrame:
    def test_init(self):
        frame = Frame(name="frame_0", width=512, height=512, annotations=[])
        assert frame.name == "frame_0"
        assert frame.width == 512
        assert frame.height == 512
        assert frame.annotations == []

    def test_from_dict(self):
        frame_as_dict = {
            "name": "frame_0",
            "width": 512,
            "height": 512,
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
        }
        frame = Frame.from_dict(frame_as_dict)
        assert frame.name == "frame_0"
        assert frame.width == 512
        assert frame.height == 512
        assert frame.annotations == [
            Box(label="blob", score=1.0, x=0.0, y=0.0, width=1.0, height=1.0)
        ]

    def test_to_dict(self):
        frame = Frame(
            name="frame_0",
            width=512,
            height=512,
            annotations=[Box(label="blob", x=0, y=0, width=1, height=1, score=1)],
        )
        assert frame.to_dict() == {
            "name": "frame_0",
            "width": 512,
            "height": 512,
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
        }

    def test_resize(self):
        frame = Frame(
            name="frame_0",
            width=512,
            height=512,
            annotations=[
                Box(label="blob", x=1, y=1, width=1, height=1, score=1),
                Keypoint(label="blob", point_list=[(1, 1), (2, 2)], score=1),
            ],
        )
        frame.resize(target_width=1024, target_height=1536)
        assert frame == Frame(
            name="frame_0",
            width=1024,
            height=1536,
            annotations=[
                Box(label="blob", x=2, y=3, width=2, height=3, score=1),
                Keypoint(label="blob", point_list=[(2, 3), (4, 6)], score=1),
            ],
        )
