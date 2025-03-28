from patrick.core import Box, Keypoint


class TestBox:
    def test_init(self):
        box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
        assert box.label == "blob"
        assert box.x == 0
        assert box.y == 0
        assert box.width == 1
        assert box.height == 1
        assert box.score == 1

    def test_xyxy(self):
        box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
        assert box.xmin == 0
        assert box.ymin == 0
        assert box.xmax == 1
        assert box.ymax == 1

    def test_from_dict(self):
        box_as_dict = {
            "type": "box",
            "label": "blob",
            "x": 0,
            "y": 0,
            "width": 1,
            "height": 1,
            "score": 1,
        }
        box = Box.from_dict(box_as_dict)
        assert box.label == "blob"
        assert box.x == 0
        assert box.y == 0
        assert box.width == 1
        assert box.height == 1
        assert box.score == 1

    def test_to_dict(self):
        box = Box(label="blob", x=0, y=0, width=1, height=1, score=1)
        box_as_dict = box.to_dict()

        assert box_as_dict == {
            "type": "box",
            "label": "blob",
            "x": 0,
            "y": 0,
            "width": 1,
            "height": 1,
            "score": 1,
        }

    def test_rescale(self):
        box = Box(label="blob", x=1, y=1, width=1, height=1, score=1)
        box.rescale(w_ratio=2, h_ratio=3)
        assert box.label == "blob"
        assert box.x == 2
        assert box.y == 3
        assert box.width == 2
        assert box.height == 3
        assert box.score == 1

    def test_repr(self):
        box = Box(label="blob", x=1, y=1, width=1, height=1, score=1)
        assert (
            repr(box) == "Box("
            "label=blob, "
            "score=1.0, "
            "x=1.0, "
            "y=1.0, "
            "width=1.0, "
            "height=1.0)"
        )

    def test_str(self):
        box = Box(label="blob", x=1, y=1, width=1, height=1, score=1)
        assert str(box) == "\n".join(
            [
                "Box(",
                "    label=blob,",
                "    score=1.0,",
                "    x=1.0,",
                "    y=1.0,",
                "    width=1.0,",
                "    height=1.0,",
                ")",
            ]
        )


class TestKeypoint:
    def test_init(self):
        keypoint = Keypoint(label="blob", point_list=[(0, 0), (1, 1)], score=1.0)
        assert keypoint.label == "blob"
        assert keypoint.point_list == [(0, 0), (1, 1)]
        assert keypoint.score == 1

    def test_from_dict(self):
        keypoint_as_dict = {
            "type": "keypoint",
            "label": "blob",
            "point_list": [(0, 0), (1, 1)],
            "score": 1,
        }
        keypoint = Keypoint.from_dict(keypoint_as_dict)
        assert keypoint.label == "blob"
        assert keypoint.point_list == [(0, 0), (1, 1)]
        assert keypoint.score == 1

    def test_to_dict(self):
        keypoint = Keypoint(label="blob", point_list=[(0, 0), (1, 1)], score=1)
        keypoint_as_dict = keypoint.to_dict()
        assert keypoint_as_dict == {
            "type": "keypoint",
            "label": "blob",
            "point_list": [(0, 0), (1, 1)],
            "score": 1,
        }

    def test_rescale(self):
        keypoint = Keypoint(label="blob", point_list=[(1, 1), (2, 2)], score=1)
        keypoint.rescale(w_ratio=2, h_ratio=3)
        assert keypoint.label == "blob"
        assert keypoint.point_list == [(2, 3), (4, 6)]
        assert keypoint.score == 1

    def test_repr(self):
        keypoint = Keypoint(label="blob", point_list=[(0, 0), (1, 1)], score=1)
        assert (
            repr(keypoint)
            == "Keypoint(label=blob, score=1.0, point_list=[(0.0, 0.0), (1.0, 1.0)])"
        )

    def test_str(self):
        keypoint = Keypoint(label="blob", point_list=[(0, 0), (1, 1)], score=1)
        assert str(keypoint) == "\n".join(
            [
                "Keypoint(",
                "    label=blob,",
                "    score=1.0,",
                "    point_list=[(0.0, 0.0), (1.0, 1.0)],",
                ")",
            ]
        )
