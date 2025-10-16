import numpy as np

from discopat.metrics import compute_ap_at_threshold


class TestMetrics:
    def test_compute_ap_at_threshold(self):
        # 2 ground truths
        groundtruths = [[0, 0, 1, 1]]  # , [2, 2, 3, 3]]

        # 3 predictions
        predictions = [[0, 0, 1, 1, 0.9]]  # ,  # perfect match with gt[0]
        #     [2, 2, 3, 3, 0.8],  # perfect match with gt[1]
        #     [4, 4, 5, 5, 0.7],  # no match
        # ]

        ap50 = compute_ap_at_threshold(groundtruths, predictions, threshold=0.5)
        assert np.isclose(ap50, 1)
