import numpy as np

from discopat.core import ComputingDevice, DataLoader, NeuralNet
from discopat.nn_training.evaluation.matching import (
    match_groundtruths_and_predictions,
)


def compute_ap(
    matching_dict: dict[str, dict[str, np.array]], threshold: float
) -> float:
    """Compute the Average Precision (AP) for a given localization threshold.

    Args:
        matching_dict: dictionary in the form:
            image_id: {
                "matching_matrix": array of shape (N_preds, N_gts),
                "scores": array of shape (N_preds,)
            }
        threshold: localization threshold,

    Returns:
        The AP.

    Note:
        The predictions and scores should already be sorted by descending score.

    """
    num_groundtruths = 0
    tp_vector_list = []
    score_vector_list = []

    for image_id in matching_dict:
        matching_matrix = matching_dict[image_id]["matching_matrix"]
        scores = matching_dict[image_id]["scores"]

        if matching_matrix.size == 0:
            continue

        _, num_gts = matching_matrix.shape

        matching_mask = (matching_matrix >= threshold).astype(float)

        score_weighted_matches = scores.reshape(-1, 1) * matching_mask
        max_indices = np.argmax(score_weighted_matches, axis=0)

        max_score_mask = np.zeros_like(matching_matrix)
        max_score_mask[max_indices, np.arange(num_gts)] = 1

        tp_vector = np.max(matching_mask * max_score_mask, axis=1)

        num_groundtruths += num_gts
        tp_vector_list.append(tp_vector)
        score_vector_list.append(scores)

    if num_groundtruths == 0:
        return 0

    big_tp_vector = np.concat(tp_vector_list)
    big_score_vector = np.concat(score_vector_list)

    # Sort the TP vector by decreasing prediction score over the whole dataset
    big_tp_vector = big_tp_vector[np.argsort(-big_score_vector)]

    # Cumulative sums
    tp_cumulative = np.cumsum(big_tp_vector)
    fp_cumulative = np.cumsum(1 - big_tp_vector)

    # Prepend zeros for the case score_threshold=1
    tp_cum = np.concatenate([[0], tp_cumulative])
    fp_cum = np.concatenate([[0], fp_cumulative])

    recall = tp_cum / num_groundtruths
    precision = tp_cum / (tp_cum + fp_cum + 1e-10)

    # Ensure precision is non-increasing
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Compute area under curve (AP)
    return np.trapezoid(precision, recall)


def evaluate(
    model: NeuralNet,
    data_loader: DataLoader,
    localization_criterion: str,
    device: ComputingDevice,
) -> dict[str, float]:
    """Evaluate a model on a data loader.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        localization_criterion: metric used for GT-pred matching,
        device: computing device on which the model is stored.

    Returns:
        A dict containing the name and values of the following metrics:
        AP50, AP[50:95:05].

    """
    model.eval()
    prediction_dict = {
        t["image_id"]: pred
        for images, targets in data_loader
        for pred, t in zip(
            model([img.to(device).float() for img in images]), targets
        )
    }
    matching_dict = {
        t["image_id"]: match_groundtruths_and_predictions(
            groundtruths=t["boxes"],
            predictions=prediction_dict[t["image_id"]]["boxes"],
            scores=prediction_dict[t["image_id"]]["scores"],
            localization_criterion=localization_criterion,
        )
        for _, targets in data_loader
        for t in targets
    }
    ap_dict = {
        f"AP{int(100 * threshold)}": compute_ap(matching_dict, threshold)
        for threshold in np.arange(0.5, 1.0, 0.05)
    }
    return {"AP50": ap_dict["AP50"], "AP": np.mean(list(ap_dict.values()))}
