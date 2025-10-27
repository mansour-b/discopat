import time

import numpy as np

from discopat.core import Array, ComputingDevice, DataLoader, NeuralNet
from discopat.evaluation.matching import (
    match_groundtruths_and_predictions,
)


def compute_ap(
    prediction_dict: dict[str, Array],
    data_loader: DataLoader,
    threshold: float,
    localization_criterion: str,
    device: ComputingDevice,
) -> float:
    """Compute the Average Precision (AP) for a given localization threshold.

    Args:
        model: the neural network to be evaluated,
        data_loader: the evaluation dataloader,
        threshold: localization threshold,
        localization_criterion: metric used to match groundtruths and predictions,
        device: computing device on which the model is stored.

    Returns:
        The AP.

    """
    num_groundtruths = 0
    big_tp_vector = np.empty((0, 2))
    for _, targets in data_loader:
        outputs = [prediction_dict[t["image_id"]] for t in targets]
        print("    Transfering outputs to CPU...")
        start = time.process_time()
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        end = time.process_time()
        print(f"    Done in {end - start:.2f} seconds.")
        print()
        for target, output in zip(targets, outputs):
            print("        Matching...")
            start = time.process_time()
            num_gts, tp_vector = match_groundtruths_and_predictions(
                groundtruths=target["boxes"],
                predictions=output["boxes"],
                scores=output["scores"],
                threshold=threshold,
                localization_criterion=localization_criterion,
            )
            end = time.process_time()
            print(f"        Done in {end - start:.2f} seconds.")
            num_groundtruths += num_gts
            big_tp_vector = np.concat((big_tp_vector, tp_vector))
        print()
    if num_groundtruths == 0:
        return 0

    # Sort the TP vector by decreasing prediction score over the whole dataset
    big_tp_vector = big_tp_vector[np.argsort(-big_tp_vector[:, 0])]

    # Cumulative sums
    tp_cumulative = np.cumsum(big_tp_vector[:, 1])
    fp_cumulative = np.cumsum(1 - big_tp_vector[:, 1])

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
    print()
    print("===")
    print("Compute predictions...")
    start = time.process_time()
    prediction_dict = {
        t["image_id"]: pred
        for images, targets in data_loader
        for pred, t in zip(
            model([img.to(device).float() for img in images]), targets
        )
    }
    end = time.process_time()
    print(f"Done in {end - start:.2f} seconds.")
    print()
    print("Build AP dict...")
    start = time.process_time()
    ap_dict = {
        f"AP{int(100 * threshold)}": compute_ap(
            prediction_dict,
            data_loader,
            threshold=threshold,
            localization_criterion=localization_criterion,
            device=device,
        )
        for threshold in np.arange(0.5, 1.0, 0.05)
    }
    end = time.process_time()
    print(f"Done in {end - start:.2f} seconds.")
    print("===")
    print()
    return {"AP50": ap_dict["AP50"], "AP": np.mean(list(ap_dict.values()))}
