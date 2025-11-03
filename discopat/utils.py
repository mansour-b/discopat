import torch


def get_device(allow_mps: bool = False):
    if torch.cuda.is_available():
        return "cuda"
    if not allow_mps:
        return "cpu"
    if torch.mps.is_available():
        return "mps"
    return "cpu"
