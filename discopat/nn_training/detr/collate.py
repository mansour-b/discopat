from discopat.nn_training.detr.nested_tensor import (
    nested_tensor_from_tensor_list,
)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
