from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from typing_extensions import Self

from patrick.core.entities.annotation import Annotation
from patrick.core.entities.array import Array
from patrick.core.entities.frame import Frame
from patrick.core.value_objects import ComputingDevice


class Model(ABC):
    """Abstract class to represent a pattern detection model."""

    @abstractmethod
    def predict(self, frame: Frame) -> Frame:
        """Run the predictions of the model on a frame.

        Args:
            frame (Frame): Object representing the image or movie frame on
                which the model will be applied.

        Returns:
            Frame: The input frame where the model predictions have been
                appended to the list of already present annotations.

        """

    @classmethod
    @abstractmethod
    def from_dict(cls, model_as_dict: dict) -> Self:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass


class NeuralNet:
    """Abstract class to model a neural network."""

    @abstractmethod
    def __call__(self, input_array: Array) -> Any:
        """Compute the predictions of the net on an array.

        Args:
            input_array (Array): An array representing the input frame/image,
                compatible with the format of the neural network (np.ndarray,
                torch.Tensor, etc.).

        Returns:
            Depending on the neural network architecture, a tensor, a dict of
                tensors, or another sort of output.

        """


class NNModel(Model):
    """Abstract class representing neural-network-based models."""

    def __init__(
        self,
        net: NeuralNet,
        label_map: dict[str, int],
        model_parameters: dict,
    ):
        """Initialise the object.

        Args:
            net (NeuralNet): The neural network that runs under the hood.
            label_map (dict): Dictionary in format {"class_name": class_id}.
                The class ids should start from 1 since 0 is reserved for the
                background class.
            model_parameters (dict): Pre-/post-processing parameters such as:
                - The input channel format (channels_first or channels_last),
                - The prediction confindence score threshold,
                - Any other parameter on which the behavior of the model relies.

        """
        self.net = net
        self.label_map = label_map
        self.model_parameters = model_parameters

    def predict(self, frame: Frame) -> Frame:
        """Run the predictions of the model on a frame.

        The method follows the following scheme:
            - input_frame ------[pre-processing ]--> input_array,
            - input_array ------[      net      ]--> net_predictions,
            - net_predictions --[post_processing]--> output_frame.

        Args:
            frame (Frame): Object representing the image or movie frame on
                which the model will be applied.

        Returns:
            Frame: The input frame where the model predictions have been
                appended to the list of already present annotations.

        """
        input_array = self.pre_process(frame)
        predictions = self.net(input_array)
        annotations = self.post_process(predictions)
        return Frame(
            name=frame.name,
            width=frame.width,
            height=frame.height,
            annotations=annotations,
            image_array=frame.image_array,
        )

    @abstractmethod
    def pre_process(self, frame: Frame) -> Array:
        """Prepare the frame's array to pass through the neural network."""

    @abstractmethod
    def post_process(self, net_predictions: Any) -> list[Annotation]:
        """Convert the net's predictions into patrick's "Annotation" format."""

    @abstractmethod
    def set_device(self, device: ComputingDevice) -> None:
        pass


class CDModel(Model):
    """Abstract class representing convolutional-dictionary-based models."""
