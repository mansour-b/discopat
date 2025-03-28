import pytest

from patrick.core import ComputingDevice, DataSource, Framework


def test_computing_device():
    ComputingDevice("cpu")
    ComputingDevice("gpu")
    with pytest.raises(ValueError):
        ComputingDevice("tpu")


def test_data_source():
    DataSource("local")
    with pytest.raises(ValueError):
        DataSource("non-local")


def test_framework():
    Framework("torch")
    with pytest.raises(ValueError):
        Framework("tf")
