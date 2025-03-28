from enum import Enum


class Framework(Enum):
    torch = "torch"


class DataSource(Enum):
    local = "local"
    osf = "osf"


class ComputingDevice(Enum):
    cpu = "cpu"
    gpu = "gpu"
