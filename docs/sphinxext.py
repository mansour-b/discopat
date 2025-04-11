import sys
from pathlib import Path


def add_package_to_path(gallery_conf, fname):
    sys.path.insert(0, Path("..").resolve())
