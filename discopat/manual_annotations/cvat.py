# %%
from pathlib import Path
from xml.etree.ElementTree import Element

import numpy as np
from defusedxml.ElementTree import parse

from discopat.core import Box, Frame, Movie
from discopat.display import plot_frame
from discopat.repositories.hdf5 import HDF5Repository
from discopat.repositories.local import DISCOPATH

# %%
# Function definitions
# I messed up:
# I let VSCode automatically determine the figure size
# -> I exported 370x370 images
# Then I did not remove the margins of the saved figure
# -> savefig added white margins in order to save 640x480 images
# Now I have to compute these paddings and substract them from the coordinates.
w_padding = (640 - 370) / 2
h_padding = (480 - 370) / 2


def xml_to_box(element: Element) -> Box:
    """Convert CVAT's box annotation to discopat's Box."""
    info_dict = element.attrib
    xmin = float(info_dict["xtl"]) - w_padding
    ymin = float(info_dict["ytl"]) - h_padding
    xmax = float(info_dict["xbr"]) - w_padding
    ymax = float(info_dict["ybr"]) - h_padding

    return Box(
        label=element.attrib["label"],
        x=xmin,
        y=ymin,
        width=xmax - xmin,
        height=ymax - ymin,
        score=1.0,
    )


def xml_to_frame(element: Element) -> Frame:
    """Make frames from CVAT annotations."""
    return Frame(
        name=element.attrib["name"].split(".")[0],
        width=int(element.attrib["width"]) - 2 * w_padding,
        height=int(element.attrib["height"]) - 2 * h_padding,
        annotations=[xml_to_box(xml_bbox) for xml_bbox in element],
    )


def xml_to_movie(element: Element) -> Movie:
    """Make movies from CVAT annotations."""
    return Movie(
        name=element.find("meta/task/name"),
        frames=[
            xml_to_frame(frame_xml) for frame_xml in element.findall("image")
        ],
        tracks=[],
    )


# %%
if __name__ == "__main__":
    simulation = "250603_105600"
    annotation_task = "250604_115800"

    # %%
    # Load annotations
    annotation_path = (
        DISCOPATH / "annotations" / annotation_task / "annotations.xml"
    )
    tree = parse(annotation_path)
    root = tree.getroot()

    annotated_movie = xml_to_movie(root)

    # %%
    # Load frames
    movie_repository = HDF5Repository("tokam2d")
    movie = movie_repository.read(simulation)

    num_frames = len(movie.frames)
    width = movie.frames[0].width
    height = movie.frames[0].height
    w = movie.frames[0].image_array.shape[1]
    h = movie.frames[0].image_array.shape[0]

    print(num_frames, width, w, height, h)

    image_dict = {frame.name: frame.image_array for frame in movie.frames}

    # %%
    # Display frames with annotations
    for frame in annotated_movie.frames:
        frame.resize(target_width=width, target_height=height)
        frame.image_array = image_dict[frame.name]
        plot_frame(frame)

# %%
