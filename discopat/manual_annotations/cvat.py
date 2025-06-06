# %%
import json
from pathlib import Path
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import parse

from discopat.core import Annotation, Box, Frame, Keypoint, Movie
from discopat.display import plot_frame
from discopat.manual_annotations.operations import turn_keypoints_into_boxes
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
        label=str(element.attrib["label"]),
        x=xmin,
        y=ymin,
        width=xmax - xmin,
        height=ymax - ymin,
        score=1.0,
    )


def xml_to_keypoint(element: Element) -> Keypoint:
    """Convert CVAT's polyline annotation to discopat's Keypoint."""
    info_dict = element.attrib

    label = str(info_dict["label"])

    point_list = str(info_dict["points"])
    point_list = point_list.split(";")
    point_list = [point.split(",") for point in point_list]
    point_list = [(float(point[0]), float(point[1])) for point in point_list]

    res = Keypoint(label=label, point_list=point_list, score=1.0)
    print(res)
    return res


def xml_to_annotation(element: Element) -> Annotation:
    if element.tag == "box":
        return xml_to_box(element)
    if element.tag == "keypoint":
        return xml_to_keypoint(element)
    raise ValueError(f"Unknown annotation type: {element.tag}")


def xml_to_frame(element: Element) -> Frame:
    """Make frames from CVAT annotations."""
    return Frame(
        name=str(element.attrib["name"].split(".")[0]),
        width=int(element.attrib["width"]) - 2 * w_padding,
        height=int(element.attrib["height"]) - 2 * h_padding,
        annotations=[
            xml_to_annotation(xml_annotation) for xml_annotation in element
        ],
    )


def xml_to_movie(element: Element) -> Movie:
    """Make movies from CVAT annotations."""
    return Movie(
        name=str(element.find("meta/task/name")),
        frames=[
            xml_to_frame(frame_xml) for frame_xml in element.findall("image")
        ],
        tracks=[],
    )


# %%
if __name__ == "__main__":
    simulation = "260605_164500"
    annotation_task = "250606_110200"

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
    # Turn all keypoints into boxes
    w_padding = 0.5
    h_padding = 0.5
    for frame in annotated_movie.frames:
        turn_keypoints_into_boxes(frame)

    # %%
    # Visual check
    for frame in annotated_movie.frames:
        plot_frame(frame)

    # %%
    # Write annotations in a file
    output_path = annotation_path.parent / "annotated_movie.json"
    with Path.open(output_path, "w") as f:
        json.dump(annotated_movie.to_dict(), f, indent=2)
