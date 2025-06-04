# %%
from pathlib import Path
from xml.etree.ElementTree import Element

from defusedxml.ElementTree import parse

from discopat.core import Box, Frame
from discopat.repositories.local import DISCOPATH

# %%
data_path = DISCOPATH / "annotations" / "250604_115800" / "annotations.xml"

# %%
tree = parse(data_path)
root = tree.getroot()


# %%
def xml_to_box(element: Element) -> Box:
    """Convert CVAT's box annotation to discopat's Box."""
    info_dict = element.attrib
    xmin = float(info_dict["xtl"])
    ymin = float(info_dict["ytl"])
    xmax = float(info_dict["xbr"])
    ymax = float(info_dict["ybr"])

    return Box(
        label=element.attrib["label"],
        x=xmin,
        y=ymin,
        width=xmax - xmin,
        height=ymax - ymin,
        score=1.0,
    )


def xml_to_frame(element: Element) -> Frame:
    return Frame(
        name=element.attrib["name"].split(".")[0],
        width=int(element.attrib["width"]),
        height=int(element.attrib["height"]),
        annotations=[xml_to_box(xml_bbox) for xml_bbox in element],
    )


# %%
for frame_xml in root.findall("image"):
    print("===")
    print(xml_to_frame(frame_xml))
    print()
# %%
for box in xml_to_frame(frame_xml).annotations:
    print(box)
# %%
