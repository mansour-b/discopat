"""
Model Inference
===============

This example demonstrates how to do a sphinx gallery example.
"""

# %%
# Imports
# -------
import numpy as np

from patrick.core import Frame
from patrick.display import plot_frame

# %%
# Big big calculations
# --------------------

frame = Frame(
    name="hello",
    width=512,
    height=512,
    annotations=[],
    image_array=np.random.rand(),
)

plot_frame(frame)
