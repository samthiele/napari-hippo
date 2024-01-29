"""
Some test hyperspectral data for napari
"""
from __future__ import annotations
import napari_hippo.testdata
from napari_hippo import napari_get_ENVI_reader
def make_sample_data():
    """Generates an image and returns a list of tuples [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]"""
    r = napari_get_ENVI_reader( napari_hippo.testdata.image )
    return r(napari_hippo.testdata.image)