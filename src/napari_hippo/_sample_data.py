"""
Some test hyperspectral data for napari
"""
from __future__ import annotations
import napari_hippo.testdata
from napari_hippo import HSICube, View
from hylite import io
def make_sample_image():
    """Generates an image and returns a list of tuples [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]"""
    #r = napari_get_ENVI_reader( napari_hippo.testdata.image )
    return [HSICube.construct( io.load(napari_hippo.testdata.image), 
                             '[HSICube] test', 
                             args_only=True)]

def make_sample_cloud():
    """Generates an image and returns a list of tuples [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]"""
    #r = napari_get_ENVI_reader( napari_hippo.testdata.image )
    cloud = io.load(napari_hippo.testdata.cloud)
    cam = cloud.header.get_camera()
    return [View.construct( cloud, 
                             name='cloud', 
                             camera=cam,
                             args_only=True,
                             path=napari_hippo.testdata.cloud)]
