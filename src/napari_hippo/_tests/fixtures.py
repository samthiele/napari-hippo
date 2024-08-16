"""
Define some useful pytest fixtures that instantiate a napari viewer with different data / layers.
"""
import os
import pathlib
import pytest
import napari

@pytest.fixture
def imageMode(make_napari_viewer):
    """
    Build a napari viewer with some test data in Image mode
    """
    # make a viewer (using other fixture)
    def f():
        viewer = make_napari_viewer()

        # add test data cube
        from napari_hippo._basicTools import search
        search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/image.hdr',
                            rgb_only=False, stack=False )

        return viewer, viewer.layers[0]
    return f

@pytest.fixture
def stackMode(make_napari_viewer):
    """
    Build a napari viewer with some test data in Image mode
    """
    # make a viewer (using other fixture)
    def f():
        viewer = make_napari_viewer()

        # add test data cube
        from napari_hippo._basicTools import search
        search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.png',
                            rgb_only=True, stack=True )
        return viewer, viewer.layers[0]
    return f
