# load custom pytest fixtures
from .fixtures import *
import numpy as np
import hylite
from napari_hippo._sample_data import make_sample_image, make_sample_cloud

@pytest.fixture
def cloudMode(make_napari_viewer):
    """
    Build a napari viewer with some test data in Image mode
    """
    # make a viewer (using other fixture)
    def f():
        # make viewer
        viewer = make_napari_viewer()

        # add hypercloud widget to the dock
        from napari_hippo import HypercloudToolsWidget
        w = HypercloudToolsWidget(viewer)
        viewer.window.add_dock_widget(w)

        # add cloud layers
        args = make_sample_cloud()[0]
        layer = viewer.add_image( args[0], **args[1] )

        # return
        return viewer, layer
    return f

def test_point_size( cloudMode ):
    viewer, layer = cloudMode()
    
    from napari_hippo._hypercloudTools import setPointSize
    nbg1 = np.sum( np.isnan(layer.data) )
    setPointSize(3)
    nbg2 = np.sum( np.isnan(layer.data) )
    assert nbg1 > nbg2 # check that point size actually increased (reducing
                       # the total number of background (0) pixels )

    # viewer.show(block=True)

def test_extractData( cloudMode ):
    """
    Test extract data and extract ID functions.
    """
    viewer, layer = cloudMode()
    from napari_hippo._hypercloudTools import extractData, extractIDs
    extractData()

    viewer.layers.selection = [layer]
    extractIDs()

    viewer.show(block=True)

def test_locate( cloudMode ):
    """
    Test PnP solution to camera position.
    """
    viewer, layer = cloudMode()
    from napari_hippo._hypercloudTools import locate
    nbg1 = np.sum( np.isnan(layer.data) )
    locate()