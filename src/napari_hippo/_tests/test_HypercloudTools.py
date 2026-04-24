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

    #viewer.show(block=True)

def test_locate( cloudMode ):
    """
    Test PnP solution to camera position.
    """
    viewer, layer = cloudMode()
    from napari_hippo._hypercloudTools import locate, extractIDs
    from napari_hippo import ROI
    ids = extractIDs() # add an IDs layer to quickly get point ids
    image = ids.toHyImage()

    # add a keypoints layer
    kps,labels = [(0,0)],["fubar"] # add some text labels to check this doesn't mess things up
    np.random.seed(42)
    for n in range(500):
        _x = int(np.random.uniform(0, image.xdim()))
        _y = int(np.random.uniform(0, image.ydim()))
        pid = image.data[_x,_y,0]
        if np.isfinite(pid) and (pid!=0):
            kps.append((_x+np.random.rand()*3, 
                        _y+np.random.rand()*3))
            labels.append( '%d'%pid  )
    
    assert len(kps) > 4 # unlikely but not impossible

    points = ROI.construct( ids.layer, mode = 'point', viewer=viewer )
    points.fromList( kps, labels, world=True, transpose=True)
    
    result = locate( layer, points.layer, projection='panoramic', 
           ifov=0.084, refine_method='None' )
    assert result.metadata['residual'] < 2 # check we're close


    viewer.show(block=True)