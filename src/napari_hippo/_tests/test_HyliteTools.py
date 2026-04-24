import os
import pathlib
import hylite
from .fixtures import *
import time
import numpy as np

@pytest.fixture
def hyliteMode(imageMode):
    """
    Build a napari viewer with some test data in Image mode
    """
    # make a viewer (using other fixture)
    def f():
        # make viewer and add an image layer using our fixture
        viewer, layer = imageMode() # get viewer and added image layer from fixture
        
        # create our widget, passing in the viewer, and add it to napari as a dock
        from napari_hippo import HyliteToolsWidget
        w = HyliteToolsWidget(viewer)
        viewer.window.add_dock_widget(w)

        time.sleep(0.1) # give some time for threads to sync (prevents strange bugs?)

        # return
        return viewer, layer
    return f

def test_Calculate(hyliteMode, capsys):
    # make viewer and add an image layer using our fixture
    viewer, layer = hyliteMode() # get viewer and added image layer from fixture

    # check calculator
    from napari_hippo._hyliteTools import calculate
    viewer.layers.selection.clear()
    viewer.layers.selection.add(layer) # select image layer
    res = calculate( bands = "%d | %d | %d" % hylite.SWIR ) # extract an RGB composite
    assert res[0].metadata['bands'] == 3
    assert res[0].metadata['type'] == 'RGB'

    # check stretch
    from napari_hippo._hyliteTools import stretch
    for method in ['Percent clip', 'Percent clip (per band)']:
        stretch(method=method)
    stretch(method='Absolute', vmin=0.1, vmax=0.9)

    #viewer.show(block=True)

def test_hullCorrect( hyliteMode, capsys ):
    # make viewer and add an image layer using our fixture
    viewer, layer = hyliteMode() # get viewer and added image layer from fixture

    from napari_hippo._hyliteTools import hullCorrect
    hullCorrect()
    assert '[HSICube] image(hc)' in viewer.layers
    assert np.nanmax(viewer.layers['[HSICube] image(hc)'].data) <= 1.01
    #viewer.show(block=True)

def test_dimReduce( hyliteMode, capsys ):
    # make viewer and add an image layer using our fixture
    viewer, layer = hyliteMode() # get viewer and added image layer from fixture

    from napari_hippo._hyliteTools import dimensionReduction
    dimensionReduction()
    assert '[HSICube] image(PCA)' in viewer.layers
    assert viewer.layers['[HSICube] image(PCA)'].metadata['bands'] == 5
    viewer.show(block=True)

def test_combine( hyliteMode, capsys):
    viewer, layer = hyliteMode() # get viewer and added image layer from fixture
    from napari_hippo._hyliteTools import combine
    from napari_hippo._base import HSICube
    
    # create a second image and select it
    img2 = HSICube(layer).toHyImage()
    img2.data = img2.data*2
    I2 = HSICube.construct( img2, 'image2', viewer=viewer)
    viewer.layers.select_all()

    # test function
    combine()

    # check it worked
    if '[HSICube] image(median)' in viewer.layers:
        l2 = viewer.layers['[HSICube] image(median)']
    elif '[HSICube] image2(median)' in viewer.layers:
        l2 = viewer.layers['[HSICube] image2(median)']
    else:
        assert False, "No output?"
    
    A = HSICube(l2).toHyImage()
    delta = np.nanmean( A.data / img2.data )
    assert np.abs( delta - 0.75) < 1e-6

    #viewer.show(block=True)
