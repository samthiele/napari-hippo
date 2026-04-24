import napari
from napari_hippo import BasicWidget
import pathlib
import os
import numpy as np
from .fixtures import *

def test_masking_stack(stackMode, capsys):
    # test coregistration (at least run it)
    # make viewer and add an image layer using our fixture
    viewer, layer = stackMode()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = BasicWidget(viewer)
    viewer.window.add_dock_widget(w)

    # load masks
    from napari_hippo._basicTools import loadMasks, saveMasks
    loadMasks()

    # check that there is a mask layer created
    assert np.any( [l.metadata['type'] == 'Mask' for l in viewer.layers] )
    
    from napari_hippo import getLayer, Mask
    for l in viewer.layers:
        if l.metadata['type'] == 'Mask':
            mask = getLayer(l)
            mask.layer.data[2:-2, :] = 1
            mask.layer.data[:, 2:-2] = 1
            mask.layer.data[10:12, :] = 0

    #viewer.show(block=True)

    # save masks 
    saveMasks()

    # apply them
    for crop in [False,True]:
        mask.apply(crop=crop)

    # delete mask files
    import glob
    masks = glob.glob( str(pathlib.Path(os.path.dirname(os.path.dirname(__file__)))/ 'testdata/*_mask.*') )
    for p in masks:
        os.remove( p )

    #viewer.show(block=True)

def test_masking_image(imageMode, capsys):
    # test coregistration (at least run it)
    # make viewer and add an image layer using our fixture
    viewer, layer = imageMode()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = BasicWidget(viewer)
    viewer.window.add_dock_widget(w)

    # load masks
    from napari_hippo._basicTools import loadMasks, saveMasks
    loadMasks(mode='directory') # test in directory mode now

    # check that there is a mask layer created
    assert np.any( [l.metadata['type'] == 'Mask' for l in viewer.layers] )
    
    # add some stuff to the mask
    from napari_hippo import getLayer, Mask
    for l in viewer.layers:
        if l.metadata['type'] == 'Mask':
            mask = getLayer(l)
            mask.layer.data[2:-2, :] = 1
            mask.layer.data[:, 2:-2] = 1
            mask.layer.data[10:12, :] = 0

    #viewer.show(block=True)

    # save masks 
    saveMasks(mode='Save to file')
    assert os.path.exists(str(pathlib.Path(os.path.dirname(os.path.dirname(__file__)))/ 'testdata/mask.hdr'))

    # apply them
    for mode in ['Set as nan', 'Nan and crop']:
        saveMasks(mode=mode)

    # delete mask files
    import glob
    masks = glob.glob( str(pathlib.Path(os.path.dirname(os.path.dirname(__file__)))/ 'testdata/mask.*') )
    for p in masks:
        os.remove( p )

    #viewer.show(block=True)
