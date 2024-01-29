import napari
from napari_hippo import IOWidget
import pathlib
import os
import numpy as np


def test_coreg(make_napari_viewer, capsys):
    # test coregistration (at least run it)
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = IOWidget(viewer)
    viewer.window.add_dock_widget(w)

    # load test data
    from napari_hippo._ioTools import search
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.png',
                        rgb_only=True, stack=True )

    # load masks
    from napari_hippo._ioTools import loadMasks, updateMasks, saveMasks
    loadMasks()
    assert 'exclude' in viewer.layers, 'Error - masks could not create exclude polygons'
    assert 'include' in viewer.layers, 'Error - masks could not create include polygons'
    assert 'mask' in viewer.layers, 'Error - masks do not exist?'

    for i in range( len( viewer.layers['mask'].data ) ):
        # check masks are not all False
        arr = np.array(viewer.layers['mask'].data[i, ... ] )
        assert not (arr == 0).all(), 'Error - mask is empty?'

        #print(arr.shape, np.array(viewer.layers['block1'].data[i,...]).shape)
        assert arr.shape[0] == np.array(viewer.layers['block1'].data[i,...]).shape[0], 'Error - shape mismatch'
        assert arr.shape[1] == np.array(viewer.layers['block1'].data[i, ...]).shape[1], 'Error - shape mismatch'

    # updateMasks()
    updateMasks() # run function - todo: add better testing here?

    # save masks
    saveMasks()

    #viewer.show(block=True)