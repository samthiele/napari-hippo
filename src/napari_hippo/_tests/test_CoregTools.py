import os
import pathlib
import hylite
from .fixtures import *
import time
import numpy as np
import napari_hippo
from pathlib import Path
from napari_hippo import getLayer, HSICube

@pytest.fixture
def coregMode(make_napari_viewer):
    """
    Build a napari viewer with some test data in Image mode
    """
    # make a viewer (using other fixture)
    def f():
        # make viewer
        viewer = make_napari_viewer()

        # add coreg widget to the dock
        from napari_hippo import CoregToolsWidget
        w = CoregToolsWidget(viewer)
        viewer.window.add_dock_widget(w)

        # add image layers

        from napari_hippo._basicTools import search
        pth = Path(os.path.join( os.path.dirname(napari_hippo.testdata.image) ))
        
        layers = search(root=pth, filter='block2.png',
               stack=False) # this should load 2 images

        # duplicate it with a small offset and some rotations / flips
        image = getLayer( layers[0] ).toHyImage()
        image.data = image.data[5:, 5:, : ]
        image.data = image.data[::-1, ::-1, :]
        image.rot90()
        HSICube.construct( image, 'block1', viewer=viewer )

        time.sleep(0.1) # give some time for threads to sync (prevents strange bugs?)

        # return
        return viewer, list(viewer.layers)
    return f

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_manualCoreg(coregMode, capsys):
    # make viewer and add an image layer using our fixture
    viewer, layers = coregMode()
    
    # while we're here, check basic coreg tools work (and get rough alignment of images)
    from napari_hippo._coregTools import scale, rot, translate, simpleT
    simpleT() # build GUI
    rot(90)
    scale(-1,1)
    translate(layers[0].metadata['xdim'],
              layers[0].metadata['ydim'],
              relative=False)
    
    p0 = layers[0].data_to_world([10,10])
    p1 = layers[1].world_to_data([10,10])
    assert np.sqrt(np.sum((p1-p0)**2)) < 30 # check approximate alignment now

    # add manual keypoints
    from napari_hippo._coregTools import addKP, matchKP
    kpl = addKP()
    for p in [(0,0),(0,10),(10,10),(10,0)]:
        kpl[0].add(p)
    for p in [(0,10),(10,0),(0,0),(10,10), (25,5)]:
        kpl[1].add(np.array(p)+10) 
    # N.B. we intentionally add one too many keypoints to this 
    # second layer for the additional challenge....

    viewer.layers.select_all()
    matchKP()
    
    # add a single final point and check that doesn't break everything
    kpl[0].add((25,5))
    matchKP()

    for l in kpl:
        for t in l.text.values:
            t
            assert t != '' # ensure there are no missing names now
    
    # now test actual coregistration


    # viewer.show(block=True)




def test_coreg(make_napari_viewer, capsys):
    # test coregistration (at least run it)
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = CrunchyToolsWidget(viewer)
    viewer.window.add_dock_widget(w)

    from napari_hippo._basicTools import search
    from napari_hippo._coregTools import addCoreg, computeAffine, exportAffine, resample
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.png', stretch=True, stack=False) # this should load 2 more images

    viewer.layers.selection.clear()
    addCoreg()
    xy = np.random.rand(11,2) * 30
    for i, l in enumerate(viewer.layers):
        if '[kp]' in l.name:
            l.data = xy - i # add a shift of a few pixels

    r = computeAffine(base_image=viewer.layers['[slice] block1'])
    assert np.max( np.abs(r) ) < 1e6 # should all be close to 0

    # save
    pths = exportAffine()
    for p in pths:
        assert os.path.exists(p)

    # remove afffine layers
    todel = [l.name for l in viewer.layers if '[kp]' in l.name]
    for l in todel:
        del viewer.layers[l]

    # add them again (and check loading!)
    addCoreg()
    for l in viewer.layers:
        if '[kp]' in l.name:
            assert len(l.data) > 0

    # clean
    for p in pths:
        os.remove(p)

    # warp
    viewer.layers.selection.clear()
    resample()

    #assert (viewer.layers['[slice] block1 [warped]']._data_view.shape[0] == viewer.layers['[slice] block2 [warped]']._data_view.shape[0])
    #assert (viewer.layers['[slice] block1 [warped]']._data_view.shape[1] == viewer.layers['[slice] block2 [warped]']._data_view.shape[1])

    # viewer.show(block=True)


