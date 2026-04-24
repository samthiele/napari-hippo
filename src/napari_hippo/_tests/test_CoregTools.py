import os
import pathlib
import hylite
from .fixtures import *
import time
import numpy as np
import napari_hippo
from pathlib import Path
from napari_hippo import getLayer, HSICube
from hylite import io

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

def test_fitExtent(coregMode):
    # make viewer and add an image layer using our fixture
    viewer, layers = coregMode()

    from napari_hippo._coregTools import  fitExtent   # match extent
    viewer.layers.selection.clear()
    fitExtent( layers[0] )

    assert np.array(layers[1].affine)[0,0] < 1 # affine should be smaller than 1

    #viewer.show(block=True)

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
    from napari_hippo._coregTools import  fitAffine  
    resid = fitAffine( base = layers[0] )
    assert resid < 1e-6 # should be very very small!

    # save affine
    from napari_hippo._coregTools import  save
    headers = save()

    # cleanup!
    for p,h in headers.items():
        assert 'affine' in h
        todel = []
        for k,v in h.items():
            if ('affine' in k) or ('point kp' in k):
                todel.append(k)
        for k in todel:
            del h[k]
        io.saveHeader( p, h )

    viewer.show(block=True)
