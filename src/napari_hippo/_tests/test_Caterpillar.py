import os
import pathlib
import numpy as np
import napari_hippo
from napari_hippo import CaterpillarWidget
from .fixtures import *
from pathlib import Path

def test_CaterpillarWidget(imageMode, capsys):
    # make viewer and add an image layer using our fixture
    viewer, layer = imageMode()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = CaterpillarWidget(viewer)
    viewer.window.add_dock_widget(w)

    # add points
    pxy = np.random.rand(3, 2) * 15
    points = viewer.add_points(pxy, name="Points")

    # plot caterpillar
    from napari_hippo._caterpillarWidget import addCaterpillar
    addCaterpillar(layer, points) # plot with fully loaded HSI
    
    # check export
    from napari_hippo._caterpillarWidget import export
    pth = os.path.join(os.path.dirname( napari_hippo.testdata.image ), 'testlib.csv' )
    export(base_image = layer, 
            query_points = viewer.layers['Points'], filename = Path(pth) )
    assert os.path.exists(pth)
    os.remove(pth) # cleanup
    #viewer.show(block=True)