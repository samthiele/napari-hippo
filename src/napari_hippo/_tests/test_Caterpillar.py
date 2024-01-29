import os
import pathlib
import numpy as np
from napari_hippo import CaterpillarWidget

def test_CaterpillarWidget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = CaterpillarWidget(viewer)
    viewer.window.add_dock_widget(w)

    # load full HSI image
    from napari_hippo import make_sample_data
    data, kwds, _ = make_sample_data()[0]
    kwds['name'] = 'image_full'
    viewer.add_image(data, **kwds)

    # load RGB only preview
    from napari_hippo._ioTools import search
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.hdr', rgb_only=True)

    # add points
    pxy = np.random.rand(3, 2) * 15
    points = viewer.add_points(pxy, name="Points")

    # plot caterpillar
    from napari_hippo._caterpillarWidget import addCaterpillar
    image1 = viewer.layers['image_full']
    image2 = viewer.layers['image']

    addCaterpillar(image1, points) # plot with fully loaded HSI
    addCaterpillar(image2, points) # plot with out-of-core HSI

    # viewer.show(block=True)