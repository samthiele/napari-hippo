import numpy as np
from napari_hippo import CrunchyToolsWidget, IOWidget
import pathlib
import os

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_CrunchyToolsWidget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w1 = IOWidget(viewer)
    w2 = CrunchyToolsWidget(viewer)
    viewer.window.add_dock_widget(w2)

    # call our widget methods
    from napari_hippo._ioTools import search
    search() # this should run but not load anything
    assert len(viewer.layers) == 0
    search(pathlib.Path(os.path.dirname( os.path.dirname(__file__) )),'testdata/image.hdr', rgb_only = True) # this should load 1 test image
    assert len(viewer.layers) == 1
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.png', stretch=True) # this should load 2 more images
    assert len(viewer.layers) == 3

    # this should load 1 image stack
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.png', stack=True)
    assert len(viewer.layers) == 4

    # test getAsHyImage functions
    for l in ['image','block1','block2']:
        pass

    # viewer.show(block=True)

def test_coreg(make_napari_viewer, capsys):
    # test coregistration (at least run it)
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = CrunchyToolsWidget(viewer)
    viewer.window.add_dock_widget(w)

    from napari_hippo._ioTools import search
    from napari_hippo._crunchyTools import addCoreg, computeAffine, exportAffine, resample
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.png', stretch=True) # this should load 2 more images

    viewer.layers.selection.clear()
    addCoreg()
    xy = np.random.rand(11,2) * 30
    for i, l in enumerate(viewer.layers):
        if '[kp]' in l.name:
            l.data = xy - i # add a shift of a few pixels

    r = computeAffine(base_image=viewer.layers['block1'])
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
    assert (viewer.layers['block1 [warped]']._data_view.shape[0] == viewer.layers['block2 [warped]']._data_view.shape[0])
    assert (viewer.layers['block1 [warped]']._data_view.shape[1] == viewer.layers['block2 [warped]']._data_view.shape[1])

    # viewer.show(block=True)


