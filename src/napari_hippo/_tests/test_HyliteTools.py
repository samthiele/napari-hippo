import os
import pathlib

from napari_hippo import HyliteToolsWidget


def test_HyliteToolsWidget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer, and add it to napari as a dock
    w = HyliteToolsWidget(viewer)
    viewer.window.add_dock_widget(w)

    from napari_hippo._hyliteTools import falseColor, hullCorrect, dimensionReduction, calculator

    def tests():
        # tests to run on loaded image
        # TODO - do better tests? This simply runs the functions...
        for f in [falseColor, hullCorrect, dimensionReduction, calculator ]:
            viewer.layers.selection.clear()
            viewer.layers.selection.add(viewer.layers['image_full'])
            f()

    # load full HSI image
    from napari_hippo import make_sample_data
    data, kwds, _ = make_sample_data()[0]
    kwds['name'] = 'image_full'
    viewer.add_image(data, **kwds)

    tests() # run test suite on this (in core) image

    # load RGB only preview
    from napari_hippo._ioTools import search
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.hdr', rgb_only=True)

    # run test suite on this (out of core) image
    tests()

    # viewer.show(block=True)