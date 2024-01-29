from napari_hippo import write_single_image, write_multiple
import pathlib
import os
import numpy as np
def test_writer_functions(make_napari_viewer, capsys):

    # setup some environment
    viewer = make_napari_viewer()

    # add an image
    from napari_hippo._ioTools import search
    search(pathlib.Path(os.path.dirname(os.path.dirname(__file__))), 'testdata/*.hdr',
           rgb_only=False)  # this should load 1 test image

    # save individual image
    layer = viewer.layers['image']
    outpath = os.path.join( os.path.dirname(__file__), 'savetest.hdr')
    from napari_hippo._writer import write_single_image, write_multiple
    write_single_image( outpath, [(layer.data,
                        dict(name = layer.name, metadata = layer.metadata), 'image')] )
    assert os.path.exists(outpath)

    # load image again and check that it matches
    from napari_hippo import napari_get_ENVI_reader
    reader = napari_get_ENVI_reader(outpath)
    layer_data_list = reader(outpath)
    layer_data_tuple = layer_data_list[0]
    viewer.add_image(layer_data_tuple[0], **layer_data_tuple[1])

    # clean up now that we have reloaded the image
    os.remove(outpath)
    os.remove(os.path.splitext(outpath)[0] + ".dat")

    # check loaded image matches
    assert (np.array( layer.data.shape ) == np.array(layer_data_tuple[0].shape)).all()

    # viewer.show(block=True)

