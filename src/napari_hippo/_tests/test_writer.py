from napari_hippo import write_single_image, write_multiple
import pathlib
import os
import numpy as np

from .fixtures import *

def test_writer_functions(imageMode, capsys):

    # setup some environment
    viewer, layer = imageMode()

    # save image
    outpath = os.path.join( os.path.dirname(__file__), 'savetest.hdr')
    from napari_hippo._writer import write_single_image, write_multiple
    write_single_image( outpath, [(layer.data,
                        dict(name = layer.name, metadata = layer.metadata), '[cube] image')] )
    assert os.path.exists(outpath)

    # load image again and check that it matches
    from napari_hippo import napari_get_hylite_reader
    reader = napari_get_hylite_reader(outpath)
    layer_data_list = reader(outpath)
    layer_data_tuple = layer_data_list[0]
    viewer.add_image(layer_data_tuple[0], **layer_data_tuple[1])

    # clean up now that we have reloaded the image
    os.remove(outpath)
    os.remove(os.path.splitext(outpath)[0] + ".dat")

    # check loaded image matches
    assert (np.array( layer.data.shape ) == np.array(layer_data_tuple[0].shape)).all()

    # viewer.show(block=True)

