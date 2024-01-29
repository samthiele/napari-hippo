import numpy as np
import os
from napari_hippo import napari_get_ENVI_reader, napari_get_specim_reader, h2n
import napari_hippo.testdata
from hylite import io

# tmp_path is a pytest fixture
def test_ENVI_reader(tmp_path, make_napari_viewer):
    """An example of how you might test your plugin."""

    # make a viewer
    viewer = make_napari_viewer()

    # load a HSI and .png test image
    for p in [napari_hippo.testdata.image, napari_hippo.testdata.block1]:
        original_data = io.load(p)
        original_data.decompress()
        if original_data.band_count() > 4:
            original_data_rgb = original_data.export_bands( original_data.header.get_list('default bands').astype(int) )
        else:
            original_data_rgb = original_data.export_bands((0,1,2))
        original_data = h2n(original_data.data)
        original_data_rgb = h2n(original_data_rgb.data)

        # get napari reader for the same file
        for original, force_rgb in zip([original_data, original_data_rgb], [False, True]):
            reader = napari_get_ENVI_reader(p)
            assert callable(reader)

            # make sure we're delivering the right format
            layer_data_list = reader(p, force_rgb=force_rgb )
            assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
            layer_data_tuple = layer_data_list[0]
            assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

            # make sure it's the same as it started
            #np.testing.assert_allclose(original, layer_data_tuple[0])
            #print(original.shape, layer_data_tuple[0].shape)
            assert original.shape == layer_data_tuple[0].shape

            # add to viewer
            viewer.add_image(layer_data_tuple[0], **layer_data_tuple[1])
    # viewer.show(block=True)

def test_specim_reader():
    # path = '/Volumes/Extreme SSD/Projects/VECTOR/ireland/HSI/TECK_RAW/20230330_Teck_1m/FENIX/SWIR_REFDATA_0_0m00_1m00_2023-03-30_12-38-40'
    path = '/Volumes/Extreme SSD/Projects/VECTOR/ireland/HSI/TECK_RAW/20230330_Teck_1m/RGB/RGB_TC_3660_008_139000140_394m20_402m00_2023-03-30_16-09-27'
    if os.path.exists(path):
        reader = napari_get_specim_reader(path)
        assert callable(reader)

        # test loading
        layer_data_list = reader( path )
        assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
        layer_data_tuple = layer_data_list[0]
        assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0

def test_get_reader_pass():
    reader = napari_get_ENVI_reader("fake.file")
    assert reader is None

    reader = napari_get_specim_reader("fake.file")
    assert reader is None


