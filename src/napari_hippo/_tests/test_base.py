"""
Test the basic / core functions of napari hippo, largely related to mapping data between napari (array format and metadata dictionary) and
hylite data classes (to facilitate quick / easy / clean hyperspectral processing).
"""

# load custom pytest fixtures
from .fixtures import *
import numpy as np
import hylite


def test_HippoData(imageMode, capsys):
    viewer, layer = imageMode() # get viewer and added image layer from fixture
    
    # construct a basic HippoData class
    from napari_hippo import HippoData
    data = HippoData( layer, test_value = 7)

    # check value is both a class attribute and in the layer metadata
    assert data.test_value == 7
    assert layer.metadata['test_value'] == 7

    # mutate value and check that it is reflected in the metadata
    data.test_value = 42
    assert layer.metadata['test_value'] == 42
    
    # check that type field is set
    assert data.type == 'HippoData'
    assert layer.metadata['type'] == 'HippoData'
    assert data.layer.name == '[HippoData] image'

    # get a HSICube object
    from napari_hippo import HSICube, getLayer
    cube = HSICube( layer )
    assert cube.type == 'HSICube'
    assert cube.layer.name == '[HSICube] image'

    image = cube.toHyImage() # get image and check it matches
    assert image is not None
    assert np.max((image.data.shape)) == np.max( layer.data.shape )
    assert np.min((image.data.shape)) == np.min( layer.data.shape )
    assert np.nanmean( image.data ) == np.nanmean( layer.data )
    assert isinstance( getLayer(cube.layer.name, viewer=viewer), HSICube) # check dtype

    image.data = image.data / 2 # modify image and check it matches
    cube.fromHyImage(image)
    assert np.max((image.data.shape)) == np.max( layer.data.shape )
    assert np.min((image.data.shape)) == np.min( layer.data.shape )
    assert np.nanmean( image.data ) == np.nanmean( layer.data )
    assert np.max(image.get_wavelengths()) > 1000. # check we have some meaningful wavelengths
    assert cube.ndim() == 3

    # check that the shape attributes are correct
    def checkDims( hippoData, hyImage, bands):
        assert hippoData.xdim == hyImage.xdim()
        assert hippoData.cols == hyImage.xdim()
        assert hippoData.ydim == hyImage.ydim()
        assert hippoData.rows == hyImage.ydim()
        assert hippoData.bands == bands
    checkDims( cube, image, image.band_count() )

    # while we are here, also check that the napari_hippo.getMode() is correct.
    from napari_hippo import getMode
    assert getMode(viewer) == 'Image'
    
    # check that the factory function works
    HSICube.construct( image, name='test' )
    assert '[HSICube] test' in viewer.layers

    # add basic image layers
    from napari_hippo import RGB, RGBA, BW
    obj = RGB.construct( image.export_bands(hylite.SWIR), name='swir')
    assert '[RGB] swir' in viewer.layers
    assert 'wavelength' in obj.layer.metadata
    assert isinstance( getLayer(obj.layer, viewer=viewer), RGB) # check dtype
    checkDims( obj, image, bands=3 ) # check that the shape attributes are correct

    obj = RGBA.construct( image.export_bands((2200., 2250., 2300., 2350.)), name='rgba')
    assert '[RGBA] rgba' in viewer.layers
    assert 'wavelength' in obj.layer.metadata
    assert isinstance( getLayer(obj.layer, viewer=viewer), RGBA) # check dtype
    checkDims( obj, image, bands=4 ) # check that the shape attributes are correct

    obj = BW.construct( image.export_bands((2200.)), name='bw')
    assert '[BW] bw' in viewer.layers
    assert 'wavelength' in obj.layer.metadata
    assert isinstance( getLayer(obj.layer, viewer=viewer), BW) # check dtype
    checkDims( obj, image, bands=1 ) # check that the shape attributes are correct
    
    # test mask object (in Image mode)
    from napari_hippo import Mask
    mask = Mask.construct( cube.layer, viewer=viewer)
    assert mask.ndim() == 2

    # mask some pixels
    mask.layer.data[2:-2, :] = 1
    mask.layer.data[:, 2:-2] = 1
    mask.layer.data[10:12, :] = 0
    mask.layer.refresh()
    mask.apply()
    assert np.isnan(cube.layer.data).any() # check some pixels were removed!

    # lastly check saving works
    mask.save()
    assert os.path.exists( mask.path )
    del viewer.layers[ viewer.layers.index(mask.layer) ]
    mask2 = Mask.construct( cube.layer, viewer=viewer)
    assert (mask2.layer.data == 1).any() # make sure something loaded!

    # cleanup
    os.remove( os.path.splitext(mask.path)[0] + '.dat')
    os.remove( os.path.splitext(mask.path)[0] + '.hdr')
    #viewer.show(block=True)
    
def test_Stack(stackMode, capsys):
    viewer, layer = stackMode() # get viewer and added image layer from fixture

    # construct a basic HippoData class
    from napari_hippo import Stack
    data = Stack( layer, test_value = 7)
    # check value is both a class attribute and in the layer metadata
    assert data.test_value == 7
    assert layer.metadata['test_value'] == 7

    # mutate value and check that it is reflected in the metadata
    data.test_value = 42
    assert layer.metadata['test_value'] == 42

    # check path is a list
    assert isinstance(data.path, list)
    assert len(data.path) > 1

    # check dimensions
    for i,l in enumerate(layer.data):
        assert data.stack_rows[i] == l.compute().shape[0]
        assert data.stack_ydim[i] == l.compute().shape[0]
        assert data.stack_cols[i] == l.compute().shape[1]
        assert data.stack_xdim[i] == l.compute().shape[1]
        assert data.stack_bands[i] == l.compute().shape[2]
        assert data.stack_bands[i] == l.compute().shape[2]
    assert data.rows == np.max(data.stack_rows)
    assert data.cols == np.max(data.stack_cols)
    assert data.bands == np.max(data.stack_bands)

    # check dimensions are actually different for different images in Stack!
    assert np.max( np.abs( np.diff( data.stack_xdim ) ) ) > 0
    assert np.max( np.abs( np.diff( data.stack_ydim ) ) ) > 0

    # while we are here, also check that the napari_hippo.getMode() is correct.
    from napari_hippo import getMode
    assert getMode(viewer) == 'Batch'

    # test toImage from Stack
    images = data.toHyImage()
    assert len(images) == 2
    assert images[0].xdim() != images[1].xdim()

    # test mask object (in Batch mode)
    from napari_hippo import Mask
    mask = Mask.construct( data.layer, viewer=viewer)
    assert mask.ndim() == 4

    # mask some pixels
    mask.layer.data[:, 2:-2, :] = 1
    mask.layer.data[:, :, 2:-2] = 1
    mask.layer.refresh()
    
    # apply mask
    mask.apply()
    assert np.isnan(data.layer.data).any() # check some pixels were removed!

    # check saving and loading works
    mask.save()
    del viewer.layers[ viewer.layers.index(mask.layer) ]
    mask2 = Mask.construct( data.layer, viewer=viewer)
    assert (mask2.layer.data == 1).any() # make sure something loaded!
    for p in mask2.path: # check files exist and cleanup
        assert os.path.exists( p )
        os.remove( os.path.splitext(p)[0] + '.dat')
        os.remove( os.path.splitext(p)[0] + '.hdr')

    #viewer.show(block=True)

def test_ROI(imageMode, stackMode, capsys):

    for M in [stackMode, imageMode]:
        viewer, layer = M()

        from napari_hippo import ROI
        for mode in ['keypoints', 'polygons']:
            roi = ROI.construct(layer, mode=mode, viewer=viewer )

            # edit layer and check it updates
            verts, text = roi.toList(world=layer)
            if mode == 'keypoints':
                verts += [(0,0,0),(0,0,10.),(0,10,10),(0,10.,0)]
                text += ['NewPoint' for p in range(4)]
            else:
                if 'batch' in roi.layer.name: # batch mode; add new 3D polygon
                    verts.append(np.array([(0,0,0),(0,0,10),(0,10,10),(0,10,0)],dtype=float))
                else: # image mode; add new 2D polygon
                    verts.append(np.array([(0,0),(0,10),(10,10),(10,0)], dtype=float))
                text.append("NewROI")
            roi.fromList( verts, text, world=layer )

            # check transpose works in fromList 
            verts1, text1 = roi.toList(world=False, transpose=False)
            verts2, text2 = roi.toList(world=False, transpose=True)
            assert not (np.array(verts1[-1]) == np.array(verts2[-1])).all()
            assert (np.array(text1) == np.array(text2)).all()

            # check various permutations of toList and fromList
            refdata = roi.layer.data.copy()
            for w in [False, True, layer]:
                for t in [False, True]:
                    verts, text = roi.toList(world=w, transpose=t)
                    assert len(verts)>0
                    assert len(verts) == len(text)
                    roi.fromList(verts, text, world=w, transpose=t)
                    data2 = roi.layer.data
                    for i,p in enumerate(refdata):
                        assert (np.array(p == data2[i])).all() # check nothing has changed 

        # check correct layers have been constructed and are not empty
        assert ('[KP] image' in viewer.layers) or ('[KP] batch' in viewer.layers)
        assert ('[ROI] image' in viewer.layers) or ('[ROI] batch' in viewer.layers)
        if '[KP] image' in viewer.layers:
            assert len( viewer.layers['[KP] image'].data) > 0
        if '[KP] batch' in viewer.layers:
            assert len( viewer.layers['[KP] batch'].data) > 0
        if '[ROI] image' in viewer.layers:
            assert len( viewer.layers['[ROI] image'].data) > 0
        if '[ROI] batch' in viewer.layers:
            assert len( viewer.layers['[ROI] batch'].data) > 0

        #viewer.show(block=True)
        #break

        # cleanup
        viewer.close()
        del layer
        del viewer


