"""
Define some common functions that are used more or less everywhere :-)
"""
import hylite
import napari
import numpy as np

def n2h( a ):
    """
    Convert a HSI image array in napari (b,y,x) or (y,x,3) or (y,x,4) format to hylite (x,y,b) format.
    """
    if (a.shape[-1] == 3) or (a.shape[-1] == 4):
        return np.transpose(a, (1,0,2) ) # bands stay in the last axis (RGB or RGBA)
    else:
        return np.transpose(a, (2,1,0) ) # bands become the first axis (Multiband)

def h2n( a ):
    """
    Convert an array in hylite format (x,y,b) to napari format (b,x,y).
    """
    if (a.shape[-1] == 3) or (a.shape[-1] == 4):
        return np.transpose(a, (1,0,2)) # bands stay in the last axis (RGB or RGBA)
    else:
        return np.transpose(a, (2, 1, 0) ) # bands become the first axis (Multiband)

def getHyImage(viewer, layer=None, bands=None, pixels=None):
    """
    Get a HyImage instance from a hyperspectral napari layer. Loads the requested data if needed.

    Args:
        viewer: The napari viewer to get data from.
        layer: The napari layer to convert. If None, the first valid selected layer will be used.
        bands: Only return these specific bands. Faster if the hyperspectral image is stored out of core.
        pixels: A list of [(x,y), ... ] pixel coordinates. If not None passed a HyData instance containing only these
                pixels will be returned. Cannot be used in conjunction with bands.
    """
    if layer is None:
        layers = viewer.layers.selection
    else:
        layers = [layer]

    if len(layers) > 0:
        for l in layers:
            if isinstance(l, napari.layers.Image):
                if ('wav' in l.metadata) and ('path' in l.metadata) and ('type' in l.metadata):
                    if 'HSI' in l.metadata['type']:
                        if l.rgb == True:
                            pth = l.metadata['path']
                            if bands is not None:  # load only some bands
                                from hylite.io import loadSubset
                                return loadSubset(pth, bands=bands), l
                            elif pixels is not None:
                                from hylite.io import loadSubset
                                return loadSubset(pth, pixels=pixels), l
                            else:  # load full image
                                image = hylite.io.load(pth)
                                image.decompress()
                                return image, l
                        else:
                            w = l.metadata['wav']
                            if bands is None:
                                return hylite.HyImage(n2h(l.data), wav=w), l
                            elif pixels is not None:
                                d = hylite.HyData( np.array( [l.data[x,y,:] for x,y in pixels] ) )
                                d.set_wavelengths( w )
                                return d
                            else:
                                return hylite.HyImage( n2h( l.data ), wav=w).export_bands(bands), l

    # if we got here, there are not appropriate images in the selection
    napari.utils.notifications.show_warning("Please select an HSI image.")
    return None, None