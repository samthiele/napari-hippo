"""
This file implements readers for loading and saving hyperspectral data.
"""
import os
import numpy as np
from hylite import io
import glob
import napari
from napari_hippo import h2n, HSICube, getMode, View
from hylite.io import loadHeader, matchHeader, loadSubset
import hylite

def napari_get_hylite_reader(path):
    """
    Load an ENVI file.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, str):
        # wrap in list
        path = [path]

    # if we know we cannot read the file, we immediately return None.
    for p in path:
        if p.endswith(".hdr") or p.endswith(".dat") or p.endswith(".ply"):
            return read_hylite
        elif p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".bmp"):
            return read_RGB
    print("??")
    # otherwise we cannot read this
    return None

def napari_get_specim_reader(path):
    """
    Read a raw specim image directory and correct it using hylite. Only compatable with FX, Fenix and OWL series
    cameras. Also note that hylite must first be setup with the relevant calibration data!

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    if isinstance(path, str):
        # wrap in list
        path = [path]

    # if we know we cannot read the file, we immediately return None.
    for p in path:
        if not os.path.isdir(p): # only accept directories
            return None

        # check for metadata files
        meta = glob.glob(os.path.join(p, '**/manifest.xml'), recursive=True)
        if len(meta) == 0:
            return None

    # otherwise we return the *function* that can read ``path``.
    return read_specim


def _getkwags(image, name, path=None, dtype=None):
    # define kwargs for the viewer.add_image method
    add_kwargs = dict(name=name, 
                        metadata=dict())

    # add all image header info to metadata
    for k,v in image.header.items():
        add_kwargs['metadata'][k.replace(' ', '_')] =  v
    if dtype is not None:
        add_kwargs['metadata']['type'] = dtype
    if path is not None:
        add_kwargs['metadata']['path'] = path
    return add_kwargs

def read_specim( path, return_data=False ):
    """
    Read and preprocess a raw specim image. If return_image is True
    then this returns a HyImage result rather than an arguments tuple as expected
    by napari.
    """
    if getMode(napari.current_viewer()) == 'Batch':
        napari.utils.notifications.show_warning("Cannot load more images when in batch mode. Please delete existing stack before opening a new file.")
        return []
    
    if isinstance(path, str):
        path = [path]

    out =  []
    for p in path:
        for manifest in glob.glob(os.path.join(p, '**/manifest.xml'), recursive=True):
            root = os.path.dirname(manifest)
            print(os.path.join(root, 'metadata/*.xml'))
            meta = glob.glob(os.path.join(root, 'metadata/*.xml'))
            if len(meta) > 0:
                image = None
                sensor = ''
                with open(meta[0], 'r') as f:
                    for l in f.readlines():
                        if 'sensor type' in l.lower():
                            if 'fenix1k' in l.lower():
                                sensor = 'fenix1k'
                            if 'fenix' in l.lower():
                                sensor = 'fenix'
                            elif 'fx50' in l.lower():
                                sensor = 'fx50'
                            elif 'lwir' in l.lower():
                                sensor = 'lwir'
                            elif 'rgb' in l.lower():
                                sensor = 'rgb'
                            break
                if 'fenix' in sensor: # Fenix
                    from hylite.reference.spectra import R90 as ref
                    from hylite.sensors import Fenix
                    if '1k' in sensor:
                        image = Fenix.correct_folder( root, calib=ref,
                                                        flip=False, # no lense flip for 1k
                                                        shift=False,
                                                        verbose=True)
                    else:
                        image = Fenix.correct_folder( root, calib=ref,
                                                            flip=True,
                                                            shift=False,
                                                            verbose=True)
                elif 'fx50' in sensor: # FX50
                    from hylite.sensors import FX50
                    image = FX50.correct_folder(root, bpr=True, flip=True,
                                                      verbose=True)
                elif 'lwir' in sensor: # OWL
                    from hylite.sensors import OWL
                    image = OWL.correct_folder(root, bpr=True, flip=True,
                                                verbose=True)
                elif 'rgb' in sensor: # RGB
                    fpath = [i for i in glob.glob( root + '/capture/*.hdr') if 'DARKREF' not in i and 'WHITEREF' not in i]
                    image =  io.load(fpath[0])
                    image.data = np.rot90(image.data, k=3)
                    image.data = image.data.astype(np.float32) / 255.

                if image is not None:
                    # store add_image kwargs
                    if return_data:
                        out.append(image)
                    else:
                        out.append( HSICube.construct( image, os.path.basename(p), args_only=True, path=p ) )
    return out

def read_RGB( path, force_rgb=True, return_data=False ):
    """
    Open one or more normal RGB images in .png, .jpg or .bmp format.

    Args:
        path: Path to the image to load
        force_rgb: True if the HSI should be flattened to a false-colour RGB image. This
                   argument is meaningless in this context (and ignored), but included for compatability.
        return_image: If True, a list of HyImage instances is returned instead of the expected napari kwargs.

    Returns: A tuple containing (image data, napari kwargs, image name ).
    """
    if getMode(napari.current_viewer()) == 'Batch':
        napari.utils.notifications.show_warning("Cannot load more images when in batch mode. Please delete existing stack before opening a new file.")
        return []

    # wrap in list
    if isinstance(path, str):
        path = [path]

    out = []
    for p in path:

        # parse layer name
        name = os.path.splitext(os.path.basename(p))[0]

        # load image
        image = io.load(p)
        image.decompress()

        # store add_image kwargs
        if return_data:
            out.append(image)
        else:
            out.append( HSICube.construct( image, name, args_only=True, path=p ) )
    return out

def read_hylite( path, force_rgb=False, return_data=False ):
    """
    Open one or more hylite-compatible files (e.g., ENVI images, PLY point clouds, etc).

    Args:
        path: Path to the image to load
        force_rgb: True if the HSI should be flattened to a false-colour RGB image.
        return_image: If True, a list of HyImage instances is returned instead of the expected napari kwargs.

    Returns: A tuple containing (image data, napari kwargs, image name ).

    """
    if getMode(napari.current_viewer()) == 'Batch':
        napari.utils.notifications.show_warning("Cannot load more images when in batch mode. Please delete existing stack before opening a new file.")
        return []

    # wrap in list
    if isinstance(path, str):
        path = [path]

    out = []
    for p in path:
        # parse layer name
        name = os.path.splitext(os.path.basename(p))[0]

        # load header
        p, d = matchHeader(p) # make sure we have the path to the header file
        h = io.loadHeader(p)

        # load point cloud
        if 'ply' in os.path.splitext(d)[1]: # load a point cloud
            cloud = io.load( d )
            cameras = set([int(k.split(' ')[1]) for k in cloud.header if 'camera' in k])
            if len(cameras) == 0:
                cameras = [0] # this will return None later, which will in turn end up as a nadir view
            if return_data:
                out.append(cloud)
            else:
                for i in cameras:
                    # store add_image kwargs
                    out.append( View.construct( cloud, name, cloud.header.get_camera(i), args_only=True, path=p))
        elif 'hyc' in os.path.splitext(d)[1]: # load a scene
            assert False # not implemented
        else: # assume this is an image
            if force_rgb: # load image as RGB preview only
                if 'default bands' in h:
                    bands = h.get_list('default bands').astype(int)
                else:
                    # defaults
                    n = h.band_count()
                    bands = np.array([0.25 * n, 0.5 * n, 0.75 * n]).astype(int)

                    # try better combo
                    for bands in [hylite.RGB, hylite.VNIR, hylite.SWIR, hylite.MWIR, hylite.LWIR]:
                        if (np.min(bands) > np.min(image.get_wavelengths())):
                            if (np.max(bands) < np.max(image.get_wavelengths())):
                                try:
                                    bands = np.array([image.get_band_index(b) for b in bands])
                                except:
                                    pass # don't worry if there isn't the target band
                image = loadSubset( p, bands=bands )
            else: # load full dataset
                image = io.load( p )
            image.decompress()
        
            # store add_image kwargs
            if return_data:
                out.append(image)
            else:
                out.append( HSICube.construct( image, name, args_only=True, path=p ) )
    return out

