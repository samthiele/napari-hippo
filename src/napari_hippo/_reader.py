"""
This file implements readers for loading and saving hyperspectral data.
"""
import os
import numpy as np
from hylite import io
import glob
from napari_hippo import h2n
from hylite.io import loadHeader, matchHeader, loadSubset

def napari_get_ENVI_reader(path):
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
        if p.endswith(".hdr") or p.endswith(".dat"):
            return read_ENVI
        elif p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".bmp"):
            return read_RGB

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

def read_specim( path ):
    """
    Read and preprocess a raw specim image.
    """
    if isinstance(path, str):
        path = [path]

    out =  []
    for p in path:
        for manifest in glob.glob(os.path.join(p, '**/manifest.xml'), recursive=True):
            root = os.path.dirname(manifest)
            print(os.path.join(root, 'metadata/*.xml'))
            meta = glob.glob(os.path.join(root, 'metadata/*.xml'))
            print(meta)
            if len(meta) > 0:
                image = None
                sensor = ''
                with open(meta[0], 'r') as f:
                    for l in f.readlines():
                        if 'sensor type' in l.lower():
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
                    data = image.data.T
                    name = '[cube]'+os.path.splitext(os.path.basename( root ))[0]
                    add_kwargs = dict(name=name, metadata=dict(
                        type='HSIf',
                        path=p,
                        wav=image.get_wavelengths()))
                    out.append((data, add_kwargs, "image"))
    return out

def read_RGB( path, force_rgb=True ):
    """
    Open one or more normal RGB images in .png, .jpg or .bmp format.

    Args:
        path: Path to the image to load
        force_rgb: True if the HSI should be flattened to a false-colour RGB image. This
                   argument is meaningless in this context (and ignored), but included for compatability.

    Returns: A tuple containing (image data, napari kwargs, image name ).
    """
    # wrap in list
    if isinstance(path, str):
        path = [path]

    out = []
    for p in path:
        #if p.endswith(".hdr") or p.endswith(".dat"):
        #    out += read_ENVI( p, force_rgb ) # use different reader here
        #    continue
        #if not p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".bmp"):
        #    continue # pass silently

        # parse layer name
        name = '[slice] ' + os.path.splitext(os.path.basename(p))[0]

        # load image
        image = io.load(p)
        image.decompress()

        # define image metadata
        add_kwargs = dict( name=name, metadata=dict(type='RGB', path=p ) )
        if (image.band_count() == 4):
            add_kwargs['metadata']['type'] = add_kwargs['metadata'].get('type', 'RGBA')  # this is a RGBA image

        out.append((h2n(image.data), add_kwargs, 'image'))
    return out

def read_ENVI( path, force_rgb=False ):
    """
    Open one or more ENVI images.

    Args:
        path: Path to the image to load
        force_rgb: True if the HSI should be flattened to a false-colour RGB image.

    Returns: A tuple containing (image data, napari kwargs, image name ).

    """

    # wrap in list
    if isinstance(path, str):
        path = [path]

    out = []
    for p in path:

        #if p.endswith(".png") or p.endswith(".jpg") or p.endswith(".jpeg") or p.endswith(".bmp"):
        #    out += read_RGB(p, force_rgb)  # use different reader here
        #    continue
        #if not (p.endswith(".hdr") or p.endswith(".dat")):
        #    continue # pass silently

        # parse layer name
        if force_rgb:
            name = '[slice] '+os.path.splitext(os.path.basename(p))[0]
        else:
            name = '[cube] '+os.path.splitext(os.path.basename(p))[0]

        # load as HyImage
        p, _ = matchHeader(p) # make sure we have the path to the header file
        h = io.loadHeader(p)
        if force_rgb: # load image as RGB preview only
            if 'default bands' in h:
                bands = h.get_list('default bands').astype(int)
            else:
                n = h.band_count()
                bands = np.array([0.25 * n, 0.5 * n, 0.75 * n]).astype(int)
            image = loadSubset( p, bands=bands )
        else: # load full image
            image = io.load( p )
        image.decompress()

        # kwargs for the viewer.add_image method
        add_kwargs = dict(name=name, metadata=dict(type='HSIf', path=p, wav=image.get_wavelengths()))
        if h.band_count() > image.band_count(): # if we opened just a preview image
            add_kwargs['metadata']['type'] = 'HSIp' # flag that this is preview slice

        # store
        out.append( ( h2n(image.data), add_kwargs, 'image') )
    return out
