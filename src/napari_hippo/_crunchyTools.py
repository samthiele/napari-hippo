"""

Some crunchy tools for data munging.

"""

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import napari
import os
import numpy as np

from magicgui import magicgui
import napari
from ._guiBase import GUIBase

from skimage.transform import AffineTransform, warp
from skimage.measure import ransac

class CrunchyToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.scale_widget = magicgui(scale, call_button='Scale',
                                     x_scale = {"min" : -np.inf, "max" : np.inf },
                                     y_scale={"min": -np.inf, "max": np.inf} )
        self.translate_widget = magicgui(translate, call_button='Translate',
                                         x = {"min" : -np.inf, "max" : np.inf },
                                         y={"min": -np.inf, "max": np.inf} )
        self.rot_widget = magicgui(rot, call_button='Rotate', angle = {"min" : -np.inf, "max" : np.inf } )
        self._add([self.scale_widget, self.translate_widget,
                   self.rot_widget], 'Simple Transforms')

        self.coreg_widget  = magicgui(addCoreg, call_button='Add Target(s)')
        self.affine_widget = magicgui(computeAffine, call_button='Align images')
        self.export_affine_widget = magicgui(exportAffine, call_button='Save affine')
        self._add([ self.coreg_widget,
                    self.affine_widget,
                    self.export_affine_widget], 'Coregister')

        self.resample_widget = magicgui(resample, call_button='Apply Affine')
        self._add([ self.resample_widget], "Resample" )

        # add spacer at the bottom of panel
        self.qvl.addStretch()

def scale( x_scale : float = -1, y_scale : float = -1 ):
    """
    Downsample the selected image
    Args:
        scale_factor: What factor to downsample the image by (spatial binning)

    Returns: The downsampled image layer.

    """
    # get selection
    viewer = napari.current_viewer()  # get viewer
    layers = viewer.layers.selection
    if len(layers) == 0:
        napari.utils.notifications.show_warning('Please select a layer to transform')
        return

    for l in layers:
        A = np.array(l.affine)
        affine = np.eye(A.shape[0])

        # compute affine for various data types
        if len(l.data.shape) == 4: # this is a stack of RGB
            affine[1, 1] = y_scale
            affine[2, 2] = x_scale
        elif (len(l.data.shape) == 3) and (l.data.shape[-1] > 4): # data cube
            affine[1, 1] = y_scale
            affine[2, 2] = x_scale
        else: # conventional RGB or greyscale
            affine[0, 0] = y_scale
            affine[1, 1] = x_scale

        l.affine = np.dot( affine, A )

def translate( x : float = -0.5, y : float = -0.5, relative=True ):
    """
    Downsample the selected image
    Args:
        x: x offset
        y: y offset
        relative: True if offsets are fractions of image dimensions

    Returns: The downsampled image layer.

    """
    # get selection
    viewer = napari.current_viewer()  # get viewer
    layers = viewer.layers.selection
    if len(layers) == 0:
        napari.utils.notifications.show_warning('Please select a layer to transform')
        return

    for l in layers:
        A = np.array(l.affine)
        affine = np.eye(A.shape[0])

        # compute affine for various data types
        if len(l.data.shape) == 4: # this is a stack of RGB
            affine[1, -1] = y
            affine[2, -1] = x
            if relative:
                affine[1, -1] *= l.data.shape[1]
                affine[2, -1] *= l.data.shape[2]
        elif (len(l.data.shape) == 3) and (l.data.shape[-1] > 4): # data cube
            affine[1, -1] = y
            affine[2, -1] = x
            if relative:
                affine[1, -1] *= l.data.shape[1]
                affine[2, -1] *= l.data.shape[2]
        else: # conventional RGB or greyscale
            affine[0, -1] = y
            affine[1, -1] = x
            if relative:
                affine[0, -1] *= l.data.shape[0]
                affine[1, -1] *= l.data.shape[1]
        l.affine = np.dot( affine, A )

def rot( angle : float = 90 ):
    """
    Rotate a HSI image by 90 degrees clockwise.
    Returns: The rotated image layer
    """
    # get selection
    viewer = napari.current_viewer()  # get viewer
    layers = viewer.layers.selection
    if len(layers) == 0:
        napari.utils.notifications.show_warning('Please select a layer to transform')

    for l in layers:
        A = np.array(l.affine)
        affine = np.eye(A.shape[0])

        # compute affine for various data types
        if len(l.data.shape) == 4: # this is a stack of RGB
            affine[0 + 1, 0 + 1] = np.cos(np.deg2rad(angle))
            affine[0 + 1, 1 + 1] = -np.sin(np.deg2rad(angle))
            affine[1 + 1, 0 + 1] = np.sin(np.deg2rad(angle))
            affine[1 + 1, 1 + 1] = np.cos(np.deg2rad(angle))
        elif (len(l.data.shape) == 3) and (l.data.shape[-1] > 4): # data cube
            affine[0 + 1, 0 + 1] = np.cos(np.deg2rad(angle))
            affine[0 + 1, 1 + 1] = -np.sin(np.deg2rad(angle))
            affine[1 + 1, 0 + 1] = np.sin(np.deg2rad(angle))
            affine[1 + 1, 1 + 1] = np.cos(np.deg2rad(angle))

        else: # conventional RGB or greyscale
            affine[0, 0] = np.cos(np.deg2rad(angle))
            affine[0, 1] = -np.sin(np.deg2rad(angle))
            affine[1, 0] = np.sin(np.deg2rad(angle))
            affine[1, 1] = np.cos(np.deg2rad(angle))
        l.affine = np.dot( affine, A )

def addCoreg():
    """
    Create coregistration layers who's vertices can be adjusted to manually-identified keypoints.
    """
    # get viewer
    viewer = napari.current_viewer()

    layers = list(viewer.layers.selection)
    if len(layers) == 0:
        layers = list(viewer.layers)

    points = []
    for l in layers:
        if isinstance(l, napari.layers.Image ):
            if (l.name + " [kp]") not in layers:
                points.append(viewer.add_points(name=l.name + " [kp]",
                                                metadata={'image': l.name,
                                                          'affine' : l.affine},
                                                size=6))
                points[-1].mode = 'ADD'

                # load data?
                if 'path' in l.metadata:
                    pth = os.path.join( os.path.dirname( l.metadata['path'] ), 'crunchycoreg.npz' )
                    if os.path.exists(pth):
                        f = np.load(pth)
                        print("Loading coreg points from %s" % pth)
                        if 'keypoints' in f.keys():
                            if ('%s_affine' % l.name) in f:
                                kp = f['keypoints'].reshape((-1,2))[:,[1,0]]
                                aff = np.vstack( [f['%s_affine' % l.name ], [0., 0., 1.]] )

                                # NASTY HACK - apply crunchy RGB scale if needed
                                if ('rgb_scale' in f) and ('RGB' in l.name):
                                    aff /= f['rgb_scale']

                                l.affine = aff # set affine of layer
                                points[-1].metadata['affine'] = l.affine
                                points[-1].data = kp.astype(float)

    viewer.layers.move_multiple([int(i) for i in np.argsort( [l.name for l in viewer.layers] )[:-1]])
    return points

def computeAffine( base_image : 'napari.layers.Image' ):
    """
    Use vertices of the coregistration targets to compute affine transforms that align the images.
    """
    viewer = napari.current_viewer() # get viewer

    # check some points are defined
    if not '%s [kp]' % base_image.name in viewer.layers:
        napari.utils.notifications.show_error("Please define coregistration targets for base image (%s)" % base_image.name)
        return

    # get target coordinates
    keypoints = {}
    for l in viewer.layers:
        if '[kp]' in l.name:
            if len(l.data) > 3:
                if ('image' in l.metadata) and ('affine' in l.metadata):
                    # get keypoints
                    xyz = np.hstack([l.data, np.ones((len(l.data), 1))])

                    # order keypoints
                    xyz = _order_keypoints(xyz)

                    # transform to image coords by applying inverse affine and store
                    keypoints[l.metadata['image']] = np.dot(xyz, l.metadata['affine'].inverse.affine_matrix.T)[:, :2]


    # reset affine of base image (as this is what we now georeference too)
    base_image.affine = np.eye(3)
    base_image.metadata['base'] = base_image.name
    base_points = viewer.layers['%s [kp]' % base_image.name]
    base_points.data = keypoints[base_image.name]
    base_points.metadata['affine'] = base_image.affine
    base_points.refresh()

    # update affine transform and points for other layers
    dst = keypoints[base_image.name]
    resid = []
    for k,src in keypoints.items():
        if k == base_image.name:
            continue
        #src, dst = _match_keypoints(src, dst)
        a, i = _est_affine(src, dst)

        # update image affine
        viewer.layers[k].affine = a
        viewer.layers[k].metadata['base'] = base_image.name

        # update keypoints positions
        xyz = np.hstack([src, np.ones((len(src), 1))])
        xyz = np.dot(xyz, a.T)
        resid.append( xyz[:,:2] - dst)
        viewer.layers['%s [kp]' % k].data = xyz[:,:2]
        viewer.layers['%s [kp]' % k].metadata['affine'] = viewer.layers[k].affine
        viewer.layers['%s [kp]' % k].refresh()
    return np.array(resid)

def _order_keypoints( kp ):
    """
    Order keypoints based on their relative angle to the local mean. For non-rotational affine transforms this should
    give reasonalbe matching regardless of the input order!
    """
    kp1c = kp - np.mean(kp, axis=0)
    ix1 = np.argsort(np.arctan2(kp1c[:, 0], kp1c[:, 1]))
    return kp[ ix1, : ]


def _est_affine(kp1, kp2, residual_threshold=3, max_trials=1000):
    # robustly estimate affine transform model with RANSAC
    model, inliers = ransac((kp1, kp2), AffineTransform, min_samples=3,
                            residual_threshold=residual_threshold, max_trials=max_trials)
    # outliers = (inliers == False)
    return np.array(model), inliers

def exportAffine():
    """
    Export the computed affine to a numpy array.
    """
    viewer = napari.current_viewer()  # get viewer

    out = {'target_shape' : () }
    dirs = []
    msg = 'Affine exported and saved to console.\n'
    for l in viewer.layers:
        if 'base' in l.metadata: # store target shape
            msg+='Affine Target Shape: (%d,%d)\n'%( viewer.layers[l.metadata['base']]._data_view.shape[0],
                                                    viewer.layers[l.metadata['base']]._data_view.shape[1] )
            out['target_shape'] = ( viewer.layers[l.metadata['base']]._data_view.shape[0],
                                    viewer.layers[l.metadata['base']]._data_view.shape[1] )
            out['keypoints'] = (viewer.layers[ '%s [kp]' % l.metadata['base'] ].data).reshape((-1,2))[:,[1,0]] # store keypoint (x,y) positions
        if 'path' in l.metadata: # store output path
            dirs.append( os.path.dirname(l.metadata['path']) )
        if isinstance(l, napari.layers.Image):
            out[l.name + '_affine'] = l.affine.affine_matrix[[0,1], :]
            msg+='%s_affine: %s\n' % ( l.name + '_affine', l.affine.affine_matrix[[0,1], :] )
    flist = []
    for p in np.unique(dirs):
        flist.append(os.path.join(p, 'crunchycoreg.npz'))
        np.savez(flist[-1], **out )
        napari.utils.notifications.show_info("Saved to %s." % flist[-1])

    napari.utils.notifications.show_info( msg )
    return np.unique(flist)

def resample():
    """
    Apply affine and resample the selected image(s). If none are selected then apply to all images.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers.selection)
    if len(layers) == 0:
        layers = list(viewer.layers)

    for l in layers:
        if isinstance(l, napari.layers.Image):
            a = AffineTransform( l.affine.affine_matrix )
            if l.rgb:
                tgt = ( viewer.layers[l.metadata['base']]._data_view.shape[0],
                                    viewer.layers[l.metadata['base']]._data_view.shape[1] )
                warped = warp(l.data, inverse_map=a.inverse, output_shape=tgt) # bands are last axis
            else:
                warped = warp( np.transpose(l.data, (1,2,0) ), inverse_map=a.inverse, output_shape=tgt ) # bands are first axis
            viewer.add_image(warped, metadata=l.metadata, name=l.name + ' [warped]')




