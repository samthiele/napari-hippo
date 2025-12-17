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
from napari.experimental import link_layers

from ._guiBase import GUIBase
from scipy.spatial import distance_matrix

from skimage.transform import AffineTransform, warp
from skimage.measure import ransac
from napari_hippo import getHyImage, h2n, n2h, getLayer, getMode, isImage, getByType, ROI, RGB, RGBA, BW, HSICube
import pathlib
from hylite import io
from napari_hippo._annotationTools import saveAnnot
import matplotlib.pyplot as plt
class CoregToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.simpleT_widget = magicgui( simpleT, call_button="Transforms")
        self.fit_widget = magicgui( fitExtent, call_button="Match Extent")
        self.addKP_widget  = magicgui(addKP, call_button='Add/Load Target(s)')
        self.sortKP_widget = magicgui(matchKP, call_button='Match New KPs')
        self.fitAffine_widget = magicgui(fitAffine, call_button='Fit Affine')
        self.save_widget = magicgui(save, call_button='Save')
        self.resample_widget = magicgui(resample, call_button='Apply Affine')

        function_widgets = [
            self.simpleT_widget,
            self.fit_widget,
            self.addKP_widget,
            self.sortKP_widget,
            self.fitAffine_widget,
            self.save_widget,
            self.resample_widget
        ]
        function_labels = [
            "Simple Transforms",
            "",
            "Coregister",
            "",
            "",
            "",
            "Resample"
        ]

        tutorial_text = (
            "<b>Simple Transforms:</b>  This feature is only for visualization. It will change the target image to base image.<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 1:</b> Select target images from the left 'layer list' panel<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 2:</b> From the right 'Coregister' panel, select 'base' image from the dropdown<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 3:</b> Click 'Match Extent'<br>"
            "<br>"
            "<b>Coregister:</b> This feature allows you to coregister multiple images based on keypoints.<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 1:</b> Select all the images from the left 'layer list' panel<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 2:</b> Click 'Add/Load Target(s)'<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 3:</b> In the [KP] layers, add keypoints for each image<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 4:</b> Click 'Match new KPs' <br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 5:</b> Select a base image from the dropdown<br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 6:</b> Click 'Fit affine' <br>"
            "&nbsp;&nbsp;&nbsp; <b>Step 7:</b> Click 'Save'<br>"
        )

        self.add_scrollable_sections(function_widgets, tutorial_text, function_labels, stretch=(2,1))
        
        # Store widgets for updating layer choices when layers are added/removed
        self.subwidgets = function_widgets
        
        # Connect viewer layer events to update widget choices when layers are added/removed
        napari_viewer.layers.events.inserted.connect(self._update_layer_choices)
        napari_viewer.layers.events.removed.connect(self._update_layer_choices)

def fitExtent( base : 'napari.layers.Image' ):
    """
    Match the width of the selected images to the width of the reference one.
    """
    viewer = napari.current_viewer()  # get viewer
    if not isImage(base):
        napari.utils.notifications.show_warning('Please select an image layer as reference')
        return
    reference = getLayer(base,viewer=viewer).toHyImage()

    layers = viewer.layers.selection
    if len(layers) == 0:
        layers = viewer.layers

    for L in getByType(layers, [HSICube, RGBA, RGB, BW]): 
        if L.layer == base:
            continue # skip
        
        # compute sf
        sf = reference.xdim() / L.toHyImage().xdim()

        # set affine
        L.layer.affine = np.array([[sf, 0, 0],
                                    [0, sf, 0], [0, 0, 1]])

def simpleT():
    """
    Build a side-panel for applying simple transformations.
    """
    viewer = napari.current_viewer()
    viewer.window.add_function_widget(transform, 
                                        magic_kwargs=dict(call_button='Transform',
                                        x_scale=dict(min=-np.inf, max=np.inf, step=0.005),
                                        y_scale=dict(min=-np.inf, max=np.inf, step=0.005),
                                        x=dict(min=-np.inf, max=np.inf, step=0.01),
                                        y=dict(min=-np.inf, max=np.inf, step=0.01),
                                        angle=dict(min=-np.inf, max=np.inf, step=0.005),
                                    ),
                                    name="Transform",
                                    area='left')

def transform( x_scale : float = 1, y_scale : float = 1, x : float = 0.0, y : float = 0.0, relative=True, angle : float = 0.0 ):
    """
    Apply a combined transformation to the specified layer.
    """
    scale(x_scale=x_scale, y_scale=y_scale)
    rot(angle=angle)
    translate(x=x,y=y,relative=relative)

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

def checkMode():
    """
    Ensure we don't do anything crazy in batch mode.
    """
    if getMode(napari.current_viewer()) != 'Image':
        napari.utils.notifications.show_error(
            "Coregistration does not work in Batch mode. Delete any [Stack] layers." )
        return False
    return True

def addKP():
    """
    Add / load editable keypoints layers for the selected layers.
    """
    if checkMode():
        viewer = napari.current_viewer()
        layers = viewer.layers.selection
        if len(layers) == 0:
            layers = viewer.layers
        
        out = []
        for i,l in enumerate(viewer.layers):
            if isImage(l):
                # N.B. this will load keypoint from header files
                # if they are defined! :-) 
                out.append( ROI.construct(l, mode='point', viewer=viewer).layer )
                out[-1].mode = 'ADD'

                # choose random color and set
                c = np.array(plt.cm.get_cmap('tab20', len(viewer.layers))(i))
                out[-1].face_color = [c for i in range(len(out[-1].face_color))]
                out[-1].opacity = 0.8
                out[-1].size = [4 for i in range(len(out[-1].size))]
                out[-1].current_face_color = [c]*100
                out[-1].current_size = [4]
                # link to allow easier toggling
                link_layers( [out[-1], l], ('visible',) )

                # check for affine matrix
                img = getLayer(l).toHyImage()
                if 'affine' in img.header:
                    # load affine matrix from header
                    A = np.eye(3)
                    A[0,:] = img.header.get_list('affine')[:3]
                    A[1,:] = img.header.get_list('affine')[3:6]

                    # set it for both image and keypoints
                    l.affine = A
                    out[-1].affine = A

        # viewer.layers.move_multiple([int(i) for i in np.argsort( [l.name for l in viewer.layers] )[:-1]])

        return out 
    return []

def matchKP():
    """
    Sort any unlabeled keypoints and name them based on a
    radial matching algorithm.
    """
    if checkMode():
        viewer = napari.current_viewer()

        # get keypoint layers
        R = [getLayer(l) for l in viewer.layers.selection if l.metadata.get('type','') == 'ROI']
        if len(R) < 2: # not enough in selection; check whole environment
            R = [getLayer(l) for l in viewer.layers if l.metadata.get('type','') == 'ROI']
        if len(R) < 2: # still not enough layers... show error and leave
            napari.utils.notifications.show_error(
            "Select multiple keypoint layers to match." )
            return

        # build dictionary of un-named points
        points = {}
        for k in R:
            verts, text = k.toList(world=True)
            verts = np.array([p for p,t in zip(verts,text) if t == '']) # unlabelled points
            ixx = [i for i,t in enumerate(text) if t == ''] # corresponding indices
            if len(verts) == 0:
                continue # no un-named keypoints here
            elif len(verts) == 1:
                angle = [0]
            else:
                v = verts - np.mean(verts, axis=0) # location relative to mean
                angle = np.arctan2(v[:, 0], v[:, 1])
            
            points[k] = (angle, ixx, verts ) # store angles

        # get reference names and angles
        # (from the smallest point set)
        ref_angle = None
        mincount = np.inf
        ref = None
        for k,(angle,idx,verts) in points.items():
            if (verts.shape[0] < mincount) and (verts.shape[0] > 0):
                ref = k
                ref_angle = angle
                mincount = verts.shape[0]
                names = np.array(["kp%d%d"%(p[0],p[1]) for p in verts])

        # do matching
        for k,(angle,idx,verts) in points.items():
            if (k == ref):
                match = names # easy!
            else:
                if len(angle) == 1:
                    match = names # edge case
                else:
                    distances = distance_matrix( angle[:,None], ref_angle[:,None] )
                    match = names[np.argmin( distances, axis=1)]
            
            # store matches
            used = []
            text = [str(v) for v in k.layer.text.values]
            for i,m in zip(idx,match):
                if m not in used:
                    text[i] = str(m)
                    #k.layer.text.values[i] = m
                    used.append(m)
            k.layer.text.values = text # set text array
            k.layer.refresh()

def fitAffine( base : 'napari.layers.Image' ):
    """
    Fit affine based on matches between all keypoint layers.
    """
    if checkMode():
        viewer = napari.current_viewer()
        layers = viewer.layers
        
        # get keypoint layers
        R = [getLayer(l) for l in layers if l.metadata.get('type','') == 'ROI']
        if len(R) < 2:
            napari.utils.notifications.show_error(
            "Select multiple keypoint layers to match." )
            return
        
        # get image layers associated with each keypoint layer
        I = [getLayer(l.base) for l in R]

        # get keypoints associated with base image
        m = [i for i,img in enumerate(I) if img.layer.name == base.name]
        if len(m) == 0:
            napari.utils.notifications.show_error(
            "No keypoints found for selected base layer (%s)."%base.name )
            return

        # get base image and associated keypoints and reset affine
        base_image = I[m[0]]
        base_hyimage = base_image.toHyImage()
        base_points = R[m[0]]
        base_image.layer.affine = np.eye(3)
        base_image.affine = np.hstack([np.eye(3)[0,:], np.eye(3)[1,:]]) # store flattened affine
        base_points.layer.affine = np.eye(3)
       

        bKP, bTxt = base_points.toList(world=base_image.layer)
        keypoints = dict(zip(bTxt, bKP))

        # compute affines for other layers
        for img,kp in zip(I,R):
            # store target shape
            img.target_shape = [base_hyimage.xdim(), base_hyimage.ydim()]
            if img == base_image: # affine here is known
                continue

            # find matching kps
            KP, Txt = kp.toList(world=img.layer)
            kp1 = np.array([to_2d(k) for k,t in zip(KP,Txt) if t in keypoints])
            kp2 = np.array([to_2d(keypoints[t]) for k,t in zip(KP,Txt) if t in keypoints])

            if len(kp1) <= 3:
                napari.utils.notifications.show_error(
                "Less than three matching keypoints found for %s."%img.layer.name )
                continue
            
            # fit affine
            affine, _ = _est_affine(kp1,kp2)
            img.layer.affine = affine # set layer affine
            img.affine = np.hstack([affine[0,:], affine[1,:]]) # store flattened affine
            
            # repeat, but for keypoints
            KP, Txt = kp.toList(world=False)
            kp1 = np.array([to_2d(k) for k,t in zip(KP,Txt) if t in keypoints])
            kp2 = np.array([to_2d(keypoints[t]) for k,t in zip(KP,Txt) if t in keypoints])
            affine, inliers = _est_affine(kp1,kp2)
            kp.layer.affine = affine

        # estimate residual for testing purposes
        KP, Txt = kp.toList(world=True) # N.B. this checks in world coords!
        kp1 = np.array([to_2d(k) for k,t in zip(KP,Txt) if t in keypoints])
        kp2 = np.array([to_2d(keypoints[t]) for k,t in zip(KP,Txt) if t in keypoints])
        resid = np.mean( np.abs( kp1[inliers] - kp2[inliers] ) )
        return resid # return residual for testing purposes

def to_2d(pt):
    '''
    Ensure keypoint is 2D by stripping leading zeros if needed.
    '''
    arr = np.asarray(pt)
    if arr.shape[0] == 2:
        return arr
    elif arr.shape[0] > 2 and np.all(arr[:arr.shape[0]-2] == 0):
        # Remove leading zeros if present
        return arr[-2:]
    else:
        napari.utils.notifications.show_error("Keypoint shape is %d, not 2D"%(arr.shape[0]))
        return arr

def save():
    """
    Save keypoints and associated affine transforms to disk.
    """
    viewer = napari.current_viewer()  # get viewer
    if checkMode():
        # save keypoints (easy, as they are just annotations!)
        saveAnnot(format = 'header')

        # save affine matrices also in header file
        I = getByType( viewer.layers, [HSICube, RGBA, RGB, BW])
        out = {}
        for img in I:
            if 'affine' in img.layer.metadata:
                pth = os.path.splitext(img.path)[0] + '.hdr'
                header = io.loadHeader(pth)
                header['affine'] = img.layer.metadata['affine']
                header['target_shape'] = img.layer.metadata['target_shape']
                io.saveHeader(pth, header)
                out[pth] = header # for testing

        napari.utils.notifications.show_info(
            "Saved affine and keypoints to header files." )
        return out

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

def resample( loadFromFile : pathlib.Path = None):
    """
    Opens a new widget that allows affine transforms to be applied to images or data cubes. 
    The path argument can be used to load affine transforms from a npz file, following the format
    used by `Save affine`. If None, available affines will be loaded from layers in the current
    viewer instead.
    """

    viewer = napari.current_viewer()  # get viewer

    # gather possible affine transforms
    A = {"Identity" : np.eye(3)} # available affine matrices
    if (loadFromFile is not None): # load from file
        print("Loading coreg points from %s" % str(loadFromFile))
        if 'npz' in loadFromFile.suffix:
            f = np.load(loadFromFile)
            for k in f.keys():
                if 'affine' in k:
                    n = k.split('_')[0]
                    aff = np.vstack( [f[k], [0., 0., 1.]] )

                    # NASTY HACK - apply crunchy RGB scale if needed
                    if ('rgb_scale' in f) and ('RGB' in k):
                        aff /= f['rgb_scale']
                    
                    A[n] = aff

    else: # load from layers
        for l in list(viewer.layers):
            if (l.affine.affine_matrix.shape[0] == 3) and \
                (not (l.affine.affine_matrix == np.eye(3)).all()):
                A[l.name] = l.affine.affine_matrix
    
    # contruct a function to do our dirty work
    def apply( source_image : 'napari.layers.Image' = None,
                affine : str = list(A.keys())[0],
                target_image : 'napari.layers.Image' = None,
                factor : int = 1 ):
        """
        Select and apply an affine transform to the selected layer(s).

        Args:
            - base_image = the result will be sampled to this images size.
            - affine = the affine matrix to apply (selected from another layer).
            - factor = a scaling factor to multiply the affine transform and output dimensions by.
                    Useful for preserving an image at e.g., 6x the resolution of another. Must be an integer.
        """
        if (source_image is None) or (target_image is None):
            napari.utils.notifications.show_warning("Please select a source and target image")
        viewer = napari.current_viewer()  # get viewer again (threading...)
        _A = A[affine][[0,1], :] # get relevant parts of affine
        dest,_ = getHyImage( viewer, target_image )
        target_shape = (dest.xdim()*factor, dest.ydim()*factor)
        for l in [source_image]:
            if isinstance(l, napari.layers.Image):
                source,_ = getHyImage( viewer, l )
                ishape = (source.xdim(), source.ydim()) # just for reference

                # construct skimage affine
                a = AffineTransform(np.vstack( [_A[0]*factor, _A[1]*factor, [0,0,1] ] ) )
                
                # apply
                source.data = warp( source.data, 
                                    inverse_map=a.inverse, 
                                    output_shape=target_shape+(source.data.shape[-1],))

                # add result
                print("Resampled image %s from shape %s to shape %s."%(l.name,
                                                                       ishape,
                                                                       target_shape ))
                viewer.add_image( h2n( source.data ), rgb=l.rgb,
                    name=l.name + " [warped]",
                    metadata=l.metadata )

    # construct GUI and show it
    #widget = magicgui(apply, 
    #                    call_button='Resample', 
    #                    affine={"choices":list(A.keys())},
    #                    auto_call=False)
    #widget.show()
    viewer.window.add_function_widget(apply, 
                                      magic_kwargs=dict(
                                          call_button='Resample', 
                                          affine={"choices":list(A.keys())},
                                          auto_call=False
                                      ),
                                      name="Resample",
                                      area='left')
