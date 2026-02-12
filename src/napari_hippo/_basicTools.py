from ._guiBase import GUIBase
from magicgui import magicgui

import pathlib
import numpy as np
from natsort import natsorted
import glob
import napari
from napari_hippo import napari_get_hylite_reader
from napari_hippo._base import *
import os
import hylite
from hylite import io

class BasicWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.query_widget = magicgui(search, call_button='Search', root={'mode': 'd'}, auto_call=False)

        self.load_mask_widget = magicgui(loadMasks, call_button='Load/Create Masks', 
                                         mode={"choices": ['filename', 'directory']} )
        self.save_mask_widget = magicgui(saveMasks, call_button='Save/Apply Masks', 
                                         mode={"choices": ['Save to file', 'Set as nan', 'Nan and crop']})

        function_widgets = [self.query_widget,
                            self.load_mask_widget,
                            self.save_mask_widget]

        function_labels = [
            "Batch",
            "Mask",
            "",
        ]

        tutorial_text = (
            "<b>Step 1:</b> TODO<br>"
            "Add more instructions here as needed.<br>"
            "You can extend this tutorial and it will remain scrollable.<br>"
            "Example:<br>"
            "<b>Step 1:</b> TODO<br>"
            "<b>Step 2:</b> TODO<br>"
            "<b>Step 3:</b> TODO<br>"
            "<b>Step 4:</b> TODO<br>"
            "<b>Step 5:</b> TODO<br>"
            "<b>Step 6:</b> TODO<br>"
            "<b>Step 7:</b> TODO<br>"
            "<b>Step 8:</b> TODO<br>"
            "<b>Step 9:</b> TODO<br>"
            "<b>Step 10:</b> TODO<br>"
            "<b>Step 11:</b> TODO<br>"
            "<b>Step 12:</b> TODO<br>"
            "<b>Step 13:</b> TODO<br>"
            "<b>Step 14:</b> TODO<br>"
            "<b>Step 15:</b> TODO<br>"
            "<b>Step 16:</b> TODO<br>"
            "<b>Step 17:</b> TODO<br>"
            "<b>Step 18:</b> TODO<br>"
            "<b>Step 19:</b> TODO<br>"
            "<b>Step 20:</b> TODO<br>"
        )

        self.add_scrollable_sections(function_widgets, tutorial_text, function_labels, stretch=(2,1))

        # Store widgets for updating layer choices when layers are added/removed
        self.subwidgets = function_widgets
        
        # Connect viewer layer events to update widget choices when layers are added/removed
        napari_viewer.layers.events.inserted.connect(self._update_layer_choices)
        napari_viewer.layers.events.removed.connect(self._update_layer_choices)

def search( root : pathlib.Path = pathlib.Path(''),
            filter : str='*.png',
            rgb_only = True,
            stack = True,
            stretch = False ):
    """
    Search for image files matching the query string and load them into Napari.

    Args:
        root: The root directory to search.
        filter: The query string. Should follow [glob syntax](https://man7.org/linux/man-pages/man7/glob.7.html).
        rgb_only: Only load the preview bands ('default bands' key in header file) of HSI images (rather than loading the whole data cube).
        stack: Stack loaded images into a single image file. Useful for flicking through large numbers of images.
        stretch: If True, loaded images will be scaled to have the same width.
    Returns:
        images: a list containing the image layer(s) added.
    """

    file_list = natsorted(glob.glob(str(root / filter), recursive=True) )

    # get viewer
    viewer = napari.current_viewer()

    # load images
    images = []
    paths = []
    for f in file_list:
        r = napari_get_hylite_reader( f )
        if r is not None:
            paths.append( f ) # store root path for stacked images
            images += r( f, force_rgb=rgb_only, return_data=True )
            
    if (len(images) == 0) & (getMode(napari.current_viewer()) != 'Batch'):
        napari.utils.notifications.show_warning("No files found.")
    if len(images) == 0:
        return []

    out = []
    if stack: # add image stack
        name = 'batch'
        out += [Stack.construct( images, paths, name, viewer=viewer ).layer]
    else: # add images individually
        for i,p in zip(images, paths):
            # create layer
            name = os.path.splitext( os.path.basename( p ) )[0]
            image = HSICube.construct(i, name, viewer=viewer, path=p )
            out.append(image.layer)

            # sort out stretch
            if stretch and len(out) > 1:
                 # find scale factor that fits to target shape
                 target_shape = out[0].data.shape
                 s = min( float(target_shape[0]) / image.layer.data.shape[0], float(target_shape[1]) / image.layer.data.shape[1] )

                 # define affine matrix
                 affine = np.array([[s, 0, 0],
                                    [0, s, 0], [0, 0, 1]])
                
                 # set layer affine
                 out[-1].affine = affine
        
        viewer.reset_view()
        viewer.dims.set_point(0, 0)
        napari.utils.notifications.show_info("Loaded %d images." % len(images))

        return out
    
def loadMasks( mode='filename' ):
    """
    Load mask files associated with an cube, slice or image stack.

    If mode is `filename` then masks are expected to match the name of the source
    of the image being masked, but with a "_mask" suffix. If `directory` is selected
    then masks are expecte to be called "mask.hdr" in the same directory as the source
    file.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers.selection) # get selected layers
    if len(layers) == 0:
        layers = list(viewer.layers)
    for l in layers:
        Mask.construct( l, viewer, mode=mode ) # construct new masks

def saveMasks(mode='Save to file'):
    """
    Apply or save masks. Depending on the mode, these will be saved to disk or 
    applied (to nan masked pixels in the selected image), with the option to also
    crop the masked image to the mask region.
    """
    viewer = napari.current_viewer()  # get viewer

    # save masks to disk
    for l in viewer.layers:
        if l.metadata.get('type', '') == 'Mask':
            mask = getLayer(l, viewer=viewer)
            if 'nan' in mode.lower():
                mask.apply(crop='crop' in mode.lower())
            elif 'save' in mode.lower():
                mask.save()
            else:
                assert False, "Error - invalid mode %s"%mode

def loadGeometry():
    """
    Loads geometries (points, lines, polygons) stored in image header files.

    Works with cubes, slices and stacks.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers.selection)
    if len(layers) == 0:
        napari.utils.notifications.show_warning("Please select a layer to load geometry for")
        return
    
    if 'Points' in viewer.layers:
        viewer.layers.remove('Points')
    if 'ROI' in viewer.layers:
        viewer.layers.remove('ROI')

    # load for first layer (only)
    l = layers[0]
    path = l.metadata.get('path', [])
    if not isinstance(path, list):
        path = [path]

    point_names = []
    points = []
    roi_names = []
    roi = []

    # load geometry
    for i, p in enumerate(path):
        p = os.path.splitext(p)[0] + ".hdr"
        if os.path.exists(p):
            h = hylite.io.loadHeader(p)
            for k,v in h.items():
                if k.lower().startswith( 'point' ) or k.lower().startswith('roi'):
                    name = k.split(' ')[-1]
                    verts = h.get_list(k).reshape((-1,2))
                    if 'point' in k:
                        if len(path) > 1: # stack require 3D shapes
                            points += [ (i, p[0], p[1]) for p in verts ] # concatenate points
                        else: # RGB or cube require 2D shapes
                            points += [ (p[0], p[1]) for p in verts  ]
                        point_names += [name for p in verts ]
                    else:
                        if len(path) > 1: # stack require 3D shapes
                            roi.append( [(i, p[0], p[1]) for p in verts] ) # add new polygon
                        else: # RGB or cube require 2D shapes
                            roi.append( [(p[0], p[1]) for p in verts] ) # add new polygon
                        roi_names.append(name)
    # add points layer
    if len(points) > 0:
        meta = {'path':l.metadata.get('path')}
        points_layer = viewer.add_points(points, 
                                         name='Points', 
                                         text=point_names,
                                         metadata=meta)
        points_layer.mode = 'SELECT'
    
    # add ROI layer
    if len(roi) > 0:
        meta = {'path':l.metadata.get('path')}
        roi_layer = viewer.add_shapes(roi, name='ROI', 
                                      text=roi_names, 
                                      shape_type='polygon',
                                      metadata=meta)
        roi_layer.mode = 'SELECT'
        
def updateGeometry( name : str = 'napari' ):
    """
    Update the geometry of the selected layer based on the New ROIs and New Points layers. Creates these shape layers if needed.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers.selection)
    if len(layers) == 0:
        napari.utils.notifications.show_warning("Please select a layer to load geometry for")
        return

    # remove spaces in name as these are not compatible in header file
    name = name.replace(' ', '_')

    # get path of header files
    l = layers[0]
    if 'path' in l.metadata:
        path = l.metadata['path']
        if not isinstance(path, list):
            path = [path]
    else:
        napari.utils.notifications.show_warning("Please select an image layer that was loaded with napari-hippo")
        return
    
    # add new points
    if 'New Points' in viewer.layers:
        new_points = viewer.layers['New Points']
        new_roi = viewer.layers['New ROIs']
        
        # get or create header files
        headers = []
        edited = []
        for p in path:
            p = os.path.splitext(p)[0] + ".hdr"
            if os.path.exists(p):
                headers.append( hylite.io.loadHeader(p) )
            else:
                headers.append( hylite.HyHeader() )
            edited.append(False)

            # clean header file from previous entries
            roi_keys = [k for (k,v) in headers[-1].items() if k.lower().startswith('roi')]
            point_keys = [k for (k,v) in headers[-1].items() if k.lower().startswith('point')]
            if 'ROI' in viewer.layers:
                for k in roi_keys:
                    del headers[-1][k]
                    edited[-1] = True
            if 'Points' in viewer.layers:
                for k in point_keys:
                    del headers[-1][k]
                    edited[-1] = True

            # add in old points
            if 'Points' in viewer.layers:
                for i, p in enumerate(viewer.layers['Points'].data):
                    n = viewer.layers['Points'].text.values[i]
                    if len(headers) > 1: # stack
                        header = headers[int(p[0])]
                        edited[int(p[0])] = True
                    else:
                        header = headers[0]
                        edited[0] = True
                    if 'point %s' % n in header:  # append new point to feature
                        prev = header.get_list('point %s' % n)
                        if len(headers) > 1: # stack
                            header['point %s' % n] = np.hstack([prev, p[1:]])
                        else:
                            header['point %s' % n] = np.hstack([prev, p])

                    else:  # create feature
                        if len(headers) > 1: # stack
                            header['point %s' % n] = p[1:]
                        else:
                            header['point %s' % n] = p.ravel()
            
            # add old ROIs
            if 'ROI' in viewer.layers:
                for i,p in enumerate(viewer.layers['ROI'].data):
                    n = viewer.layers['ROI'].text.values[i]
                    if len(headers) > 1: # stack
                        header = headers[int(p[0, 0])]
                        edited[int(p[0, 0])] = True
                    else: # slice or cube
                        header = headers[0]
                        edited[0] = True
                    
                    if len(headers) > 1: # stack
                        header['roi %s' % n] = p[:, 1:].ravel()
                    else: # slice or cube
                        header['roi %s' % n] = p.ravel()
            

            # add new points
            if len(new_points.data) > 0:
                for p in new_points.data:
                    if len(path) > 1: # this is a stack
                        print("Adding new point '%s' to %s" % (name, path[ int(p[0]) ]))
                        header = headers[int(p[0])]
                        edited[int(p[0])]=True
                    else:
                        print("Adding new point '%s' to %s" % (name, path[ 0 ]))
                        header = headers[0]
                        edited[0]=True
                    if 'point %s'%name in header: # append new point to feature
                        prev = header.get_list('point %s'%name)
                        if len(path) > 1: # this is a stack
                            header['point %s' % name] = np.hstack([prev, p[1:]])
                        else:
                            header['point %s' % name] = np.hstack([prev, p])

                    else: # create feature
                        if len(path) > 1: # this is a stack
                            header['point %s'%name] = p[1:]
                        else:
                            header['point %s'%name] = p.ravel()
                    

            # add new ROIs
            if len(new_roi.data) > 0:
                for p in new_roi.data:
                    if len(path) > 1: # this is a stack
                        print("Adding new ROI '%s' to %s" % (name, path[int(p[0, 0])]))
                        header = headers[int(p[0,0])]
                        edited[int(p[0,0])]=True
                    else:
                        print("Adding new ROI '%s' to %s" % (name, path[0]))
                        header = headers[0]
                        edited[0]=True
                    
                    if len(path) > 1: # this is a stack
                        header['roi %s'%name] = p[:,1:].ravel()
                    else: # slice or cube
                        header['roi %s'%name] = p.ravel()
                    
            # save
            for h,e,p in zip(headers, edited, path):
                if e: # save header file
                    p = os.path.splitext(p)[0] + '.hdr'
                    io.saveHeader(p, h)

    # reload
    loadGeometry()

    # refresh editing layers
    if 'New ROIs' in viewer.layers:
        viewer.layers.remove('New ROIs')
    if 'New Points' in viewer.layers:
        viewer.layers.remove('New Points')
    meta = {'path':l.metadata.get('path')}
    if len(path) > 1: # stack; must have a 3D shapes
        ndim = 3
    else:
        ndim = 2

    new_points = viewer.add_points(None, ndim=ndim, name='New Points', opacity=0.5,
                metadata=meta)
    new_points.mode = 'ADD'
    new_roi = viewer.add_shapes(None, ndim=ndim, name='New ROIs', edge_color=np.array((0, 1, 1)),
                                face_color=np.array((1, 1, 1, 0)), edge_width=2, opacity=0.5,
                                metadata=meta)
    new_roi.mode = 'add_polygon'


