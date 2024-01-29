from ._guiBase import GUIBase
from magicgui import magicgui

import pathlib
import numpy as np
from natsort import natsorted
import glob
import napari
from napari_hippo import napari_get_ENVI_reader
from napari_hippo._util import getHyImage, n2h
import os
import hylite
from hylite import io

class IOWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        #btn = QPushButton("Load with query")
        #btn.clicked.connect(self._query)

        self.query_widget = magicgui(search, call_button='Search', root={'mode': 'd'}, auto_call=False)
        self.info_widget = magicgui(info, call_button="Show Layer Info")
        self._add([self.query_widget, self.info_widget], 'Import')

        self.load_mask_widget = magicgui(loadMasks, call_button='Load Masks', mode={"choices": ['filename', 'directory']})
        self.save_mask_widget = magicgui(saveMasks, call_button='Save Masks')
        self.update_mask_widget = magicgui(updateMasks, call_button='Update')
        self._add( [self.load_mask_widget, self.save_mask_widget,
                     self.update_mask_widget], 'Mask' )

        self.updateGeometry_widget = magicgui(updateGeometry, call_button="Save/Load")
        self.exportPatch_widget = magicgui(exportPatches, call_button='Export Patches',
                                             filename={"mode": "w", "filter":"*.hyc"})
        self._add( [self.updateGeometry_widget,self.exportPatch_widget], 'Geometry')

# /Users/thiele67/Documents/Python/napari-hippo/src/napari_hippo/testdata
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
        stretch: Stretch images so they are approximately the same. Note that this is incompatible with the stack argument.

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
        r = napari_get_ENVI_reader( f )
        if r is not None:
            paths.append( f ) # store root path for stacked images
            images += r( f, force_rgb=rgb_only )
            if stack:
                images[-1][1]['name'] = images[-1][1]['name'].replace('[slice]','[stack]')
    if len(images) == 0:
        napari.utils.notifications.show_warning("No files found.")
        return []

    out = []
    if stack: # add image stack
        for i in range(len(images)-1, -1, -1):
            if not ( (images[i][0].shape[-1] == 3) or (images[i][0].shape[-1] == 4) ):
                del images[i] # drop non-RGB images
                napari.utils.notifications.show_warning("Warning: non-rgb images cannot be stacked.")

        # build dask array stack as it allows images to have different sizes
        try:
            import dask
        except:
            napari.utils.notifications.show_warning("Warning: images cannot be stacked without having dask installed.")
            return []

        shape = np.max([i[0].shape for i in images], axis=0)
        stack = dask.array.stack([dask.array.from_delayed(dask.delayed(i[0]), shape=shape, dtype=i[0].dtype,
                                                          ) for i in images])
        out += [viewer.add_image( stack, **images[0][1] )]
        out[-1].metadata['path'] = paths
        out[-1].metadata['type'] = 'STACK'

        viewer.reset_view()
        viewer.dims.set_point(0,0)
        napari.utils.notifications.show_info("Loaded stack of %d images." % len(images))

    else: # add images individually
        if stretch:
            target_shape = images[0][0].shape
            for image, kwargs, name in images:
                # find scale factor
                s = min( float(target_shape[0]) / image.shape[0], float(target_shape[1]) / image.shape[1] )

                # define affine matrix
                affine = np.array([[s, 0, 0],
                                   [0, s, 0], [0, 0, 1]])
                # add to kwargs
                kwargs['affine']  = affine

                # filthy hack: ignore _aruco in name (for SiSuRock coreg workflows)
                if '_aruco' in kwargs['name']:
                    kwargs['name'] = kwargs['name'].split('_')[0]

        for image, kwargs, name in images:
            out += [viewer.add_image(image, **kwargs)]

        viewer.reset_view()
        viewer.dims.set_point(0, 0)
        napari.utils.notifications.show_info("Loaded %d images." % len(images))

        return out

def info():
    """
    Displays metadata on the selected image as a notification.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers.selection)
    if len(layers) > 0:
        l = layers[0]

        # get data
        path = l.metadata.get('path', 'undefined')
        if isinstance(path, list):
            path = path[viewer.dims.current_step[0]]
        name = os.path.basename( os.path.dirname(path) ) + '/' + os.path.basename(path)

        # create message
        msg = '%s (%s)\n'%(name, l.metadata.get('type', 'undefined'))
        for k,v in l.metadata.items():
            if k not in ['type']: # add other metadata if it exists
                print(k, ': ', v ) # print metadata to console for reference too
                line = ''
                if isinstance(v, list) or isinstance(v, np.ndarray):
                    line="%s: %s"%(k, str(v[viewer.dims.current_step[0]]))
                else:
                    line="%s: %s"%(k, str(v))

                # clip for clarity
                if len(line) > 30:
                    line = line[:30] + "..." + line[-15:]
                msg+=line+"\n"

        # show message
        napari.utils.notifications.show_info(msg)

def loadMasks( mode : str = 'filename' ):
    """
    Load mask files associated with an cube, slice or image stack.

    Args:
        mode = the matching mode to identify mask files. Options are:
                - filename: masks will be envi files (.hdr extension) with the same filename
                            as the source image (must be .png, .jpg or similar).
                - directory: masks will be envi files called mask.hdr in the same directory as the
                            source image.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers.selection)
    if len(layers) == 0:
        layers = list(viewer.layers)

    for l in layers:
        if 'path' in l.metadata:

            # get paths to load
            if l.metadata.get('type', '') == 'STACK':
                paths = l.metadata['path']
            elif len(layers) == 1:
                paths = [l.metadata['path']]
            else:
                continue

            # load or create masks
            maskpaths=[]
            data = []
            for i, p in enumerate(paths):
                if mode == 'filename':
                    p = os.path.splitext(p)[0] + ".hdr"
                elif mode == 'directory':
                    p = os.path.join( os.path.dirname(p), 'mask.hdr' )
                else:
                    assert False, "Error - invalid mode %s" % mode

                maskpaths.append(p)
                if os.path.exists(p):
                    print("Loading mask for %s" % p)
                    data.append(io.load(p).data[:,:,0].T.astype(int))
                else:
                    if l.metadata.get('type', '') == 'STACK':
                        shape = np.array( l.data[i] ).shape
                    elif (len(l.data.shape) == 3) and (l.data.shape[-1] > 4):  # data cube; (band, x, y)
                        shape = np.array(l.data[0].shape)
                    else:  # conventional RGB or greyscale (x,y,band)
                        shape = np.array([l.data.shape[0], l.data.shape[1]])

                    print("Creating mask for %s with shape %s." % (p,shape))
                    data.append( np.zeros( (shape[0], shape[1]), dtype=int) )

            # build dask array
            try:
                import dask
            except:
                assert False, "Error - please install dask before building image stacks!"
            shape = np.max([d.shape for d in data], axis=0)
            for i in range(len(data)):
                if data[i] is None:
                    data[i] = np.zeros(shape, dtype=int)
            stack = dask.array.stack([dask.array.from_delayed(dask.delayed(d), shape=shape, dtype=int,
                                                              ) for d in data])
            viewer.add_labels(stack, name='mask', metadata=dict(type='MASK', path=maskpaths))


    # add polygon layers
    for n, c in zip(['include', 'exclude'], [(0, 1, 1), (1, 0, 0)]):
        l = viewer.add_shapes(None, ndim=3,
                          name=n,
                          edge_color=np.array(c),
                          face_color=np.array((1, 1, 1, 0)),  # transparent
                          edge_width=2, opacity=0.5)
        l.mode = 'add_polygon'

def saveMasks():
    viewer = napari.current_viewer()  # get viewer

    # update selected layers
    for l in viewer.layers:
        if l.metadata.get('type', '') == 'MASK':
            for i,p in enumerate(l.metadata.get('path',[])):
                mask = hylite.HyImage( np.array( l.data[i] ).T[...,None] )
                mask.data = mask.data.astype(np.uint8)
                print("Saving mask to %s" % p )
                io.save(p, mask )

def updateMasks():
    """
    Update masks based on include and exclude polygons.
    """
    viewer = napari.current_viewer()  # get viewer

    # update selected layers
    for l in viewer.layers:
        if l.metadata.get('type', '') == 'MASK':

            # get masks for include and exclude
            eMask = None
            iMask = None
            if 'exclude' in viewer.layers:
                eMask = viewer.layers['exclude'].to_masks(l.data.shape)
            if 'include' in viewer.layers:
                iMask = viewer.layers['include'].to_masks(l.data.shape)

            stack=[]
            for i in range(l.data.shape[0]):
                mask = np.array( l.data[i] )
                if (eMask is not None) and (len(eMask) > 0):
                    sub = eMask[:,i, :mask.shape[0], :mask.shape[1]].any(axis=0)
                    if sub.any():
                        mask[sub] = 0
                if (iMask is not None) and (len(iMask) > 0):
                    add = iMask[:,i, :mask.shape[0], :mask.shape[1]].any(axis=0)
                    if add.any():
                        mask[add] = 1
                stack.append(mask)

            # rebuild dask stack
            try:
                import dask.array
                import dask.delayed
            except:
                napari.utils.notifications.show_warning(
                    "Warning: stacked masks cannot be edited without dask installed.")
                return

            shape = np.max([i.shape for i in stack], axis=0)
            stack = dask.array.stack([dask.array.from_delayed(dask.delayed(i), shape=shape, dtype=i.dtype,
                                                              ) for i in stack])

            # update layer data
            l.data = stack
            l.refresh()

    # clear exclude and include polygons
    viewer.layers['exclude'].data = []
    viewer.layers['include'].data = []

    print("--")

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

def exportPatches( filename : pathlib.Path = pathlib.Path('.'),
                   plot = True ):
    """
    Export each ROI to separate images, stored in a HyCollection directory structure. Spectral
    libraries for any point data (and averages of each ROI) will also be created. If `plot` is True 
    (default) then figures will be created showing these libraries.
    """
    if filename == '.':
        napari.utils.notifications.show_warning("Please select a path to save output patches.")
        return
    
    # get layers
    viewer = napari.current_viewer()  # get viewer
    layers = list(viewer.layers)

    # get ROI and Points layer
    points = None
    roi = None
    if 'Points' in viewer.layers:
        points = viewer.layers['Points']
    if 'ROI' in viewer.layers:
        roi = viewer.layers['ROI']
    if (points is None) and (roi is None):
        napari.utils.notifications.show_warning("No Points or ROI layers found.")
        return
    
    # create output HyCollection
    O = hylite.HyCollection(os.path.basename(filename), os.path.dirname(filename))

    # loop through images
    for l in viewer.layers:
        if 'type' in l.metadata:

            # is this a stack?
            if 'stack' in l.metadata['type'].lower():
                paths = l.metadata['path']
                #images = [io.load(p) for p in paths]
                pass # not implemented yet

            elif ('hsi' in l.metadata['type'].lower()) or \
                    ('rgb' in l.metadata['type'].lower()):
                if 'hsi' in l.metadata['type'].lower():
                    image, _ = getHyImage(viewer, layer=l)
                else:
                    try: # look for an ENVI file with the same name
                        image = io.load(os.path.splitext(l.metadata['path'])[0] + ".hdr" )
                    except:
                        continue # skip this one

                img_name = os.path.splitext( os.path.basename( l.metadata['path'] ) )[0]
                labels = points.text.values
                # extract points
                if points is not None:
                    spectra = []
                    names = []
                    for i,(py,px) in enumerate(points.data):
                        print("Extracting spectra from pixel (%d,%d) in image %s with %d bands"%(px,py,img_name, image.band_count()))
                        if (px > 0) and (py > 0):
                            if (px < image.xdim()) and (py < image.ydim()):
                                spectra.append(image.data[int(px),int(py),:])
                                if i < len(labels):
                                    names.append("P%d_%s"%(i,labels[i]))
                                else:
                                    names.append("P%d"%i)
                    lib = hylite.HyLibrary( 
                            np.array(spectra)[:,None,:],
                            lab=names, wav=image.get_wavelengths()  )
                    
                    # plot?
                    if plot:
                        fig,ax = lib.quick_plot()
                        fig.canvas.manager.set_window_title(img_name+"_points")
                        fig.show()
                    O.set(img_name+"_points", lib)
                    O.save()
                    O.free()

                # extract ROIs 
                if roi is not None:
                    labels = roi.text.values
                    masks = roi.to_masks((image.ydim(), image.xdim()))
                    spectra = [] # for average spectra library
                    names = [] 
                    for i,m in enumerate(masks):
                        m = np.array( m ).T # convert mask to hylite layout

                        print("Exporting ROI %s from image %s with %d bands."%(labels[i], img_name, image.band_count()))
                        xmin = np.argmax(m.any(axis=1))
                        xmax = m.shape[0] - np.argmax(m.any(axis=1)[::-1])
                        ymin = np.argmax(m.any(axis=0))
                        ymax = m.shape[1] - np.argmax(m.any(axis=0)[::-1])
                        
                        # construct output patch
                        img = hylite.HyImage( image.data[xmin:xmax,ymin:ymax,:].copy() )
                        img.set_wavelengths( image.get_wavelengths() )
                        try:
                            img.data[ ~m[xmin:xmax,ymin:ymax] ] = np.nan # float data
                        except:
                            img.data[ ~m[xmin:xmax,ymin:ymax] ] = 0 # int data
                        if image.has_band_names():
                            img.set_band_names(image.get_band_names())
                        
                        # save it
                        S = O.addSub(str(labels[i]))
                        S.set(img_name, img)
                        S.save()
                        S.free()

                        # compute average for spectral library
                        names.append(str(labels[i]))
                        spectra.append(np.nanpercentile(img.data, (5,50,95), axis=(0,1)))
                    
                    lib = hylite.HyLibrary( np.array(spectra), names, wav=img.get_wavelengths())

                    # plot?
                    if plot:
                        fig,ax = lib.quick_plot()
                        fig.canvas.manager.set_window_title('%s_ROI'%img_name)
                        fig.show()

                    # save
                    O.set('%s_ROI'%img_name, lib)
                    O.save()
                    O.free()

