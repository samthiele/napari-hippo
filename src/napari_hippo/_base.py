"""
Define the base data classes that Hippo uses to manage seamless transition of various data formats
between the napari GUI and e.g., hylite.
"""

import hylite
import napari
import numpy as np
import os
from pathlib import Path

def isImage( layer ):
    """
    Return True if the specified napari layer is associated with
    an HSIImage, RGB, RGBA or BW type.
    """
    t = layer.metadata.get('type', '')
    return ('HSICube' in t) \
        or ('RGB' in t) \
          or ('RGBA' in t) \
            or ('BW' in t)

def getMode( viewer ):
    """
    Return the current mode of the specified viewer. Options are "Batch" (if multiple datasets are loaded as stack objects) or "Image". 
    If no Image or Stack layers are present, this will return None.
    """
    image = False
    for l in viewer.layers:
        t = l.metadata.get('type', '')
        if isImage(l):
            image = True
        if 'Stack' in t:
            return 'Batch'
    if image:
        return 'Image'
    return '' # mode is undefined

def getLayer( layer, viewer=None ):
    """
    Get the specified napari layer (object or string name) as it's corresponding
    HippoData type. If no corresponding type exists, this will return None.
    """
    # get layer
    if isinstance(layer, str):
        if viewer is None:
            viewer = napari.curent_viewer()
        if layer in viewer.layers:
            layer = viewer.layers[layer]
        else:
            return None
    
    dtype = layer.metadata.get('type', 'None')
    for cls in [HippoData,HSICube,BW,RGB,RGBA,Stack,Mask,ROI,Scene,View]:
        if cls.__name__ == dtype:
            return cls(layer)
        
    # no matches
    return None

def getByType( layers, layer_type ):
    """
    Filter the selected layers list (e.g., selection) and return 
    only those layers with the specified hippo type.
    """
    out = []
    if not isinstance(layer_type, list):
        layer_type = [layer_type]

    for l in layers:
        I = getLayer(l)
        for t in layer_type:
            if isinstance(I, t):
                out.append(I)
                continue
    return out
class HippoData( object ):
    """
    A base class for all hippo data objects, containing various shared functions.
    """

    def __init__(self, layer, **kwargs):
        """
        Construct a HippoData object and it's link to an existing napari layer object. Keyword arguments are used to flag
        properties that are shared between the data object and the napari layers metadata dictionary. Note that to ensure safe
        multi-threading these should be kept simple (i.e. primitive types or small numpy arrays).
        """
        self.layer = layer # store link to layer

        # pull also any properties that exist already in the layer metdata
        for k,v in self.layer.metadata.items():
            if k not in kwargs:
                kwargs[k] = v

        # convert to class attributes (and also add as keys in metadata)
        for k,v in kwargs.items():
            if (' ' in k) or ('-' in k) or ('.' in k):
                continue # this is a problem, but lets try to roll with it (ignorance is bliss!)
                #assert False, "Error - invalid metadata key for HippoData object." # this is a problem, but should never happen!
            self.__setattr__(k, v) # n.b. this will also update class metadata!

        # also store class type
        self.type = self.__class__.__name__

        # ensure name is correctly set
        self.layer.name = self.getTypedName()

    def __setattr__( self, name : str, value ):
        object.__setattr__(self, name, value ) # set class attribute
        if '__' not in name: # ignore private attributes
            self.layer.metadata[name] = value # update metadata
    
    def getName(self):
        """
        Return the name for this layer without the type prefix.
        """
        name = self.layer.name.strip()
        if len(name.split(']')) > 1:
            name = name.split(']')[1]
        return name.strip()
    def getTypedName(self):
        """
        Return the typed name for this layer by prefixing the current layer name
        with [DataType] if necessary.
        """
        name = self.getName()
        typestr = "[%s]"%self.type
        return f"{typestr} {name.strip()}"
    
    # placeholder functions
    def ndim(self):
        """
        Data classes should override this such that it returns: -1 for non-data, 0 for points, 1 for lines, 2 for 2D images (including RGB), 3 for data cubes and 4 for data stacks.
        """
        return -1
    
    def toHyImage(self):
        """
        Data classes should override this such that it returns a HyImage copy of the layer data (with appropriate wavelengths etc.) if possible. 
        """
        return None
    
    def fromHyImage(self, image : hylite.HyImage):
        """
        Data classes should override this such that it updates the underlying image layer data array and metadata (wavelength arrays) according to the specified HyImage `image`.
        """
        pass

class HSICube( HippoData ):
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

        # add image shape to metadata (n.b. we use n2h as napari uses 
        # inconsistent axes for the bands dimension.... how annoying.
        self.xdim = self.cols = n2h(layer.data).shape[0]
        self.ydim = self.rows = n2h(layer.data).shape[1]
        self.bands = n2h(layer.data).shape[-1]
    @classmethod
    def construct( cls, image, name, viewer=None, args_only=False, **kwds ):
        """
        Factory function that takes a HyImage instance, creates a new napari 
        layer based on it, and returns a corresponding HSICube object.

        Params:
            - image = a HyImage dataset to construct the napari layer for.
            - name = a name for this layer.
            - viewer = a viewer to add the layer too, or None (use napari.current_viewer())
            - args_only = If True, don't create a HippoData instance (and napari layer), but rather 
                          return the (data, args) tuple expected by napari.add_image( ... ).
        Keywords:
            - keywords are added to the layer metadata as keys.
        """

        if viewer is None:
            viewer = napari.current_viewer()
        
        # get data array
        arr = h2n( image.data )

        # get data type
        dtype = 'HSICube'
        if image.band_count() == 1:
            dtype = 'BW'
        elif image.band_count() == 3:
            dtype = 'RGB'
        elif image.band_count() == 4:
            dtype = 'RGBA'

        # define kwargs for the viewer.add_image method
        add_kwargs = dict(name=name, 
                            metadata=dict())
        for k,v in image.header.items(): # add all image header info to metadata
            add_kwargs['metadata'][k.replace(' ', '_')] =  v
        for k,v in kwds.items(): # add kwarg info into metadata
            add_kwargs['metadata'][k.replace(' ', '_')] =  v
        add_kwargs['metadata']['type'] = dtype
        add_kwargs['name'] = '[%s] %s'%(dtype, name)
        if args_only:
            return (arr, add_kwargs, 'image')

        # add layer and return relevant HSIData instance
        layer = viewer.add_image( arr, **add_kwargs )
        if image.band_count() == 1:
            return BW(layer)
        elif image.band_count() == 3:
            return RGB(layer)
        elif image.band_count() == 4:
            return RGBA(layer)
        else: 
            return HSICube( layer )
    
    def toHyImage(self):
        """
        Return a HyImage view of this data cube (easy - just reshape!).
        """
        img = hylite.HyImage( n2h(self.layer.data) )
        for k,v in self.layer.metadata.items():
            img.header[k.replace('_',' ')] = v
        return img

    def fromHyImage(self, image : hylite.HyImage):
        """
        Update layer data and metadata to match a HyImage object.
        """
        # copy data from header to this object
        for k,v in image.header.items():
            self.__setattr__( k.replace(' ', '_'), v )

        # set image data
        self.layer.data = h2n( image.data )

    def ndim(self):
        return 3

class BW( HSICube ):
    """
    Hippo data class for handling greyscale imagery.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self.bands = 1
    def ndim(self):
        return 2

class RGB( HSICube ):
    """
    Hippo data class for handling ternary (RGB) imagery.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    def ndim(self):
        return 2

class RGBA( HSICube ):
    """
    Hippo data class for handling ternary (RGBA) imagery.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    def ndim(self):
        return 2

class Stack( HippoData ):
    """
    Hippo data class for handling stacks of ternary (RGB) images.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

        # add image shape to metadata
        self.stack_xdim = self.stack_cols = [l.compute().shape[1] for l in layer.data]
        self.stack_ydim = self.stack_rows = [l.compute().shape[0] for l in layer.data]
        self.stack_bands = [l.shape[2] for l in layer.data]
        self.xdim = self.cols = np.max( self.stack_xdim )
        self.ydim = self.rows = np.max( self.stack_ydim )
        self.bands = np.max( self.stack_bands )

    def ndim(self):
        return 4
    
    @classmethod
    def construct( cls, images : list, paths : list, name : str, viewer=None, args_only=False, **kwds ):
        """
        Factory function that takes a list of HyImage instances, creates a new napari 
        layer based on it, and returns a corresponding Stack object.
        """
        # check image shape
        warn = False
        for i in range(len(images)-1, -1, -1):
            if not ( (images[i].band_count() == 3) or (images[i].band_count() == 4) ):
                del images[i] # drop non-RGB images
                del paths[i]
                warn = True
        if warn:
            napari.utils.notifications.show_warning("Warning: non-rgb images cannot be stacked.")

        # build dask array stack as it allows images to have different sizes
        try:
            import dask
        except:
            napari.utils.notifications.show_warning("Warning: images cannot be stacked without having dask installed.")
            return None

        arr = [h2n(i.data) for i in images]
        shape = np.max([i.shape for i in arr], axis=0)
        stack = dask.array.stack([dask.array.from_delayed(dask.delayed(i), shape=shape, dtype=i.dtype,
                                                          ) for i in arr])
        
        # define kwargs for the viewer.add_image method from the first image
        add_kwargs = dict(name=name, 
                          metadata=dict(**kwds))
        for k,v in images[0].header.items(): # add all image header info to metadata
            add_kwargs['metadata'][k.replace(' ', '_')] =  v
        add_kwargs['metadata']['type'] = 'Stack'
        add_kwargs['metadata']['path'] = list(paths)

        if args_only:
            return (stack, add_kwargs, 'image')
        
        if viewer is None:
            viewer = napari.current_viewer()

        layer = viewer.add_image( stack, **add_kwargs )
        viewer.reset_view()
        viewer.dims.set_point(0,0)
        return cls(layer)
    
    def toHyImage(self):
        """
        Convert this Stack to a list of HyImages. Note that these will have a 
        path attribute in their header files.
        """
        images = []
        for i,l in enumerate(self.layer.data):
            images.append( hylite.HyImage( n2h( l.compute() ) ) )
            images[-1].header['path'] = self.path[i]
        return images
    
    def fromHyImage(self, image):
        paths = [i.header.get('path','') for i in image]
        data, meta, _ = self.construct( image, paths, self.getName,
                                viewer='notNone', args_only=True )
        self.layer.data = data # update image data
        self.__init__( self.layer, **meta ) # update metadata

class Mask( HippoData ):
    """
    Hippo data class for image masks or labels. 
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)

    @classmethod
    def construct( cls, target_layer, viewer=None, args_only=False, **kwds ):
        """
        Factory function that takes a napari image layer (target_layer) and creates a new mask object
        for it. If the layer has a path attribute, and the directory specified contains
        either a "mask.hdr" or a "[filename]_mask.hdr" file then this will be loaded.
        """
        if viewer is None:
            viewer = napari.current_viewer()
        
        # get target layer as class
        base = getLayer( target_layer )
        pth = target_layer.metadata.get('path', '')
        if isinstance(pth, str) or isinstance(pth, Path):
            pth = [pth] # iterable
       
        # load or create masks
        arr = []
        paths = []
        for i,p in enumerate(pth):
            # get mask filename
            kwds['mode'] = kwds.get('mode','filename') # handle default
            if kwds['mode'] == 'filename':
                path = os.path.splitext(p)[0] + '_mask.hdr'
            elif kwds['mode'] == 'directory':
                path = str(Path(os.path.dirname(p)) / 'mask.hdr')
            else:
                assert False, "Error - %s is an invalid mask storage mode"%kwds['mode']
            
            if os.path.exists( path ):
                if isinstance( base, Stack):
                    # deal with potentially different shapes
                    _arr = np.zeros((base.rows, base.cols), dtype=int )
                    msk = (hylite.io.load(path).data[:,:,0].T).astype(int)
                    _arr[:base.stack_rows[i], :base.stack_cols[i]] = msk
                else:
                    _arr = (hylite.io.load(path).data[:,:,0].T).astype(int)
            else:
                _arr = np.zeros((base.rows, base.cols), dtype=int )
            
            arr.append(_arr)
            paths.append(path)
        
        # setup metadata etc.
        add_kwargs = dict( name='[Mask] %s'%(base.getName()),
                               metadata=kwds )
        add_kwargs['metadata']['base'] = base.layer

        if isinstance(base, HSICube): # single-image mask
            add_kwargs['metadata']['path'] = paths[0] # will only be one path here
            args = (arr[0], add_kwargs, 'labels')
        elif isinstance(base, Stack): # mask stack
            add_kwargs['metadata']['path'] = paths # store list of paths
            args = (np.array(arr), add_kwargs, 'labels')
        
        if args_only:
            return args
        
        # add layer and return relevant HSIData instance
        layer = viewer.add_labels( args[0], **args[1] )
        return cls(layer)
        
    def apply(self, crop=False):
        """
        Apply this mask by flagging masked pixels in the target image as np.nan.
        """
        base = getLayer(self.base)
        image = base.toHyImage()
        mask = self.toHyImage()
        if isinstance(base, Stack):
            for i,m in enumerate(mask):
                image[i].data = image[i].data.astype(np.float32)
                image[i].mask( m.data[...,0]==0, crop=crop )
        else:
            image = base.toHyImage()
            image.data = image.data.astype( np.float32 )
            image.mask( mask.data[...,0]==0, crop=crop )
            if crop:
                # we also need to crop the mask!!
                mask.mask( mask.data[...,0]==0, flag=0, crop=crop )
                self.fromHyImage( mask )
                
        # update layer
        base.fromHyImage( image )

    def save(self, force_dirmode=False):
        """
        Save this mask to the corresponding mask file(s).
        """
        mask = self.toHyImage()
        path = self.path
        if not isinstance(path, list):
            path = [path]
            mask = [mask]
        
        for p,m in zip(path, mask):
            m.data = m.data.astype(np.uint16)
            if force_dirmode: # override path to save in directory mode
                p = str(Path(os.path.dirname(path)) / "mask.hdr")
            hylite.io.save(p, m)

    def ndim(self):
        if isinstance( getLayer(self.base), Stack):
            return 4
        else:
            return 2
    
    def toHyImage(self):
        if self.ndim() == 2: # image mask
            #image = hylite.HyImage( n2h( self.layer.data != 0 ) )
            image = hylite.HyImage( n2h( self.layer.data ) )
        else: # Stack mask
            image = []
            base = getLayer( self.base )
            for i in range(self.layer.data.shape[0]):
                # ensure mask shape matches underlying image shape in stack
                #mask = self.layer.data[i, :base.stack_rows[i], 
                #                       :base.stack_cols[i]] != 0
                mask = self.layer.data[i, :base.stack_rows[i], 
                                       :base.stack_cols[i]]
                image.append( hylite.HyImage( n2h( mask )  ) )
        return image
    
    def fromHyImage(self, image):
        if isinstance(image, list):
            pass
        else:
            self.layer.data = h2n( image.data )

class ROI( HippoData ):
    """
    Hippo data class for ROIs. 
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    @classmethod
    def construct( cls, target_layer, mode='polygon', viewer=None, args_only=False, **kwds ):
        """
        Factory function that takes a napari image layer (target_layer) and creates a new ROI object
        for it. If the layer has a path attribute, these ROI points or polygons can be quickly saved 
        to the corresponding image header. The `mode` argument determines if a points or a polygon layer
        is created (i.e. which type of ROI is created).
        """
        if viewer is None:
            viewer = napari.current_viewer()
        
        # get target layer as class
        base = getLayer( target_layer )
        pth = target_layer.metadata.get('path', None)
        if isinstance(pth, str) or isinstance(pth, Path):
            pth = [pth] # iterable for consistency with Stack objects

        # remove existing layer
        if 'poly' in mode.lower():
            name = '[ROI] %s'% base.getName()
            mode = 'poly' # standardise
        elif 'point' in mode.lower():
            name = '[KP] %s'% base.getName()
            mode = 'point'
        else:
            assert False, "%s is an unknown mode. Should be `point` or `poly`"%mode
        if name in viewer.layers:
            viewer.layers.remove(name)

        # load / create ROI points / polygons
        roi_text = []
        roi_vert = []
        for i,p in enumerate(pth):
            # open image header (if it exists)
            if (p is not None):
                p = os.path.splitext(p)[0] + ".hdr"
                if os.path.exists(p):
                    # load geometry from header
                    h = hylite.io.loadHeader(p)
                    
                    for k,v in h.items():
                        k = k.strip()
                        if (k.lower().startswith( 'point' )) or (k.lower().startswith( 'roi' )):
                            # load data from header
                            text = k.split(' ')[-1]
                            verts = h.get_list(k)[::-1].reshape((-1,2)) # reverse order to convert to row, col indexing
                            if ('point' in mode.lower()) and (k.lower().startswith( 'point' )):
                                # these should be treated as points
                                roi_text += [text for p in verts]
                                if isinstance(base, Stack): # batch mode
                                    roi_vert += [ (i, p[0], p[1]) for p in verts ] # concatenate 3D points
                                else: # image mode
                                    roi_vert += [ (p[0], p[1]) for p in verts  ] # concatenate 2D points
                            elif ('poly' in mode.lower()) and (k.lower().startswith( 'roi' )):
                                roi_text += [text]
                                if isinstance(base, Stack): # batch mode
                                    roi_vert.append( [(i, p[0], p[1]) for p in verts] ) # add new 3D polygon
                                else: # image mode
                                    roi_vert.append( [(p[0], p[1]) for p in verts] ) # add new 2D polygon
        
        # setup add kwargs and metadata
        add_kwargs = dict( name=name, text=roi_text,
                               metadata=kwds )
        add_kwargs['metadata']['base'] = base.layer
        add_kwargs['metadata']['mode'] = mode
        if isinstance(base, Stack): # batch mode
            add_kwargs['metadata']['path'] = pth # store list of paths
            add_kwargs['ndim'] = 3 # 3D points (because we're a stack)
        else:
            add_kwargs['metadata']['path'] = pth[0] # will only be one path here
            add_kwargs['ndim'] = 2 # 2D points (just one image)
        if ('poly' in mode.lower()):
            add_kwargs['shape_type']='polygon'
        if args_only:
            return add_kwargs
        
        # construct empty geometry if needed
        if len( roi_text ) > 0:
            pass
        if ('point' in mode.lower()):
            layer = viewer.add_points(roi_vert, **add_kwargs)
        elif ('poly' in mode.lower()):
            layer = viewer.add_shapes(roi_vert, **add_kwargs)
        layer.mode = 'SELECT'
        return cls(layer)
    
    def toList(self, world=True, transpose=False):
        """
        Get the keypoints or ROIs stored in this layer as a list of numpy vertex arrays,
        and associated text (label) array. If world is True, any affine associated
        with this layer will be applied (i.e. coordinates will be in viewer coords). If world
        is a napari layer (with a world_to_data function), coordinates will be transformed into
        that layer's data coordinates. If transpose is True, coordinates will be transformed from 
        napari's (band, row, column) indexing to (x,y) coordinates compatible with HyImages.
        """
        verts = [np.array(verts) for verts in self.layer.data]
        text = list(self.layer.text.values)
        if len(verts) == 0: # edge case...
            return [],[]
        
        # for point layers, shape like a polygon for simplicity during transforms
        if 'point' in self.mode:
            verts = [np.array(verts)] # wrap in a list for compatability with polygons
            if len(verts[0]) == 0: # edge case...
                return [], []
        
        if world:
            verts = [np.array([self.layer.data_to_world(v) for v in p]) for p in verts]
            
            # a layer has been passed; transform into that layer's data coordinates
            if hasattr(world, 'world_to_data'):
                verts = [np.array([world.world_to_data(v) for v in p])
                            for p in verts ]
                    
        # convert from row, col to x,y index as this
        # is the format used for data in hylite
        if transpose:
            if len(verts[0][0]) == 3:
                verts = [p[:,[0,2,1]] for p in verts]
            else:
                verts = [p[:,[1,0]] for p in verts]

        if 'point' in self.mode:
            return list(verts[0]), text # remove "fake" polygon shape
        else:
            return verts, text
    
    def fromList(self, verts, text=None, world=True, transpose=False):
        """
        Set the keypoints or ROIs stored in this layer given a list of numpy vertex arrays
        and correspondign text labels. Text can be None if needed. If world is True, the
        inverse of any affine associated with this layer will be applied 
        (i.e. coordinates will be converted from viewer coords to data coords). If a layer is
        passed, then it is assumed the vertices are in that layer's data coordinates. If transpose 
        is True, coordinates will be transformed from hylite's (x,y) indexing to napari (row, col)
        indexing.
        """
        # check verts is appropriately shaped
        if 'point' in self.mode:
            verts = [verts] # needs to be wrapped into a list for compatability with polygons
        
        # handle transpose
        if transpose:
            if len(verts[0][0]) == 3:
                verts = [np.array(p)[:,[0,2,1]] for p in verts]
            else:
                verts = [np.array(p)[:,[1,0]] for p in verts]
        
        # handle coordinate transforms
        if world:
            # a layer has been passed; transform from that layer's data coordinates
            if hasattr(world, 'world_to_data'):
                verts = [np.array([world.data_to_world(v) for v in p])
                            for p in verts ]
            
            # transform from world coordinates to this ROIs data coordinates
            verts = [np.array([self.layer.world_to_data(v) for v in p])
                        for p in verts ]
        
        # set data
        if 'point' in self.mode:
            self.layer.data = np.vstack(verts)
        else:
            self.layer.data = verts

        # set text
        if text is not None:
            assert len(self.layer.data) == len(text), "Error - text and values must have matching length."
            self.layer.text.values = [t for t in text]
        else:
            self.layer.text.values = ['' for v in verts]
        
        # update
        self.layer.refresh()

    def getTypedName(self):
        """
        Return the typed name for this layer by prefixing the current layer name
        with [KP] (for keypoint annotations) or [ROI] for polygon annotations.
        """
        name = self.getName()
        if 'poly' in self.mode.lower():
            typestr = '[ROI]'
        elif 'point' in self.mode.lower():
            typestr = '[KP]'
        else:
            assert False, "%s is an unknown mode. Should be `polygon` or `points`"
        return f"{typestr} {name.strip()}"
    
class Scene(HippoData):
    """
    Hippo data class for HyScene instances.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    def ndim(self):
        return 3
    @classmethod
    def construct( cls, image, name, viewer=None ):
        """
        Factory function that takes a HyScene instance, creates a new napari 
        layer for it, and returns a corresponding Scene object.
        """
        pass

class View(HippoData):
    """
    Hippo data class for views of 3D point clouds.
    """
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    def ndim(self):
        return 3
    @classmethod
    def construct( cls, cloud, name, camera, args_only=False, viewer=None, **kwds ):
        """
        Factory function that takes a HyCloud instance, creates a new napari 
        layer for each listed camera (view), and returns a corresponding list of Cloud objects.
        """
        if camera is not None:
            image = cloud.render(camera, 'rgb', s=1, fill_holes=True)
        else:
            camera = 'ortho'
            image = cloud.render('ortho', 'rgb', s=1, fill_holes=True )
        image.set_as_nan(0) # remove background

        # basically an image now!
        add_kwargs = HSICube.construct( image, name=name, args_only=True, viewer=viewer, **kwds)
        add_kwargs[1]['metadata']['type'] = 'View'
        add_kwargs[1]['name'] = '[%s] %s'%(add_kwargs[1]['metadata']['type'], name)
        add_kwargs[1]['metadata']['pointsize'] = 1
        add_kwargs[1]['metadata']['camera'] = camera

        if args_only:
            return add_kwargs

        # add layer and return relevant HSIData instance
        layer = viewer.add_image( add_kwargs[0], **add_kwargs[1] )
        return View( layer )
        
    def setPointSize( self, n : int = 1):
        """
        Re-render this view with a different point size.
        """
        self.pointsize = n
        cloud = hylite.io.load(self.path)
        image = cloud.render(self.camera, 'rgb', s=n, fill_holes=True)
        image.set_as_nan(0) # remove background
        self.layer.data = h2n( image.data )

    def toDataImage( self ):
        """
        Return a HyImage containing the rendered data array of the source
        point cloud.
        """
        cloud = hylite.io.load(self.path)
        cloud.decompress()
        image = cloud.render(self.camera, (0,-1), s=self.pointsize, fill_holes=True)
        image.data = image.data.astype(np.float32)
        image.set_as_nan(0)
        return image
    
    def toIDImage( self ):
        """
        Return a HyImage containing the rendered data array of the source
        point cloud.
        """
        cloud = hylite.io.load(self.path)
        image = cloud.render(self.camera, 'i', s=self.pointsize, fill_holes=True)
        image.data = image.data.astype(int)
        return image

def n2h( a ):
    """
    Convert a HSI image array in napari (b,y,x) or (y,x,3) or (y,x,4) format to hylite (x,y,b) format.
    """
    if len(a.shape)==2: # BW image
        return a.T
    if (a.shape[-1] == 3) or (a.shape[-1] == 4):
        return np.transpose(a, (1,0,2) ) # bands stay in the last axis (RGB or RGBA)
    else:
        return np.transpose(a, (2,1,0) ) # bands become the first axis (Multiband)

def h2n( a ):
    """
    Convert an array in hylite format (x,y,b) to napari format (b,x,y).
    """
    if a.shape[-1] == 1:
        return a[:,:,0].T # BW image -- 2D array
    if (a.shape[-1] == 3) or (a.shape[-1] == 4):
        return np.transpose(a, (1,0,2)) # bands stay in the last axis (RGB or RGBA)
    else:
        return np.transpose(a, (2, 1, 0) ) # bands become the first axis (Multiband)

# TODO - delete this function once it is properly redundant
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
                elif l.rgb:
                    if bands is None:
                        return hylite.HyImage( n2h( l.data )), l
                    else:
                        return hylite.HyImage( n2h( l.data )).export_bands(bands), l

    # if we got here, there are not appropriate images in the selection
    napari.utils.notifications.show_warning("Please select an HSI image.")
    return None, None