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
from napari_hippo import getByType, getLayer, View, HSICube, BW, ROI, Scene
import hylite
from hylite import io
from pathlib import Path

from hylite.project.align import align_to_cloud_manual
from hylite.project import Camera
class HypercloudToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.pointsize_widget = magicgui(setPointSize, call_button='Set')
        self._add([self.pointsize_widget], 'Rendering')

        self.extractdata_widget = magicgui(extractData, call_button='Data Array')
        self.pointid_widget = magicgui(extractIDs, call_button='Point IDs')
        self._add([self.extractdata_widget, self.pointid_widget], 'Extract')

        #self.fit_elc_widget = magicgui(ELC, call_button='Define/Apply ELC' )
        #self.apply_elc_widget = magicgui(applyELC, call_button='Apply Previous ELC' )
        #self._add([self.fit_elc_widget, self.apply_elc_widget], 'Quick Calibration')

        self.locate_widget = magicgui(locate, 
                                      projection={'choices':['panoramic', 'perspective']},
                                      refine_method={'choices':['None', 'SIFT', 'ORB']},
                                      ifov=dict(min=0, max=np.inf, step=0.005),
                                      call_button='Locate')
        self._add([self.locate_widget], 'Camera Pose')

        self.scene_widget = magicgui(buildScene, cloud={'mode': 'r', "filter":"*.ply"},
                                      camera={"mode": "r", "filter":"*.txt"},
                                      call_button='Build')
        self._add([self.scene_widget], 'Scene')


        #root={'mode': 'd'}

        # add spacer at the bottom of panel
        self.qvl.addStretch()

def setPointSize( size : int = 2 ):
    viewer = napari.current_viewer()  # get viewer

    # get selected View layers
    layers = viewer.layers.selection
    if len(layers) == 0:
        layers = viewer.layers
    layers = getByType( viewer.layers.selection, View)

    # set size
    for I in layers:
        I.setPointSize(size)

def _extract(ids=False):
    """
    Utility function to allow reuse of code between extract data and extract ID
    functions.
    """
    viewer = napari.current_viewer()  # get viewer
    layers = getByType( viewer.layers.selection, View)
    if len(layers) == 0:
        napari.utils.notifications.show_warning("Please select a View layer to export.")
    
    # extract data and add to viewer
    for I in layers:
        name = I.getName()
        if ids:
            image = I.toIDImage()
            name += '(ID)'
            return BW.construct( image, name, viewer )
        else:
            image = I.toDataImage()
            return HSICube.construct( image, name, viewer )

def extractData():
    return _extract(False)

def extractIDs():
    return _extract(True)

def locate( view : 'napari.layers.Image', 
            keypoints : 'napari.layers.Points',
            projection : str = 'panoramic',
            ifov : float = 0.039,
            refine_method = 'None' ):
    """
    Read keypoints from the selected image header file (entries with the format `point [pointID]: { xcoord, ycoord }`)
    and, using the 3D coordinates retrieved from the specified point cloud, solve the 3D location of the camera
    using openCV's implementation of the PnP problem.

    Args:
        view: A View that defines the cloud to which the keypoints belong.
        keypoints: A KP layer containing the point keypoints with integer (point ID) names.
        refine_method: True if SIFT matching should be performed to refine the initial PnP solution (by including more
                    keypoints).
    """
    viewer = napari.current_viewer()  # get viewer   

    # get keypoints layer 
    kps = getLayer( keypoints, viewer = viewer)
    if not isinstance(kps, ROI):
        napari.utils.notifications.show_warning("Keypoints must be a [KP] layer")
        return
    points, names = kps.toList( world=True, transpose=True )

    # load cloud layer
    view = getLayer( view, viewer = viewer)
    if not isinstance(view, View):
        napari.utils.notifications.show_warning("Cloud must be a [Scene] layer.")
        return
    if not os.path.exists(view.path):
        napari.utils.notifications.show_warning("Cloud not found: %s"%view.path)
        return
    cloud = io.load(view.path)

    # find keypoints with integer names
    kps_3d = []
    kps_2d = []
    for p,n in zip(points, names):
        try:
            pid = int(n) # try converting to an integer
        except:
            continue # try next one
        if (pid > 0) and (pid < cloud.point_count()):
            kps_3d.append( pid )
            if len(p) == 3:
                kps_2d.append( (p[1], p[2] ) )
            else:
                kps_2d.append( (p[0], p[1] ) )
    
    # create Camera object
    xdim = 500  # how to get this info?
    ydim = 401
    cam = Camera( np.zeros(3), 
                 np.zeros(3), 
                 proj=projection, fov=ydim*ifov, step=ifov, dims=(xdim,ydim) )
    
    # solve pnp
    cam, residual = align_to_cloud_manual( cloud, cam, kps_3d, kps_2d )

    # add output
    result = View.construct( cloud, view.getName()+"(est)", cam,viewer=viewer, 
                                residual=residual, path=view.path )
    return result.layer

# def qaqc( saturation : bool = True, noise : bool = True ):
#     viewer = napari.current_viewer()  # get viewer
#     image, layer = getHyImage(viewer)
#     image.decompress()
#     if image is not None:
#         if noise:
#             noise = np.nanmean(np.abs(np.diff(image.data, axis=-1)) / image.data[..., 1:], axis=-1)
#             oversat = np.isinf(noise)  # oversaturated pixels result in div 0
#             noise[oversat] = np.nan  # replace these with nans
#             vmn,vmx = np.nanpercentile(noise, (2,98))
#             viewer.add_image( noise.T, name=layer.name + ' [noise]', colormap='coolwarm', contrast_limits=(vmn,vmx))
#         if saturation:
#             sat = np.max(image.data, axis=-1).astype(np.float32)
#             vmn, vmx = np.nanpercentile(sat, (2, 98))
#             viewer.add_image(sat.T, name=layer.name + ' [maximum saturation]', colormap='coolwarm', contrast_limits=(vmn, vmx))
#     else:
#         napari.utils.notifications.show_info("Please select a hyperspectral image image to compute QAQC.")

# def ELC( ):
#     """
#     Select a region in an image and compute a rough ELC correction from it.
#     (by assuming that the region is a white panel)
#     """
#     viewer = napari.current_viewer()  # get viewer
#     if 'panel' not in viewer.layers:
#         panel_layer = viewer.add_shapes(None, ndim=2, name='panel', shape_type='polygon')
#         panel_layer.mode = 'add_polygon'
#         napari.utils.notifications.show_info("Please use the 'panel' layer to select the panel.")
#     else:
#         panel_layer = viewer.layers['panel']
#         image, layer = getHyImage(viewer)
#         if image is not None:
#             # get mask
#             image.decompress()
#             mask = panel_layer.to_masks((image.ydim(), image.xdim()))[0]

#             # compute average spectra
#             ref = np.nanmedian( image.data[ mask.T, : ], axis=0 )

#             # store it in labels layer showing panel pixels
#             viewer.add_labels(mask, name='Panel Pixels', 
#                     metadata=dict(spectra=ref))
            
#             # correct image
#             image.data = image.data / ref[None,None,:].astype(np.float32)

#             # add it
#             viewer.add_image(h2n(image.data), name=layer.name + ' [elc]', contrast_limits=(0.0,0.75), metadata=dict(
#                 type='HSIf',  # HSI full
#                 path=layer.metadata['path'],
#                 wav=image.get_wavelengths()))
#         else:
#             napari.utils.notifications.show_info("Please select a hyperspectral image image to correct.")

# def applyELC():
#     """
#     Apply the previously computed ELC (see ELC()) to a different image.
#     """
#     viewer = napari.current_viewer()  # get viewer
#     image, layer = getHyImage(viewer) # get selected image
#     if 'Panel Pixels' in viewer.layers:
#         ref = viewer.layers['Panel Pixels'].metadata.get('spectra',None)
#         if ref is not None:
#             # correct image
#             image.decompress()
#             image.data = image.data / ref[None,None,:].astype(np.float32)

#             # add it
#             viewer.add_image(h2n(image.data), name=layer.name + ' [elc]', contrast_limits=(0.0,0.75), metadata=dict(
#                 type='HSIf',  # HSI full
#                 path=layer.metadata['path'],
#                 wav=image.get_wavelengths()))
#         else:
#             napari.utils.notifications.show_error("'Panel Pixels' layer contains no ELC?") 
#     else:
#         napari.utils.notifications.show_info("Please compute an ELC first. This will be stored in the 'Panel Pixels' layer.") 

def panel( panel_ROI : str = 'white',
           panel_corners : str = 'corners',
           shaded : bool = False,
           spectra : Path = Path('') ):
    pass



def buildScene( 
    cloud : Path = Path(''), 
    camera : Path = Path('camera.txt') ):

    pass

