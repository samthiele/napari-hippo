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
from napari_hippo import getHyImage, h2n, n2h
import hylite
from pathlib import Path
class FieldToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        #self.qaqc_widget = magicgui(qaqc, call_button='Compute')
        #self._add([self.qaqc_widget], 'QAQC')

        #self.fit_elc_widget = magicgui(ELC, call_button='Define/Apply ELC' )
        #self.apply_elc_widget = magicgui(applyELC, call_button='Apply Previous ELC' )
        #self._add([self.fit_elc_widget, self.apply_elc_widget], 'Quick Calibration')

        self.combine_widget = magicgui(combine, 
                                       method={"choices": ['median (p50)', 'mean', 'brightest', 'darkest', 'p90', 'p75', 'p25', 'p10']},
                                       call_button='Compute' )
        self._add([self.combine_widget], 'Combine')

        self.locate_widget = magicgui(locate, cloud={'mode': 'r', "filter":"*.ply"},
                                      output={"mode": "w", "filter":"*.txt"},
                                      projection={'choices':['panoramic', 'perspective']},
                                      call_button='Locate')
        self._add([self.locate_widget], 'Camera Pose')

        self.scene_widget = magicgui(buildScene, cloud={'mode': 'r', "filter":"*.ply"},
                                      camera={"mode": "r", "filter":"*.txt"},
                                      call_button='Build')
        self._add([self.scene_widget], 'Scene')


        #root={'mode': 'd'}

        # add spacer at the bottom of panel
        self.qvl.addStretch()


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

def combine(method='median (p50)'):
    """
    Average multiple HSI images (with the same dimensions) into one file. 
    Useful for e.g., averaging outcrop scans captured several times to reduce
    noise.
    """

    # get selected images
    viewer = napari.current_viewer()  # get viewer
    images = [getHyImage(viewer, layer=l)[0] for l in viewer.layers.selection]
    
    # compute dims
    xdm = np.array([i.xdim() for i in images])
    ydm = np.array([i.ydim() for i in images])
    if (np.diff(xdm) != 0).any() or (np.diff(ydm) != 0).any():
        napari.utils.notifications.show_info("Warning - selected images differ in size but up to (%d,%d) pixels."%(np.max(np.abs(np.diff(xdm))), np.max(np.abs(np.diff(ydm))))) 
    if (np.diff([i.band_count() for i in images])!=0).any():
        napari.utils.notifications.show_error("Error - selected images have different numbers of bands.")
        return
     
    # average
    if 'median' in method.lower():
        arr = np.nanmedian([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], axis=0)
    elif 'mean' in method.lower():
        arr = np.nanmean([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], axis=0)
    elif 'brightest' in method.lower():
        arr = np.nanmax([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], axis=0)
    elif 'darkest' in method.lower():
        arr = np.nanmin([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], axis=0)
    elif 'p90' in method.lower():
        arr = np.nanpercentile([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], 90, axis=0)
    elif 'p75' in method.lower():
        arr = np.nanpercentile([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], 75, axis=0)
    elif 'p25' in method.lower():
        arr = np.nanpercentile([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], 25, axis=0)
    elif 'p10' in method.lower():
        arr = np.nanpercentile([i.data[:np.min(xdm), :np.min(ydm), :] for i in images], 10, axis=0)

    # put in HyImage object
    image = hylite.HyImage(arr)
    image.set_wavelengths( images[0].get_wavelengths() )

    # add to viewer
    data = image.data.T
    name = '[cube] combined'
    add_kwargs = dict(name=name, metadata=dict(
        type='HSIf',
        path=None,
        wav=image.get_wavelengths()))
    viewer.add_image(data, **add_kwargs)

def panel( panel_ROI : str = 'white',
           panel_corners : str = 'corners',
           shaded : bool = False,
           spectra : Path = Path('') ):
    pass

def locate( cloud : Path = Path(''), 
            output : Path = Path('camera.txt'),
            projection : str = 'panoramic',
            ifov : float = 0.039,
            auto_match = True ):
    """
    Read keypoints from the selected image header file (entries with the format `point [pointID]: { xcoord, ycoord }`)
    and, using the 3D coordinates retrieved from the specified point cloud, solve the 3D location of the camera
    using openCV's implementation of the PnP problem.

    Args:
        cloud: A pathlib.Path to the point cloud to which pointIDs correspond.
        output: A pathlib.Path to a text file where the resulting camera pose information should be written.
        auto_match: True if SIFT matching should be performed to refine the initial PnP solution (by including more
                    keypoints).
    """
    pass

def buildScene( 
    cloud : Path = Path(''), 
    camera : Path = Path('camera.txt') ):

    pass

