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
class FieldToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.qaqc_widget = magicgui(qaqc, call_button='Compute')
        self._add([self.qaqc_widget], 'QAQC')

        self.elc_widget = magicgui(ELC, call_button='Quick Correct' )
        self._add([self.elc_widget], 'Calibration')

        # add spacer at the bottom of panel
        self.qvl.addStretch()

def qaqc( saturation : bool = True, noise : bool = True ):
    viewer = napari.current_viewer()  # get viewer
    image, layer = getHyImage(viewer)
    image.decompress()
    if image is not None:
        if noise:
            noise = np.nanmean(np.abs(np.diff(image.data, axis=-1)) / image.data[..., 1:], axis=-1)
            oversat = np.isinf(noise)  # oversaturated pixels result in div 0
            noise[oversat] = np.nan  # replace these with nans
            vmn,vmx = np.nanpercentile(noise, (2,98))
            viewer.add_image( noise.T, name=layer.name + ' [noise]', colormap='coolwarm', contrast_limits=(vmn,vmx))
        if saturation:
            sat = np.max(image.data, axis=-1).astype(np.float32)
            vmn, vmx = np.nanpercentile(sat, (2, 98))
            viewer.add_image(sat.T, name=layer.name + ' [maximum saturation]', colormap='coolwarm', contrast_limits=(vmn, vmx))
    else:
        napari.utils.notifications.show_info("Please select a hyperspectral image image to compute QAQC.")

def ELC( ):
    viewer = napari.current_viewer()  # get viewer
    if 'panel' not in viewer.layers:
        panel_layer = viewer.add_shapes(None, ndim=2, name='panel', shape_type='polygon')
        panel_layer.mode = 'add_polygon'
        napari.utils.notifications.show_info("Please use the 'panel' layer to select the panel.")
    else:
        panel_layer = viewer.layers['panel']
        image, layer = getHyImage(viewer)
        if image is not None:
            # get mask
            image.decompress()
            mask = panel_layer.to_masks((image.ydim(), image.xdim()))[0]
            viewer.add_labels(mask, name='Panel Pixels')

            # compute average spectra
            ref = np.nanmedian( image.data[ mask.T, : ], axis=0 )

            # correct image
            image.data = image.data / ref[None,None,:].astype(np.float32)

            # add it
            viewer.add_image(h2n(image.data), name=layer.name + ' [elc]', contrast_limits=(0.0,0.75), metadata=dict(
                type='HSIf',  # HSI full
                path=layer.metadata['path'],
                wav=image.get_wavelengths()))
        else:
            napari.utils.notifications.show_info("Please select a hyperspectral image image to correct.")

