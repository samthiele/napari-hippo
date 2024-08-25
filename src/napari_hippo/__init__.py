__version__ = "0.2.0"

from ._base import *
from ._reader import napari_get_hylite_reader, napari_get_specim_reader
from ._sample_data import make_sample_image
from ._basicTools import BasicWidget
from ._coregTools import CoregToolsWidget
from ._hyliteTools import HyliteToolsWidget
from ._hypercloudTools import HypercloudToolsWidget
from ._caterpillarWidget import CaterpillarWidget
from ._annotationTools import AnnotToolsWidget
from ._writer import write_multiple, write_single_image

__all__ = (
    "napari_get_hylite_reader",
    "napari_get_specim_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_image",
    "make_sample_cloud",
    "BasicWidget",
    "CoregToolsWidget",
    "HyliteToolsWidget",
    "HypercloudToolsWidget",
    "CaterpillarWidget",
    "AnnotToolsWidget",
)

# setup plugin callbacks
import napari
import os
viewer = napari.current_viewer()  # get viewer
import hylite
hylite.band_select_threshold = 100. # be a bit tolerant here

def update_slider(event):
    viewer.text_overlay.text = '' # no overlay
    if getMode(napari.current_viewer()) == 'Batch':
        viewer.text_overlay.text = '[Batch Mode]' # flag that napari is in batch mode

    step = viewer.dims.current_step[0]
    # check layers in selection, then entire tree
    for layers in [list(viewer.layers.selection), list(viewer.layers)]:
        for l in layers:
            if 'STACK' in l.metadata.get('type', '') and ('path' in l.metadata):
                if step < len(l.metadata['path']):
                    path = l.metadata['path'][step]
                    viewer.text_overlay.text = os.path.basename( os.path.dirname(path) ) + '/' + os.path.basename(path)
                    return
            elif ('HSIf' in l.metadata.get('type', '')) and ('wav' in l.metadata):
                if step < len(l.metadata['wav']):
                    viewer.text_overlay.text = "wavelength: %.1f" % l.metadata['wav'][step]
                    return
    
if viewer is not None:
    viewer.text_overlay.visible = True
    viewer.dims.events.current_step.connect(update_slider)
