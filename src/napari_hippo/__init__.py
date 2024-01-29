__version__ = "0.1.0"

from ._util import n2h, h2n, getHyImage
from ._reader import napari_get_ENVI_reader, napari_get_specim_reader
from ._sample_data import make_sample_data
from ._ioTools import IOWidget
from ._crunchyTools import CrunchyToolsWidget
from ._hyliteTools import HyliteToolsWidget
from ._fieldTools import FieldToolsWidget
from ._caterpillarWidget import CaterpillarWidget
from ._writer import write_multiple, write_single_image

__all__ = (
    "napari_get_ENVI_reader",
    "napari_get_specim_reader",
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "IOWidget",
    "CrunchyToolsWidget",
    "HyliteToolsWidget",
    "FieldToolsWidget",
    "CaterpillarWidget"
)

# setup plugin callbacks
import napari
import os
viewer = napari.current_viewer()  # get viewer

def update_slider(event):
    viewer.text_overlay.text = '' # no overlay
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
