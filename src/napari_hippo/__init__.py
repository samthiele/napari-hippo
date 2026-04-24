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

# ToolManager to ensure only one tool widget is open at a time
class ToolManager:
    def __init__(self, viewer):
        self.viewer = viewer
        self.active_tool_widget = None

    def open_tool(self, tool_cls):
        # Remove previous tool widget if it exists
        if self.active_tool_widget is not None:
            try:
                self.viewer.window.remove_dock_widget(self.active_tool_widget)
            except Exception:
                pass  # Already removed or not docked

        # Create and add new tool widget
        new_widget = tool_cls(self.viewer)
        self.viewer.window.add_dock_widget(new_widget, name=tool_cls.__name__, area='right')
        self.active_tool_widget = new_widget

# Example usage:
# tool_manager = ToolManager(viewer)
# tool_manager.open_tool(BasicWidget)   # Opens Basic tool
# tool_manager.open_tool(LibraryWidget) # Closes Basic, opens Library

# setup plugin callbacks
import napari
import os
viewer = napari.current_viewer()  # get viewer
import hylite
hylite.band_select_threshold = 100. # be a bit tolerant here
from napari_hippo._base import getMode, getLayer, Stack

def update_slider(event):
    viewer.text_overlay.text = '' # no overlay
    step = viewer.dims.current_step[0]
    if getMode(napari.current_viewer()) == 'Batch':
        layers = getByType(viewer.layers, Stack)
        if len(layers) > 0:
            if hasattr(layers[0], 'path'):
                if step < len(layers[0].path):
                    p = layers[0].path[step]
                    dname = os.path.basename( os.path.dirname(p))
                    fname = os.path.basename( p )
                    viewer.text_overlay.text = "%s/%s"%(dname,fname)
                    return
    else:
        # check layers in selection, then entire tree
        for layers in [list(viewer.layers.selection), list(viewer.layers)]:
            for l in layers:
                if 'wavelength' in l.metadata:
                    if len(l.metadata['wavelength']) > step:
                        w = l.metadata['wavelength'][step]
                        viewer.text_overlay.text = "%d nm"%(w)
                        return
    
if viewer is not None:
    viewer.text_overlay.visible = True
    viewer.dims.events.current_step.connect(update_slider)
