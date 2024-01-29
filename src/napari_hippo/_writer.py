"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]

import napari
import hylite
from napari_hippo import getHyImage

def write_single_image(path: str, data: List[FullLayerData] ) -> List[str]:
    """Writes a single image layer"""
    viewer = napari.current_viewer()
    out = []
    for d, attr, t in data:
        name = attr['name']
        if name in viewer.layers:
            image, layer = getHyImage( viewer, viewer.layers[name] )
            if image is not None:
                hylite.io.save(path, image)
                out.append(path)

    if len(out) == 0:
        napari.utils.notifications.show_warning("Error - could not save %s" % name)
    return out

def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
    """Writes multiple layers of different types."""

    pass

    # implement your writer logic here ...
    """viewer = napari.current_viewer()
    path = os.path.splitext(path)[0]
    root = os.path.dirname(path)
    name = os.path.basename(path)
    C = hylite.HyCollection(name, root)
    for dtype, attr, name in data:
        if name in viewer.layers:
            # HSI image data
            image, layer = getHyImage( viewer, viewer.layers[name] )
            if image is not None:
                C.set(name, image)

            # todo - also store point and polygon data?

    C.save()
    # return path to any file(s) that were successfully written
    napari.utils.notifications.show_info("Saved collection to %s" % C.getDirectory())
    return [path]"""
