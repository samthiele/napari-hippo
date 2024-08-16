"""

Some crunchy tools for data munging.

"""

from typing import TYPE_CHECKING
from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget, QFrame, QGroupBox
import pathlib
if TYPE_CHECKING:
    import napari

import numpy as np
from magicgui import magicgui
import napari
from napari.experimental import link_layers
from ._guiBase import GUIBase
from napari_hippo import getLayer, HSICube, Stack, ROI
import re
import hylite
from pathlib import Path
import os
class AnnotToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.load_widget = magicgui(loadAnnot, call_button='Load')
        self.update_widget = magicgui(setLabel, call_button='Update')
        self.save_widget = magicgui(saveAnnot, call_button='Save',
                                    format={"choices": ['header', 'numpy', 'csv']}
                                    )
        self._add( [self.load_widget, 
                    self.update_widget,
                    self.save_widget
                    ], 'Annotate' )

        # add spacer at the bottom of panel
        self.qvl.addStretch()

def loadAnnot( ):
    viewer = napari.current_viewer()  # get viewer

    # get layers
    layers = viewer.layers.selection
    if len(layers) == 0:
        layers = viewer.layers

    # add ROIs and KPs layers
    roi = []
    points = []
    for l in layers:
        base = getLayer( l ) 
        if isinstance(base, HSICube) or isinstance(base, Stack): # check data type
            roi.append( ROI.construct( l, mode='polygon', viewer=viewer ) )
            points.append( ROI.construct( l, mode='point', viewer=viewer ) )
            
            # link layer visibility
            link_layers( [base.layer, roi[-1].layer, points[-1].layer], ('visible',) )
    return roi, points

def setLabel( label : str = 'cool thing'):
    """
    Loop through shapes in annotation layers and replace any undefined 
    text labels with the specified one.
    """
    viewer = napari.current_viewer()  # get viewer
    
    # update text
    for l in viewer.layers:
        if l.metadata.get('type','') == 'ROI':
            text = [str(t) for t in l.text.values]
            for i in range(len(text)):
                if text[i] == '':
                    text[i] = label
            l.text.values = text
            l.refresh()
    
def saveAnnot( format : str = 'header'):
    """
    Save annotation layers to header file or numpy array. The `format` argument determines
    if annotations are saved to the base image header file or adjacent .npy or .txt files.
    """
    viewer = napari.current_viewer()  # get viewer
    
    # update text
    for l in viewer.layers:
        if l.metadata.get('type','') == 'ROI':
            roi = ROI(l)

            # get base path
            base = getLayer(roi.base)
            pth = base.path
            if isinstance(pth, str) or isinstance(pth, Path):
                pth = [pth]
            
            # get points
            points, text = roi.toList()
            if len(points) == 0:
                return # nothing to do here
            
            for i,_pth in enumerate(pth):
                data={}
                if 'poly' in roi.mode: # extract polygons
                    if points[0].shape[-1] == 3:
                        # Stack (3D) mode; check if there are any actual annotations for this image!
                        for j,p in enumerate(points):
                            if p[0,0] == i: # this is in the correct part of the stack
                                data[text[j]] = p[:,1:]
                    else:
                        for j,p in enumerate(points):
                            data[text[j]] = p
                else: # extract points
                    if points[0].shape[-1] == 3:
                        points = [p[...,1:] for p in points if p[...,0] == i] # Stack (3D) mode
                    else:
                        pass # Image (2D) mode - nothing fancy to do here
                    if len(points) == 0:
                        continue # also nothing to do here!
                    for j,p in enumerate(points):
                        if text[j] in data:
                            data[text[j]].append(p)
                        else:
                            data[text[j]] = [p]

                # write file
                if 'header' in format.lower():
                    path = os.path.splitext(_pth)[0] + '.hdr'
                    if os.path.exists(path):
                        header = hylite.io.loadHeader(path)
                    else:
                        header= hylite.HyHeader()

                    # clean header
                    todel = []
                    for k,v in header.items():
                        if k.startswith('point') and ('point' in roi.mode):
                            todel.append(k)
                        if k.startswith('roi') and ('poly' in roi.mode):
                            todel.append(k)
                    for k in todel:
                        del header[k]
                    
                    for k,v in data.items():
                        # N.B. we reverse the order to cleverly change from row,col to x,y indexing.
                        if 'point' in roi.mode:
                            header['point ' + k.strip()] = np.vstack(v).reshape(-1)[::-1] 
                        else:
                            header['roi ' + k.strip()] = np.vstack(v).reshape(-1)[::-1] 
                    hylite.io.saveHeader( path, header )
                elif 'csv' in format.lower():
                    path = os.path.splitext(_pth)[0] + '.%s.txt'%roi.mode
                    txt = ''
                    for k,v in data.items():
                        txt+='%s,'%k+','.join(np.vstack(v).reshape(-1)[::-1].astype(str))+'\n'
                    with open(path,'w') as f:
                        f.write(txt)
                elif 'numpy' in format.lower():
                    path = os.path.splitext(_pth)[0]+ '.%s'%roi.mode
                    np.savez(path, **data)

