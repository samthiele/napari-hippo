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
from napari_hippo import getLayer, getByType, HSICube, RGB, RGBA, BW, Stack, ROI
import hylite
from hylite import io
from pathlib import Path
import os
class AnnotToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)
        self.load_widget = magicgui(loadAnnot, call_button='Load')
        self.transpose_widget = magicgui(transpose, call_button='Transpose')
        self.update_widget = magicgui(setLabel, call_button='Update')
        self.save_widget = magicgui(saveAnnot, call_button='Save',
                                    format={"choices": ['header', 'numpy', 'csv']}
                                    )
        self.export_widget = magicgui(export, call_button='Export Patches',
                                             filename={"mode": "w", "filter":"*.hyc"})

        function_widgets = [self.load_widget,
                            self.transpose_widget,
                            self.update_widget,
                            self.save_widget,
                            self.export_widget]
        function_labels = [
            "Annotate",
            "",
            "",
            "",
            ""
        ]

        tutorial_text = (
            "<b>Step 1:</b> TODO<br>"
            "Add more instructions here as needed.<br>"
            "You can extend this tutorial and it will remain scrollable.<br>"
            "Example:<br>"
            "<b>Step 1:</b> TODO<br>"
            "<b>Step 2:</b> TODO<br>"
            "<b>Step 3:</b> TODO<br>"
            "<b>Step 4:</b> TODO<br>"
            "<b>Step 5:</b> TODO<br>"
            "<b>Step 6:</b> TODO<br>"
            "<b>Step 7:</b> TODO<br>"
            "<b>Step 8:</b> TODO<br>"
            "<b>Step 9:</b> TODO<br>"
            "<b>Step 10:</b> TODO<br>"
            "<b>Step 11:</b> TODO<br>"
            "<b>Step 12:</b> TODO<br>"
            "<b>Step 13:</b> TODO<br>"
            "<b>Step 14:</b> TODO<br>"
            "<b>Step 15:</b> TODO<br>"
            "<b>Step 16:</b> TODO<br>"
            "<b>Step 17:</b> TODO<br>"
            "<b>Step 18:</b> TODO<br>"
            "<b>Step 19:</b> TODO<br>"
            "<b>Step 20:</b> TODO<br>"
        )

        self.add_scrollable_sections(function_widgets, tutorial_text, function_labels, stretch=(1,1))

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

def transpose():
    """
    Transpose the seleted annotation layer.
    """
    viewer = napari.current_viewer()  # get viewer
    for l in viewer.layers.selection:
        if l.metadata.get('type','') == 'ROI':
            if 'poly' in l.metadata.get('mode',''):
                l.data = [v[:,[1,0]] for v in l.data]
            elif 'point' in l.metadata.get('mode',''):
                l.data = l.data[:,[1,0]]

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
            points, text = roi.toList(world=base.layer, transpose=True)
            
            for i,_pth in enumerate(pth):
                data={}
                if len(points) > 0:
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
                        _points = []
                        if np.array(points)[0].shape[-1] == 3:
                            # Filter points to relevant slice in Stack (3D) mode
                            _points = [p[...,1:] for p in np.array(points) if p[...,0] == i] 
                        else:
                            _points = points # Image (2D) mode - nothing needs to change
                        if len(_points) == 0:
                            continue # also nothing to do here!
                        for j,p in enumerate(_points): # store points
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
                        if 'point' in roi.mode:
                            header['point ' + k.strip()] = np.vstack(v).reshape(-1)
                        else:
                            header['roi ' + k.strip()] = np.vstack(v).reshape(-1)
                    hylite.io.saveHeader( path, header )
                elif 'csv' in format.lower():
                    if len(data) > 0:
                        path = os.path.splitext(_pth)[0] + '.%s.txt'%roi.mode
                        txt = ''
                        for k,v in data.items():
                            txt+='%s,'%k+','.join(np.vstack(v).reshape(-1).astype(str))+'\n'
                        with open(path,'w') as f:
                            f.write(txt)
                elif 'numpy' in format.lower():
                    if len(data) > 0:
                        path = os.path.splitext(_pth)[0]+ '.%s'%roi.mode
                        np.savez(path, **data)

def export( filename : pathlib.Path = pathlib.Path('.'),
            plot = True ):
    """
    Export each ROI to separate images, stored in a HyCollection directory structure. Spectral
    libraries for any point data (and averages of each ROI) will also be created. If `plot` is True 
    (default) then figures will be created showing these libraries. Note that all selected ROIs will
    be applied to all selected images.
    """
    if filename == '.':
        napari.utils.notifications.show_warning("Please select a path to save output patches.")
        return
    
    # get ROI layers
    viewer = napari.current_viewer()  # get viewer
    layers = viewer.layers.selection
    if len(layers) == 0:
        layers = viewer.layers
    rois = getByType( layers, ROI)
    if len(rois) == 0:
        napari.utils.notifications.show_warning("No ROI or KP layers found in selection.")
        return
    
    # get image layers
    images = getByType( layers, [HSICube, RGB, RGBA, BW, Stack] )
    if len(images) == 0:
        napari.utils.notifications.show_warning("No image layers found in selection.")
        return
    
    # create output HyCollection
    O = hylite.HyCollection(os.path.basename(filename), os.path.dirname(filename))

    for _image in images: # loop through image layers 
        # get HyImage data from each layer to export
        # (noting that Stacks will contain data from multiple images)
        if isinstance(_image, Stack):
            slices = _image.toHyImage() # get RGB previews (slices)
            ilist = []
            iname = []
            for p,s in zip(_image.path, slices):
                p = os.path.splitext(p)[0] + '.hdr'
                iname.append( os.path.splitext( os.path.basename(p))[0] )
                if os.path.exists(p):
                    ilist.append( io.load(p) ) # load HSI
                else:
                    ilist.append(s) # else, just export RGB slice
        else:
            ilist = [_image.toHyImage()] # easy!
            iname = [os.path.splitext( os.path.basename(_image.path))[0]]

        library = {} # spectral library will be built in here
        wav = {}
        for i,(img,name) in enumerate(zip(ilist,iname)): # loop through actual images
            if name not in library:
                library[name] = {}
                wav[name] = img.get_wavelengths()
            for _roi in rois: # loop through ROI layers
                # get points/polygons in image data coordinates
                verts, names = _roi.toList(world=_image.layer, transpose=True)
                if len(verts) == 0:
                    continue # nothing to do here
                if 'point' in _roi.mode:
                    verts = [verts] # give same shape as polygons for simplicity
        
                # loop through each polygon (or point list)
                for v, n in zip(verts, names):
                    if (len(v[0]) == 3) and (i != v[0][0]):
                        continue # wrong slice
                    if (len(v[0]) == 3):
                        v = np.vstack(v)[:,1:] # drop slice index
                    else:
                        v = np.vstack(v) # no slice index here
                    
                    if 'poly' in _roi.mode.lower():
                        # extract ROI
                        roi = img.copy()
                        f=0 # background flag
                        if roi.data.dtype == 'f':
                            f=np.nan # use nan for float data
                        roi.mask( v, crop=True, flag=f )

                        # average for spectral library
                        X = roi.X(onlyFinite=True)
                        X = X[ (X!=f).all(axis=-1), : ]
                        avg = np.mean( X, axis=0 )
                        if n not in library[name]:
                            library[name][n] = []
                        library[name][n] += [avg]

                        # store it
                        n = str(n).replace(' ','_').replace('-','_').replace('.','')
                        O.addSub(n).set( name, roi )
                        O.save()
                        O.free()
                    elif 'point' in _roi.mode.lower():
                        # export points to spectral library too
                        if n not in library[name]:
                            library[name][n] = []
                        for x,y in v:
                            if (x>=0) and (x<img.xdim()):
                                if (y>=0) and (y<img.ydim()):
                                    library[name][n] += [img.data[int(x),int(y),:]]
    
    # save libraries
    for k,v in library.items():
        names = list(v.keys())
        spectra = np.vstack( v[names[0]] )
        lib = hylite.HyLibrary( spectra[None,:,:], lab=[names[0]], wav=wav[k])
        for n in names[1:]:
            spectra = np.vstack( v[n] )
            lib = lib + hylite.HyLibrary( spectra[None,:,:], lab=[n], wav=wav[k])
        O.set( k, lib)
        O.save()
        O.free()
        if plot:
            fig,ax = lib.quick_plot()
            ax.set_title(k)
            fig.show()
    O.save() # write everything to disk
    O.free()

