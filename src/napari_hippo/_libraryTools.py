from ._guiBase import GUIBase
from magicgui import magicgui

import pathlib
import numpy as np
from natsort import natsorted
import glob
import napari
from napari_hippo import napari_get_hylite_reader
from napari_hippo._base import *
import os
import hylite
from hylite import io

class LibraryWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.test_widget = magicgui(construct, input={'mode': 'd'}, output={'mode': 'd'}, fingerprints={"filter": "*.json"}, call_button='Build', auto_call=False)
        self._add([self.test_widget], 'Construct Library')

def construct( input : pathlib.Path = pathlib.Path(''), 
               output : pathlib.Path = pathlib.Path(''),
               fingerprints : pathlib.Path = pathlib.Path('') ):
    print( input, output, fingerprints )


