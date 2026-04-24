# load custom pytest fixtures
from .fixtures import *
import numpy as np
import hylite
from napari_hippo._libraryTools import construct

def test_construct():
    inputs = ''
    outputs = ''
    fingerprints = ''

    construct( inputs, outputs, fingerprints )
