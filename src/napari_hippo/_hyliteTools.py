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
from ._guiBase import GUIBase
from napari_hippo import getHyImage, h2n, n2h
import re
import hylite

class HyliteToolsWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        #btn = QPushButton("Load with query")
        #btn.clicked.connect(self._query)

        self.falseColor_widget = magicgui(falseColor,
                                          call_button='Update',
                                          stretch={"choices": ['Absolute',
                                                               'Percent clip',
                                                               'Percent clip (per band)']},
                                          auto_call=False )
        self._add( [self.falseColor_widget], 'Psuedocolour Visualisation' )


        self.hullCorrect_widget = magicgui(hullCorrect,
                                          call_button='Compute',
                                          auto_call=False)
        self._add([self.hullCorrect_widget], 'Hull Correction')

        self.dimensionReduction_widget = magicgui(dimensionReduction,
                                           method={"choices": ['PCA', 'MNF']},
                                           ndim={'min': 1, 'max': 100},
                                           call_button='Reduce',
                                           auto_call=False)
        self._add([self.dimensionReduction_widget], 'Dimension Reduction')

        """
        self.bandRatio_widget = magicgui(bandRatio,
                                           call_button='Update',
                                           auto_call=False)
        self._add([self.bandRatio_widget], 'Band Ratio')
        """
        self.calculator_widget = magicgui(calculator,
                                         call_button='Calculate',
                                         auto_call=False)
        self._add([self.calculator_widget], 'Calculator (e.g., Band Ratios)')

        # add spacer at the bottom of panel
        self.qvl.addStretch()

def falseColor( bands : str = "%d, %d, %d" % hylite.RGB,
                vmin : float = 2, vmax : float = 98,
                stretch : str = 'Percent clip (per band)' ):
    viewer = napari.current_viewer()  # get viewer
    image, layer = getHyImage(viewer)
    if image is None:
        napari.utils.notifications.show_warning(
            "Warning: Could not find valid HSI data.")
        return

    sbands = bands.split(",")
    if len(sbands) != 3:
        napari.utils.notifications.show_warning(
            "Please provide 3 bands, separated by commas.")
        return

    # parse bands and get HyImage
    result = []
    for b in sbands:
        result.append( calculate( image, b ).data[..., 0 ] )

    result = hylite.HyImage( np.dstack( result ) )

    # apply normalisation
    if stretch == 'Percent clip (per band)':
        result.percent_clip( int(vmin), int(vmax), per_band=True )
    elif stretch == 'Percent clip':
        result.percent_clip( int(vmin), int(vmax), per_band=False)
    elif stretch == 'Absolute':
        result.data = np.clip( (result.data - vmin) / (vmax-vmin), 0, 1 )

    # add to napari
    # todo - we should copy the affine matrix here?
    # (but have to deal with the different dimensionality of hypercubes vs rgb images)
    meta = dict(path=layer.metadata['path'], type='HSIp', wav=result.get_wavelengths())
    return viewer.add_image( h2n( result.data ), rgb=True,
                       name=layer.name + " [%s]"%(bands),
                       metadata=meta )

def hullCorrect(wmin : float = 2000.,
                wmax : float = 2500.,
                upper : bool = True):
    viewer = napari.current_viewer()  # get viewer
    image, layer  = getHyImage(viewer)
    if image is not None:
        from hylite.correct import get_hull_corrected
        if upper:
            hc = get_hull_corrected(image, band_range=(wmin, wmax), hull='upper')
        else:
            hc = get_hull_corrected(image, band_range=(wmin, wmax), hull='lower')

        # add layer
        return viewer.add_image( h2n(hc.data ), name=layer.name + ' [hc]', metadata=dict(
                type='HSIf', # HSI full
                path=layer.metadata['path'],
                wav=hc.get_wavelengths()) )

def dimensionReduction( method : str = 'PCA', ndim : int = 3, wmin : float = 2000., wmax : float = 2500. ):
    viewer = napari.current_viewer()  # get viewer
    image, layer =  getHyImage(viewer)
    if image is None:
        napari.utils.notifications.show_warning(
            "Warning: Could not find valid HSI data.")
        return

    from hylite.filter import MNF, PCA
    if 'mnf' in method.lower():
        R = MNF
    elif 'pca' in method.lower():
        R = PCA
    else:
        napari.utils.notifications.show_warning(
            "Warning: Unknown dimension reduction method %s."%method)
        return

    brange = ( image.get_band_index(wmin), image.get_band_index(wmax) )
    result, _ = R( image, bands= ndim, band_range=brange )

    print(np.isfinite(result.data).any() )

    meta = dict(path=layer.metadata['path'],
                type='HSIf',
                wav=np.arange(result.band_count()))
    data = h2n(result.data)
    if (result.band_count() == 3) or (result.band_count() == 4):
        data = np.transpose(data, (2, 0, 1 ) ) # bands should be in first axis...

    return viewer.add_image(data, rgb=False, colormap='gist_earth',
                            name=layer.name + " [%s]" % method, metadata=meta )

""" def bandRatio(numerator='2180:2190+2230', denominator='2190:2210'):
    viewer = napari.current_viewer()  # get viewer
    image, layer = getHyImage(viewer)
    if image is None:
        napari.utils.notifications.show_warning(
            "Warning: Could not find valid HSI data.")
        return

    # evaluate numerator and denominator using calculate
    num = calculate(image, numerator )
    den = calculate(image, denominator )

    br = num.data / den.data
    return viewer.add_image(h2n(br), rgb=False, colormap='gist_earth',
                            name=layer.name + " [(%s)/(%s)]"%(numerator, denominator) )
"""

def calculator( operation : str = '$2 * 2190:2210 / (2150+2230)' ):
    """
    Compute a calculation on a HSI image using simple text notation. Can be used to do e.g. band ratios.

    Args:
        image: The hyperspectral images.
        string: The computation to run. Conventions are that:
            1. All numerical values represent band wavelengths, unless preceded by a b (e.g., b10).
            2. the : operator averages a range of bands (e.g. 1000:1500 averages from 1000 nm to 1500 nm).
            3. Arithmetic operators (+, -, * and /) behave as expected
            4. Constants are flagged using a $ character (e.g. $2).
        For example, a simple band ratio might be expressed as:

        '''
        $2 * 2190:2210 / (2150+2230)
        '''
    """
    viewer = napari.current_viewer()  # get viewer
    image, layer = getHyImage(viewer)
    if image is None:
        napari.utils.notifications.show_warning(
            "Warning: Could not find valid HSI data.")
        return

    result = calculate( image, operation )
    meta = dict(path=layer.metadata['path'],
                type='HSIp',
                wav=np.array(result.get_wavelengths()))
    return viewer.add_image(h2n(result.data), rgb=False, colormap='gist_earth',
                            name=layer.name + " [calc]", metadata=meta )


def calculate(image, op: str = "1000. + $2 * 2190:2210 / (2150+2230)"):
    """

    Args:
        image: A HyImage instance to run a calculation on.
        op: a string defining the operation to compute. Syntax is as follows:
            - 'b': flags that the following number is a band index (e.g. b10)
            - '$': flags that the following number is a constant (e.g., $2 )
            - ':': flags that bands between the previous and the following number should be averaged (e.g.
                    2190:2210 averages all bands between 2190 nm and 2210 nm wavelengths)
            - all other numbers are treated as wavelengths
            - arithmetic operations +, -, / and * are all supported.
    Returns:
        A HyImage instance containing the result of the specified operation.
    """
    # strip spaces (but keep one at start)
    op = ' ' + op.strip()

    # convert non-constants to band indices
    bidx = re.findall('.[0-9.]+', op)
    for s in bidx:
        if s[0] == 'b':  # specified as band index already
            continue
        if s[0] == '$':  # constant - skip for now
            continue
        else:
            ix = image.get_band_index(float(s[1:]))
            op = op.replace(s[1:], 'b' + str(ix))

    # replace all ':' operators with np.mean operations
    bidx = re.findall('[b][0-9]+[:][b][0-9]*[0-9]', op)
    for s in bidx:
        ix0 = int(s.split(':')[0][1:])
        ix1 = int(s.split(':')[1][1:])
        op = op.replace(s, 'np.nanmean( a[..., %d:%d] )' % (ix0, ix1))

    # replace all remaining bands with relevant slices
    bidx = re.findall('[b][0-9]+', op)
    for s in bidx:
        ix = int(s[1:])
        op = op.replace(s, 'a[..., %d]' % ix)

    # finally, remove any $ from constants
    const = re.findall('[$][0-9]+', op)
    for s in const:
        op = op.replace(s, s[1:])

    return hylite.HyImage(eval(op, {'np': np, 'a': image.data}))