import numpy as np
import matplotlib as mpl
C = np.array((38,41,48))/255
mpl.rcParams["figure.facecolor"] = C
mpl.rcParams["axes.facecolor"] = C
mpl.rcParams["savefig.facecolor"] = C
C = 'white'
mpl.rcParams['text.color'] = C
mpl.rcParams['axes.labelcolor'] = C
mpl.rcParams['xtick.color'] = C
mpl.rcParams['ytick.color'] = C


from qtpy.QtWidgets import QWidget, QSizePolicy, QVBoxLayout, QDockWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from magicgui import magicgui
import napari

from ._guiBase import GUIBase
from napari_hippo import getHyImage


# matplotlib widgets
# (blatantly stolen from :
#  https://biapol.github.io/blog/johannes_mueller/entry_user_interf2/#creating-advanced-standalone-guis )
class MplCanvas(FigureCanvas):
    """
    Defines the canvas of the matplotlib window
    """

    def __init__(self):
        self.fig = Figure()                         # create figure
        self.axes = self.fig.add_subplot(111)       # create subplot

        FigureCanvas.__init__(self, self.fig)       # initialize canvas
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

class Caterpillar(QWidget):
    def __init__(self, napari_viewer, image_layer, points_layer, median=True, quartiles=True):
        super().__init__()

        self.image_layer = image_layer
        self.points_layer = points_layer.name # store name only in case of changes
        self.plot_median = median
        self.median = None # this will be computed later
        self.plot_quartiles = quartiles
        self.quartiles = None # this will be computed later

        self.viewer = napari_viewer
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        # set layout and add them to widget
        self.vbl = QVBoxLayout()
        self.vbl.addWidget(self.toolbar)
        self.vbl.addWidget(self.canvas)
        self.setLayout(self.vbl)

        # ensure points layer is 2D, if not, flatten it
        points_layer = self.viewer.layers[self.points_layer]
        if points_layer.data.shape[-1] >= 3:
            # delete and re-add points layer as 2D
            d = points_layer.data[:, 1:]
            n = points_layer.name
            fc = points_layer.face_color
            del self.viewer.layers[n]
            if len(d) > 0:
                points_layer = self.viewer.add_points(d, name=n, face_color=fc)
            else:
                points_layer = self.viewer.add_points(ndim=2, name=n)
        points_layer.opacity = 0.6

        # connect change event with plotting
        points_layer.events.set_data.connect( self.update_plot )

        # update plot
        self.update_plot( None )

    def update_plot(self, event):
        # get image
        image, _ = getHyImage(self.viewer, layer=self.image_layer)
        if image is None:
            return

        # clear / reset plot
        self.canvas.axes.clear()
        self.canvas.axes.grid()

        # plot median?
        if self.plot_median:
            if self.median is None:
                self.median = np.nanmedian(image.data, axis=(0,1) )
            self.canvas.axes.plot( image.get_wavelengths(), self.median, color='cyan', lw=2 )

        # plot quartiles?
        if self.plot_quartiles:
            if self.quartiles is None:
                self.quartiles = np.nanpercentile( image.data, (25,75), axis=(0,1) )

            self.canvas.axes.plot(image.get_wavelengths(), self.quartiles[0], color='cyan', lw=1, ls=':')
            self.canvas.axes.plot(image.get_wavelengths(), self.quartiles[1], color='cyan', lw=1, ls=':')
            self.canvas.axes.fill( np.hstack([image.get_wavelengths(),image.get_wavelengths()[::-1]]),
                                   np.hstack([self.quartiles[0], self.quartiles[1][::-1]]),
                                   color='cyan', alpha=0.2 )

        # plot points
        points_layer = self.viewer.layers[self.points_layer]
        for pxy, c in zip(points_layer.data, points_layer.face_color):
            x = int(pxy[1])
            y = int(pxy[0])
            if (x >= 0) and (x < image.xdim()) and (y >= 0) and ( y < image.ydim() ):
                self.canvas.axes.plot(image.get_wavelengths(), image.data[x,y,:], color=c )

        # update matplotlib plot
        self.canvas.draw()

class CaterpillarWidget(GUIBase):
    def __init__(self, napari_viewer):
        super().__init__(napari_viewer)

        self.caterpillar_widget = magicgui(addCaterpillar, call_button='Create', auto_call=False)
        self._add([self.caterpillar_widget], 'Spectral Caterpillar')


def addCaterpillar( base_image : 'napari.layers.Image', query_points : 'napari.layers.Points', median=True, quartiles=True ):
    """
    Add a spectral caterpillar plot window and associated query points layer.
    Args:
        base_image : The image to plot spectra from.
        query_points : Layer containing the points to plot.
        npoints: The number of points to add. By default these will be located on endmember spectra using
                 N-Findr.
        median: Add the median spectra of the whole image for reference.
        quartiles: Add the 25th and 75th percentile of the whole image for reference.

    Returns:

    """
    viewer = napari.current_viewer()

    # get points
    widget = Caterpillar(viewer, base_image, query_points, median, quartiles )
    viewer.window.add_dock_widget(widget, area='bottom')




