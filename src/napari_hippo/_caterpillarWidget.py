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
from napari_hippo import getLayer
import pathlib
import hylite
import os
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
        image = getLayer(self.image_layer, viewer=self.viewer)
        if image is None:
            return
        image = image.toHyImage()

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
            pxy = points_layer.data_to_world(pxy)
            pxy = self.image_layer.world_to_data( pxy )
            if pxy.shape[0] > 2:
                x = int(pxy[2])
                y = int(pxy[1])
            else:
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

        self.export_widget = magicgui(export, call_button='Export', auto_call=False,
                                      filename={"mode": "w"},
                                      format={"choices": ['csv',
                                                          'txt','lib']})
        
        function_widgets = [
            self.caterpillar_widget,
            self.export_widget,
        ]
        function_labels = [
            "Spectral Caterpillar",
            "Export library",
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


def addCaterpillar( base_image : 'napari.layers.Image', query_points : 'napari.layers.Points', median=False, quartiles=False ):
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
    image = getLayer( base_image )
    if (image is None) or (image.ndim() != 3):
        napari.utils.notifications.show_warning(
        "Please select a valid HSICube image.")
        return

    # get points
    widget = Caterpillar(viewer, base_image, query_points, median, quartiles )
    viewer.window.add_dock_widget(widget, area='bottom')

def export( base_image : 'napari.layers.Image', 
            query_points : 'napari.layers.Points', 
            size : int = 4,
            format : str = 'csv',
            filename : pathlib.Path = pathlib.Path('') ):
    """
    Export the spectra from the specified points to a spectral library.
    """
    viewer = napari.current_viewer()

    # get image
    image = getLayer( base_image, viewer=viewer )
    if (image is None) or (image.ndim() != 3):
        napari.utils.notifications.show_warning(
        "Please select a valid HSICube image.")
        return
    image = image.toHyImage()

    # assembl spectral library
    indices = []
    names = []
    for i, (pxy, c) in enumerate(zip(query_points.data, query_points.face_color)):
        # convert to image coordinates to allow for affine transforms
        pxy = base_image.world_to_data( query_points.data_to_world(pxy) )
        if pxy.shape[0] > 2:
            x = int(pxy[2])
            y = int(pxy[1])
        else:
            x = int(pxy[1])
            y = int(pxy[0])
        if (x >= 0) and (x < image.xdim()) and (y >= 0) and ( y < image.ydim() ):
            indices.append( (x,y) )
            names.append("P%d"%(i+1))
    if len(indices) > 0:
        lib = hylite.hylibrary.from_indices( image, indices, s=size, names=names )
        if 'csv' in format:
            filename = os.path.splitext(filename)[0] + ".csv"
            hylite.io.libraries.saveLibraryCSV( filename, lib )
        elif 'txt' in format:
            filename = os.path.splitext(filename)[0] + ".txt"
            hylite.io.libraries.saveLibraryTXT( filename, lib )
        elif 'lib' in format:
            filename = os.path.splitext(filename)[0] + ".lib"
            hylite.io.libraries.saveLibraryLIB( filename, lib )
        napari.utils.notifications.show_info("Exported %d spectra."%lib.data.shape[0])


