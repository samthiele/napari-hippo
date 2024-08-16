# load custom pytest fixtures
from .fixtures import *
import numpy as np
import hylite

def test_annotation(imageMode, stackMode, capsys):

    for M in [stackMode, imageMode]:
        viewer, layer = M()

        # open annotation toolbox
        # create our widget, passing in the viewer, and add it to napari as a dock
        from napari_hippo._annotationTools import AnnotToolsWidget

        w = AnnotToolsWidget(viewer)
        viewer.window.add_dock_widget(w)

        # create/load annotations
        from napari_hippo._annotationTools import loadAnnot, setLabel, saveAnnot
        roi, points = loadAnnot()
        assert ('[KP] image' in viewer.layers) or ('[KP] batch' in viewer.layers)
        assert ('[ROI] image' in viewer.layers) or ('[ROI] batch' in viewer.layers)

        # edit them (as though done in the GUI)
        if 'batch' in roi[0].layer.name:
            # add new 3D polygon
            roi[0].layer.add([np.array([(0,0,0),(0,0,10),(0,10,10),(0,10,0)])],
                             shape_type='polygon')
            points[0].layer.add(np.array([(0,0,0),(0,0,10),(0,10,10),(0,10,0)]))
        else:
            # add new 2D polygon
            roi[0].layer.add([np.array([(0,0),(0,10),(10,10),(10,0)])],
                             shape_type='polygon')
            points[0].layer.add(np.array([(0,0),(0,10),(10,10),(10,0)]))

        # update to add labels
        setLabel("Cool Test")
        assert roi[0].layer.text.values[-1] == "Cool Test"
        assert points[0].layer.text.values[-1] == "Cool Test"

        # save them
        import napari_hippo, glob
        saveAnnot('header')
        saveAnnot('csv')
        saveAnnot('numpy')
        dir = os.path.dirname( napari_hippo.testdata.image )
        assert len(glob.glob( os.path.join(dir, '*.txt'))) > 0
        assert len(glob.glob( os.path.join(dir, '*.npz'))) > 0
        #viewer.show(block=True)
        #break

        # cleanup
        for p in glob.glob( os.path.join(dir, '*.txt')): # check files exist and cleanup
            os.remove( p )
        for p in glob.glob( os.path.join(dir, '*.npz')): # check files exist and cleanup
            os.remove( p )

        viewer.close()
        del layer
        del viewer