# load custom pytest fixtures
from .fixtures import *
import numpy as np
import hylite
import glob
import napari_hippo, glob

def test_cleanAnnotations(imageMode, stackMode):
    for M in [stackMode, imageMode]:
        viewer, layer = M()

        # open annotation toolbox
        # create our widget, passing in the viewer, and add it to napari as a dock
        from napari_hippo._annotationTools import AnnotToolsWidget

        w = AnnotToolsWidget(viewer)
        viewer.window.add_dock_widget(w)

        # create/load annotations
        from napari_hippo._annotationTools import loadAnnot, setLabel, saveAnnot, export
        roi, points = loadAnnot()
        assert ('[KP] image' in viewer.layers) or ('[KP] batch' in viewer.layers)
        assert ('[ROI] image' in viewer.layers) or ('[ROI] batch' in viewer.layers)
        
        # cleanup
        dir = os.path.dirname( napari_hippo.testdata.image )
        if os.path.exists(os.path.join(dir, 'segments.hyc')):
            import shutil
            shutil.rmtree(os.path.join(dir, 'segments.hyc'))
        if os.path.exists(os.path.join(dir, 'segments.hdr')):
            os.remove(os.path.join(dir, 'segments.hdr'))
        for p in glob.glob( os.path.join(dir, '*.txt')): # check files exist and cleanup
            os.remove( p )
        for p in glob.glob( os.path.join(dir, '*.npz')): # check files exist and cleanup
            os.remove( p )
        
        # remove point / ROI keys from header files so they don't 
        # interfere with other tests
        from hylite import io
        for l in [points[0], roi[0]]:
            path = l.layer.metadata['path']
            if not isinstance(path, list):
                path = [path]
            for p in path:
                p = os.path.splitext(p)[0] + '.hdr' # ensure we've got the header file
                header = io.loadHeader(p)
                todel = []
                for k,v in header.items():
                    if k.lower().startswith('point') or k.lower().startswith('roi'):
                        todel.append(k)
                for k in todel:
                    del header[k]
                io.saveHeader(p,header)


def test_annotation(imageMode, stackMode, capsys):

    for M in [stackMode, imageMode]:
        viewer, layer = M()

        # open annotation toolbox
        # create our widget, passing in the viewer, and add it to napari as a dock
        from napari_hippo._annotationTools import AnnotToolsWidget

        w = AnnotToolsWidget(viewer)
        viewer.window.add_dock_widget(w)

        # create/load annotations
        from napari_hippo._annotationTools import loadAnnot, setLabel, saveAnnot, export
        roi, points = loadAnnot()
        assert ('[KP] image' in viewer.layers) or ('[KP] batch' in viewer.layers)
        assert ('[ROI] image' in viewer.layers) or ('[ROI] batch' in viewer.layers)
        
        # edit them (as though done in the GUI)
        if M == stackMode:
            # add new 3D polygon
            roi[0].layer.add([np.array([(0,0,0),(0,0,10),(0,10,10),(0,10,0)])],
                             shape_type='polygon')
            roi[0].layer.add([np.array([(1,0,0),(1,15,10),(1,2,5)])],
                             shape_type='polygon') # add to second layer of stack
            points[0].layer.add(np.array([(0,0,0),(0,0,10),(0,10,10),(0,10,0)]))
            points[0].layer.add(np.array([(1,5,5)])) # add to second layer of stack
        else:
            # add new 2D polygon
            roi[0].layer.add([np.array([(0,0),(0,10),(10,10),(10,0)])],
                             shape_type='polygon')
            points[0].layer.add(np.array([(0,0),(0,10),(10,10),(10,0)]))

        # update to add labels
        setLabel("Cool Test")
        assert roi[0].layer.text.values[-1] == "Cool Test"
        assert points[0].layer.text.values[-1] == "Cool Test"
        
        # add a second KPs (to check spectral library export can handle this)
        if M == stackMode:
            # add new 3D polygon
            roi[0].layer.add([np.array([(0,20,0),(0,22,7),(0,22,5)])],
                             shape_type='polygon') # add to second layer of stack
            points[0].layer.add(np.array([(0,12,7)]))
            points[0].layer.add(np.array([(1,2,5)])) # add to second layer of stack
        else:
            # add new 2D polygon
            roi[0].layer.add([np.array([(20,20),(12,7),(2,5)])],
                             shape_type='polygon')
            points[0].layer.add(np.array([(18,12),(12,14)]))
        setLabel("MoreStuff")

        # export ROIs / KPs
        dir = os.path.dirname( napari_hippo.testdata.image )
        viewer.layers.selection = [] # deselect all
        export(os.path.join(dir, 'segments.hyc') )
        assert os.path.exists( os.path.join(dir, 'segments.hyc') )
        assert os.path.exists( os.path.join(dir, 'segments.hyc/Cool_Test.hdr') )
        assert os.path.exists( os.path.join(dir, 'segments.hyc/block1.lib') )
        assert os.path.exists( os.path.join(dir, 'segments.hyc/block2.lib') )

        # save ROI / KP layers
        saveAnnot('header')
        saveAnnot('csv')
        saveAnnot('numpy')
        txt = glob.glob( os.path.join(dir, '*.txt') )
        assert len(txt) > 0
        assert len(glob.glob( os.path.join(dir, '*.npz'))) > 0

        # check outputs have different sizes in in stack mode
        if M == stackMode:
            fsize = [os.path.getsize( p ) for p in glob.glob( os.path.join(dir, '*poly.txt') )]
            assert len(np.unique(fsize)) > 1
            fsize = [os.path.getsize( p ) for p in glob.glob( os.path.join(dir, '*point.txt') )]
            assert len(np.unique(fsize)) > 1

        #viewer.show(block=True)
        #break
    
    # cleanup
    test_cleanAnnotations(imageMode, stackMode)



    
