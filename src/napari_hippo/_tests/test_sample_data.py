# from napari_hippo import make_sample_data

import os
import napari_hippo.testdata
from napari_hippo._sample_data import make_sample_data
from napari_hippo import n2h, h2n
import numpy as np

def test_sampleimage():
    assert os.path.exists(napari_hippo.testdata.image)
    sample = make_sample_data()

    # get data array and play with it a bit
    data = sample[0][0]
    assert h2n( n2h(data) ).shape == data.shape
    assert data.shape == (450, 38, 23 )
    assert n2h(data).shape == (23, 38, 450)

    # also test h2n here with three and for channel RGB data
    for i in [3,4]:
        data = np.zeros( (100,200,i) )
        assert h2n(data).shape == (200,100,i)
        assert h2n(n2h(data)).shape == data.shape