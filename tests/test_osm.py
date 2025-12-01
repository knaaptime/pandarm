import os
import tempfile

import numpy as np
import osmnx as ox
import pandas as pd
import pytest

import pandarm as pdna

from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal


@pytest.fixture
def tmpfile(request):
    fname = tempfile.NamedTemporaryFile().name

    def cleanup():
        if os.path.exists(fname):
            os.remove(fname)
    request.addfinalizer(cleanup)

    return fname


@pytest.fixture(scope='module')
def osm_network():
    irvine = ox.geocode_to_gdf("irvine, ca")
    network = pdna.Network.from_gdf(irvine)    
    return network

def test_osm_network_download(osm_network):

    # this is a liberal test because the network can evolve over time
    assert osm_network.nodes_df.shape[0] >= 42691

    assert_array_equal(np.array(['x', 'y', 'geometry']), osm_network.nodes_df.columns)
    assert_array_equal(np.array(['from', 'to', 'length', 'geometry']), osm_network.edges_df.columns)

def test_save_hdf_with_geoms(osm_network, tmpfile):
    osm_network.save_hdf5(tmpfile)
    
    store = pd.HDFStore(tmpfile)
    
    assert store['nodes'].shape[0] >= 42691
    
    roundtrip_net = pdna.Network.from_hdf5(tmpfile)
    
    assert_frame_equal(osm_network.nodes_df, roundtrip_net.nodes_df)