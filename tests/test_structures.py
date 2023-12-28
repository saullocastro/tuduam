import numpy as np
import pdb 
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam.structures as struct
import tuduam.structures_new as struct1

def test_wing_weight(FixtVTOL, FixtFlightPerformance, FixtSingleWing):
    wing_mass = struct.class2_wing_mass(FixtVTOL, FixtFlightPerformance, FixtSingleWing)
    assert  wing_mass == FixtSingleWing.mass


def test_get_fuselage_weight(FixtVTOL, FixtFlightPerformance, FixtFuselage):
    fuselage_mass = struct.class2_fuselage_mass(FixtVTOL, FixtFlightPerformance, FixtFuselage)
    assert fuselage_mass ==  FixtFuselage.mass

def test_read_geometry(naca24012, naca0012):
    res1 = struct1.read_coord(naca24012)
    res2 = struct1.read_coord(naca0012)

    assert isinstance(res1, np.ndarray)
    assert res1.shape[1] == 2
    assert ((res1 < 1.01) * (res1 > -1.01)).all()

    assert isinstance(res2, np.ndarray)
    assert res2.shape[1] == 2
    assert ((res2 < 1.01) * (res2 > -1.01)).all()

def test_get_centroids(naca24012, naca0012):
    x1, y1 = struct1.get_centroids(naca24012)
    x2, y2 = struct1.get_centroids(naca0012)

    assert not np.isclose(y1, 0)
    assert np.isclose(y2, 0)
    assert np.isclose(x1,0.5,atol=0.01)
    assert np.isclose(x2,0.5,atol=0.01)


def test_discretize_airfoil(FixtWingbox, naca24012, naca0012, naca45112):
    res = struct1.discretize_airfoil(naca45112, 2, FixtWingbox)
    res.plot()


    pass
