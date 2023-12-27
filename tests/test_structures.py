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

def test_read_geometry():
    coord_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tests", "airfoil_geometry.txt"))
    res = struct1.read_coord(coord_path)

    assert isinstance(res, np.ndarray)
    assert res.shape[1] == 2
    assert ((res < 1.01) * (res > -1.01)).all()

def test_create_panels():
    pass

def test_discretize_airfoil(FixtWingbox):
    coord_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tests", "airfoil_geometry.txt"))
    res = struct1.discretize_airfoil(coord_path, 2, FixtWingbox)
    pdb.set_trace()


    pass
