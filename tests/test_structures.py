import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

import tuduam.structures as struct

def test_wing_weight(FixtVTOL, FixtFlightPerformance, FixtSingleWing):
    wing_mass = struct.class2_wing_mass(FixtVTOL, FixtFlightPerformance, FixtSingleWing)
    assert  wing_mass == FixtSingleWing.mass


def test_get_fuselage_weight(FixtVTOL, FixtFlightPerformance, FixtFuselage):
    fuselage_mass = struct.class2_fuselage_mass(FixtVTOL, FixtFlightPerformance, FixtFuselage)
    assert fuselage_mass ==  FixtFuselage.mass

def test_get_x_le(FixtGeometry, FixtSingleWing):
    res = FixtGeometry.get_x_le(FixtSingleWing.span/2)
    print(res)
    assert res > 6
    pass


def test_moment_y_from_tip(FixtInternalForces, FixtSingleWing):
    res = FixtInternalForces.moment_y_from_tip(np.linspace(0,FixtSingleWing.span,20))
    assert isinstance(res, np.ndarray)


