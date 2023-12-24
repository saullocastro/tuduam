import sys
import os

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam.aerodynamics as aero
from warnings import warn


def test_wing_geometry(FixtFlightPerformance, FixtVTOL, FixtSingleWing):
    aero.planform_geometry(FixtFlightPerformance, FixtVTOL, FixtSingleWing)
    warn("Proper testcases still have to be build, current tests only check for code execution")

def test_lift_distribution(FixtAero, FixtSingleWing):
    lift_func = aero.lift_distribution(FixtAero, FixtSingleWing, 3*3.14/180, 1.225, 80)
    warn("Proper testcases still have to be build, current tests only check for code execution ")








