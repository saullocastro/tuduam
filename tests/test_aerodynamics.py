import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

import tuduam.aerodynamics as aero
import matplotlib.pyplot as plt
from warnings import warn
import pdb


def test_wing_geometry(FixtFlightPerformance, FixtVTOL, FixtSingleWing):
    aero.planform_geometry(FixtFlightPerformance, FixtVTOL, FixtSingleWing)
    warn("Proper testcases still have to be build, current tests only check for code execution")

def test_lift_distribution(FixtAero, FixtSingleWing):
    lift_func = aero.lift_distribution(FixtAero, FixtSingleWing, 3*3.14/180, 1.225, 80)
    warn("Proper testcases still have to be build, current tests only check for code execution ")








