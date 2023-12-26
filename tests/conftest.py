import pytest
import numpy as np
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

import tuduam as tud
import tuduam.structures as struct

with open(os.path.join(os.path.dirname(__file__), "setup", "test_values.json")) as f:
    values = json.load(f)

@pytest.fixture
def FixtSingleWing():
    return tud.SingleWing(**values["SingleWing"])

@pytest.fixture
def FixtVTOL():
    return tud.VTOL(**values["VTOL"])

@pytest.fixture
def FixtHybridPowertrain():
    return tud.HybridPowertrain(**values["HybridPowertrain"])

@pytest.fixture
def FixtEngine():
    return tud.Engine(**values["Engine"])

@pytest.fixture
def FixtFlightPerformance():
    return tud.FlightPerformance(**values["FlightPerformance"])

@pytest.fixture
def FixtFuselage():
    return tud.Fuselage(**values["Fuselage"])

@pytest.fixture
def FixtAero():
    return tud.Aerodynamics(**values["Aerodynamics"])

@pytest.fixture
def FixtAirfoil():
    return tud.Airfoil(**values["Airfoil"])

@pytest.fixture
def FixtMaterial():
    return tud.Material(**values["Material"])

@pytest.fixture
def FixtGeometry(FixtAero, FixtAirfoil, FixtEngine, FixtFlightPerformance, FixtMaterial, FixtSingleWing):
    return struct.WingboxGeometry(FixtAero, FixtAirfoil, FixtEngine, FixtFlightPerformance, FixtMaterial, FixtSingleWing)

@pytest.fixture
def FixtInternalForces(FixtAero, FixtAirfoil, FixtEngine, FixtFlightPerformance, FixtMaterial, FixtSingleWing):
    return struct.WingboxInternalForces(FixtAero, FixtAirfoil, FixtEngine, FixtFlightPerformance, FixtMaterial, FixtSingleWing)

@pytest.fixture
def FixtConstraints(FixtAero, FixtAirfoil, FixtEngine, FixtFlightPerformance, FixtMaterial, FixtSingleWing):
    return struct.Constraints(FixtAero, FixtAirfoil, FixtEngine, FixtFlightPerformance, FixtMaterial, FixtSingleWing)

@pytest.fixture
def FixtPropDSE2021():
    data_dict = {
        "r_prop": 0.50292971,
        "n_blades": 6,
        "rpm_cruise": 1350,
        "xi_0": 0.1
    }
    return tud.Propeller(**data_dict)

@pytest.fixture
def DSE2021OffDesignAnalysis():
    data_dict = {
        "r_prop": 0.50292971,
        "n_blades": 6,
        "rpm_cruise": 1350,
        "xi_0": 0.1,
        "chord_arr": np.array([0.03465952, 0.04455214, 0.05395549, 0.06279065, 0.07098139,
       0.07845934, 0.08517839, 0.09109053, 0.09615647, 0.10035978,
       0.10367558, 0.10609018, 0.10758779, 0.10816815, 0.10781527,
       0.10651308, 0.10423676, 0.10094716, 0.09658681, 0.09104805,
       0.08417472, 0.07569153, 0.06508562, 0.05118965, 0.02995261]),
        "pitch_arr":np.array([1.59010535, 1.56060352, 1.5278318 , 1.49881861, 1.47011773,
       1.43827993, 1.41207003, 1.38454137, 1.35397965, 1.32913886,
       1.3030613 , 1.27751229, 1.2507629 , 1.2263168 , 1.20243842,
       1.17913448, 1.15640891, 1.13426302, 1.11269583, 1.09170422,
       1.07128323, 1.05142623, 1.03387054, 1.01860692, 1.00387976]),
       "rad_arr":np.array([0.05934571, 0.07745118, 0.09555665, 0.11366212, 0.13176759,
       0.14987305, 0.16797852, 0.18608399, 0.20418946, 0.22229493,
       0.2404004 , 0.25850587, 0.27661134, 0.29471681, 0.31282228,
       0.33092775, 0.34903322, 0.36713869, 0.38524416, 0.40334963,
       0.4214551 , 0.43956057, 0.45766604, 0.47577151, 0.49387698])
    }
    return tud.Propeller(**data_dict)

