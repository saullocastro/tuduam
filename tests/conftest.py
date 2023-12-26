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
        "chord_arr": np.array([0.03574108, 0.04591601, 0.05557541, 0.06463885, 0.073029  ,
       0.08067707, 0.08753596, 0.09355469, 0.09870661, 0.10296327,
       0.10630526, 0.10871533, 0.11019308, 0.1107256 , 0.11030286,
       0.10891001, 0.10652333, 0.10310445, 0.0985919 , 0.09289188,
       0.08583214, 0.07713958, 0.06629445, 0.05211147, 0.03047527]),
        "pitch_arr":np.array([1.58312402, 1.55362219, 1.52085047, 1.49183727, 1.46313639,
       1.43304392, 1.40508868, 1.37581469, 1.34874362, 1.32390283,
       1.29782528, 1.27053093, 1.24552687, 1.22108077, 1.19720238,
       1.17389844, 1.15117287, 1.12902698, 1.10745979, 1.08821351,
       1.06779251, 1.04793552, 1.03037983, 1.01337087, 0.99864371]),
       "rad_arr":np.array([0.05934571, 0.07745118, 0.09555665, 0.11366212, 0.13176759,
       0.14987305, 0.16797852, 0.18608399, 0.20418946, 0.22229493,
       0.2404004 , 0.25850587, 0.27661134, 0.29471681, 0.31282228,
       0.33092775, 0.34903322, 0.36713869, 0.38524416, 0.40334963,
       0.4214551 , 0.43956057, 0.45766604, 0.47577151, 0.49387698])
    }
    return tud.Propeller(**data_dict)

