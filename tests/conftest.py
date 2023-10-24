import pytest
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
