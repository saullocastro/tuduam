import pytest
import sys
import pathlib as pl
import os
import json

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

import tuduam as tud

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
