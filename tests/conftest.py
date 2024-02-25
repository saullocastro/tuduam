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
    attr_dict = {
        "r_prop": 0.50292971,
        "n_blades": 6,
        "rpm_cruise": 1350,
        "xi_0": 0.1
    }
    return tud.Propeller(**attr_dict)

@pytest.fixture
def DSE2021OffDesignAnalysis():
    attr_dict = {
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
    return tud.Propeller(**attr_dict)

@pytest.fixture
def FixtWingbox1():
    pass
    attr_dict = {
        "n_cell":3,
        "spar_loc_nondim":[0.3, 0.7],
        "t_sk_cell":[0.002,0.004,0.002],
        "area_str":20e-6,
        "t_sp":0.01,
        "str_cell":[6,8,6],
        "booms_sk": 69,
        "booms_spar": 5,
    }
    return tud.Wingbox(**attr_dict)

@pytest.fixture
def FixtWingbox2():
    pass
    attr_dict = {
        "n_cell":4,
        "spar_loc_nondim":[0.3, 0.5, 0.75],
        "t_sk_cell":[0.004,0.004,0.004, 0.004],
        "area_str":20e-6,
        "t_sp":0.001,
        "str_cell":[4,8,7,20],
        "booms_sk": 100,
        "booms_spar": 15,
    }
    return tud.Wingbox(**attr_dict)

@pytest.fixture
def case1():
    """"This case set ups the wingbox shown on page 560 of Megson"""
    pass
    attr_dict = {
        "n_cell":2,
        "spar_loc_nondim":[0.5],
        "t_sk_cell":[2, 1.5],
        "area_str":0,
        "t_sp":0,
        "str_cell":[1,1],
        "booms_sk": 0,
        "booms_spar": 0,
    }
    data_struct = tud.Wingbox(**attr_dict)

    width = 1.1958260743101399

    boom1 = struct.Boom()
    boom1.x = 0
    boom1.y = 0.4
    boom1.bid = 1

    boom6 = struct.Boom()
    boom6.x = 0
    boom6.y = 0.
    boom6.bid = 6

    boom2 = struct.Boom()
    boom2.x = width/2
    boom2.y = 0.35
    boom2.bid = 2
    
    boom5 = struct.Boom()
    boom5.x = width/2
    boom5.y = 0.05
    boom5.bid = 5
    
    boom3 = struct.Boom()
    boom3.x = width
    boom3.y = 0.3
    boom3.bid = 3
    
    boom4 = struct.Boom()
    boom4.x = width
    boom4.y = 0.1
    boom4.bid = 4

    pnl1 = struct.IdealPanel()
    pnl1.b1 = boom1
    pnl1.bid1 = 1
    pnl1.b2 = boom2
    pnl1.bid2 = 2
    pnl1.t_pnl = 2e-3

    pnl2 = struct.IdealPanel()
    pnl2.b1 = boom2
    pnl2.bid1 = 2
    pnl2.b2 = boom3
    pnl2.bid2 = 3 
    pnl2.t_pnl = 1.5e-3

    pnl3 = struct.IdealPanel()
    pnl3.b1 = boom3
    pnl3.bid1 = 3
    pnl3.b2 = boom4
    pnl3.bid2 = 4
    pnl3.t_pnl = 2e-3

    pnl4 = struct.IdealPanel()
    pnl4.b1 = boom4
    pnl4.bid1 = 4
    pnl4.b2 = boom5
    pnl4.bid2 = 5
    pnl4.t_pnl = 1.5e-3

    pnl5 = struct.IdealPanel()
    pnl5.b1 = boom5
    pnl5.bid1 = 5
    pnl5.b2 = boom6
    pnl5.bid2 = 6
    pnl5.t_pnl = 2e-3

    pnl6 = struct.IdealPanel()
    pnl6.b1 = boom5
    pnl6.bid1 = 5
    pnl6.b2 = boom2
    pnl6.bid2 = 2
    pnl6.t_pnl = 2.5e-3

    pnl7 = struct.IdealPanel()
    pnl7.b1 = boom6
    pnl7.bid1 = 6 
    pnl7.b2 = boom1
    pnl7.bid2 = 1
    pnl7.t_pnl = 3e-3

    wingbox = struct.IdealWingbox(data_struct, width)

    wingbox.x_centroid = 0.6
    wingbox.y_centroid = 0.2
    wingbox.boom_dict = dict(a=boom1,
                             b=boom2,
                             c=boom3,
                             d=boom4,
                             e=boom5,
                             f=boom6
                             )
    wingbox.panel_dict = dict(a=pnl1,
                              b=pnl2,
                              c=pnl3,
                              d=pnl4,
                              e=pnl5,
                              f=pnl6,
                              g=pnl7
                              )
    return wingbox

    

@pytest.fixture
def case2(case1):
    case2 =  case1
    data_struct = case2.wingbox_struct

    data_struct.str_cell = [10, 2]
    data_struct.area_str = 3e-4
    return case2

@pytest.fixture
def naca24012():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "naca24012.txt"))

@pytest.fixture
def naca0012():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "naca0012.txt"))

@pytest.fixture
def naca45112():
    return os.path.realpath(os.path.join(os.path.dirname(__file__), "naca45112.txt"))

@pytest.fixture
def case23_5_Megson():
    """"This case set ups the wingbox of problem 23.5 shown in Megson and will serve as the main verification of the shear
    flows of values"""
    pass
    attr_dict = {
        "n_cell":2,
        "spar_loc_nondim":[635/1398],
        "t_sk_cell":[2, 1.5],
        "area_str":0,
        "t_sp":0,
        "str_cell":[1,1],
        "booms_sk": 0,
        "booms_spar": 0,
    }
    data_struct = tud.Wingbox(**attr_dict)

    width = 1.398

    boom1 = struct.Boom()
    boom1.x = 0
    boom1.y = 0.076
    boom1.A = 1290e-6
    boom1.bid = 1

    boom2 = struct.Boom()
    boom2.x = 0.635
    boom2.y = 0.
    boom2.A = 1936e-6
    boom2.bid = 2

    boom3 = struct.Boom()
    boom3.x = 1.398
    boom3.y = 0.102
    boom3.A = 645e-6
    boom3.bid = 3
    
    boom4 = struct.Boom()
    boom4.x = 1.398
    boom4.y = 0.304
    boom4.A = 645e-6
    boom4.bid = 4
    
    boom5 = struct.Boom()
    boom5.x = 0.635
    boom5.y = 0.406
    boom5.A = 1936e-6
    boom5.bid = 5
    
    boom6 = struct.Boom()
    boom6.x = 0
    boom6.y = 0.330
    boom6.A = 1290e-6
    boom6.bid = 6

    # Create panels
    pnl1 = struct.IdealPanel()
    pnl1.pid = 1
    pnl1.b1 = boom6
    pnl1.bid1 = 6
    pnl1.b2 = boom1
    pnl1.bid2 = 1
    pnl1.t_pnl = 1.625e-3
    pnl1.length = lambda : 0.254 # Force length of the panel

    pnl2 = struct.IdealPanel()
    pnl2.pid = 2
    pnl2.b1 = boom1
    pnl2.bid1 = 1
    pnl2.b2 = boom2
    pnl2.bid2 = 2 
    pnl2.t_pnl = 0.915e-3
    pnl2.length = lambda : 0.647 # Force length of the panel

    pnl3 = struct.IdealPanel()
    pnl3.pid = 3
    pnl3.b1 = boom2
    pnl3.bid1 = 2
    pnl3.b2 = boom3
    pnl3.bid2 = 3
    pnl3.t_pnl = 0.559e-3
    pnl3.length = lambda : 0.775 # Force length of the panel

    pnl4 = struct.IdealPanel()
    pnl4.pid = 4
    pnl4.b1 = boom3
    pnl4.bid1 = 3
    pnl4.b2 = boom4
    pnl4.bid2 = 4
    pnl4.t_pnl = 1.220e-3
    pnl4.length = lambda : 0.202 # Force length of the panel

    pnl5 = struct.IdealPanel()
    pnl5.pid = 5
    pnl5.b1 = boom4
    pnl5.bid1 = 4
    pnl5.b2 = boom5
    pnl5.bid2 = 5
    pnl5.t_pnl = 0.559e-3
    pnl5.length = lambda : 0.775 # force length of the panel

    pnl6 = struct.IdealPanel()
    pnl6.pid = 6
    pnl6.b1 = boom5
    pnl6.bid1 = 5
    pnl6.b2 = boom6
    pnl6.bid2 = 6
    pnl6.t_pnl = 0.915e-3
    pnl6.length = lambda : 0.647 # Force length of the panel

    pnl7 = struct.IdealPanel()
    pnl7.pid = 7
    pnl7.b1 = boom2
    pnl7.bid1 = 2
    pnl7.b2 = boom5
    pnl7.bid2 = 5
    pnl7.t_pnl = 2.032e-3
    pnl7.length = lambda : 0.406 # Force length of the panel

    wingbox = struct.IdealWingbox(data_struct, width)

    wingbox.x_centroid = 0
    wingbox.y_centroid = 0.203
    wingbox.boom_dict = dict(a=boom1,
                             b=boom2,
                             c=boom3,
                             d=boom4,
                             e=boom5,
                             f=boom6
                             )
    wingbox.panel_dict = dict(a=pnl1,
                              b=pnl2,
                              c=pnl3,
                              d=pnl4,
                              e=pnl5,
                              f=pnl6,
                              g=pnl7
                              )
    wingbox.read_cell_areas = lambda : [0.232, 0.258]
    
    # Some placeholder classes so we can get the correct centroid without chaning 
    # the source code. The normal function for the centroids does not work
    # due the assumptions of the leading and trailing edge
    class centroid:
        def __init__(self, arr) -> None:
            self.xy = arr

    class Foo:
        def __init__(self, arr) -> None:
            self.centroid = centroid(arr)

    cell1 = Foo([[0.3175],[0.203]])
    cell2 = Foo([[1.0165],[0.203]])
    wingbox.get_polygon_cells = lambda : [cell1, cell2]
    return wingbox
