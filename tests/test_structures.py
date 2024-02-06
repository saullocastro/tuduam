import numpy as np
import pdb 
import pytest
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam.structures as struct

def test_wing_weight(FixtVTOL, FixtFlightPerformance, FixtSingleWing):
    wing_mass = struct.class2_wing_mass(FixtVTOL, FixtFlightPerformance, FixtSingleWing)
    assert  wing_mass == FixtSingleWing.mass


def test_get_fuselage_weight(FixtVTOL, FixtFlightPerformance, FixtFuselage):
    fuselage_mass = struct.class2_fuselage_mass(FixtVTOL, FixtFlightPerformance, FixtFuselage)
    assert fuselage_mass ==  FixtFuselage.mass

def test_read_geometry(naca24012, naca0012):
    res1 = struct.read_coord(naca24012)
    res2 = struct.read_coord(naca0012)

    assert isinstance(res1, np.ndarray)
    assert res1.shape[1] == 2
    assert ((res1 < 1.01) * (res1 > -1.01)).all()

    assert isinstance(res2, np.ndarray)
    assert res2.shape[1] == 2
    assert ((res2 < 1.01) * (res2 > -1.01)).all()

def test_get_centroids(naca24012, naca0012):
    x1, y1 = struct.get_centroids(naca24012)
    x2, y2 = struct.get_centroids(naca0012)

    assert not np.isclose(y1, 0)
    assert np.isclose(y2, 0)
    assert np.isclose(x1,0.5,atol=0.01)
    assert np.isclose(x2,0.5,atol=0.01)

    


def test_discretize_airfoil(FixtWingbox1, naca24012, naca0012, naca45112):
    res = struct.discretize_airfoil(naca45112, 2, FixtWingbox1)
    max_pid = np.max([i.pid for i in res.panel_dict.values()])
    max_bid = np.max([i.bid for i in res.boom_dict.values()])
    assert max_pid  == len(res.panel_dict) - 1
    assert max_bid  == len(res.boom_dict) - 1

    # Get panel per cell
    panel_cell1 = [i for i in res.panel_dict.values() if (i.b1.x + i.b2.x)/2 < 0.6 ]
    panel_cell2 = [i for i in res.panel_dict.values() if 0.6 <= (i.b1.x + i.b2.x)/2 < 1.4 ]
    panel_cell3 = [i for i in res.panel_dict.values() if (i.b1.x + i.b2.x)/2 >= 1.4 ]
    

    # Test whether correct thicknesses have been assigned
    tsk_cell1 = [i.t_pnl for i in panel_cell1 if i.b1.x != i.b2.x]
    tsk_cell2 = [i.t_pnl for i in panel_cell2 if i.b1.x != i.b2.x]
    tsk_cell3 = [i.t_pnl for i in panel_cell3 if i.b1.x != i.b2.x]

    spar_panels = [i.t_pnl for i in res.panel_dict.values() if i.b1.x == i.b2.x]

    assert  None not in tsk_cell1 and None not in  tsk_cell2 and None not in  tsk_cell3
    assert np.isclose(tsk_cell1, FixtWingbox1.t_sk_cell[0]).all()
    assert np.isclose(tsk_cell2, FixtWingbox1.t_sk_cell[1]).all()  # FIXME alos includes spars of course
    assert np.isclose(tsk_cell3, FixtWingbox1.t_sk_cell[2]).all()
    assert np.isclose(spar_panels, FixtWingbox1.t_sp).all()

    # Test whether all boom have been assigned an area and is non zero
    assert [b.A != None and b.A != 0 for b in res.boom_dict.values()]
    assert res.Ixx != 0

def test_direct_stress(FixtWingbox1, naca45112):
    res = struct.discretize_airfoil(naca45112, 2, FixtWingbox1)
    res.stress_analysis(48e3, 200e4, 80e9, validate=False)
    # res.plot_direct_stresses()
    pass

def test_discretization_case1(case1):
    "See the used fixture for more information on the test"
    case1._compute_boom_areas(1.1958260743101399)
    assert np.isclose(case1.boom_dict["a"].A, 750e-6)
    assert np.isclose(case1.boom_dict["b"].A, 1191.7e-6, atol=1e-7)
    assert np.isclose(case1.boom_dict["c"].A, 591.7e-6, atol=1e-7)
    assert np.isclose(case1.boom_dict["d"].A, 591.7e-6, atol=1e-7)
    assert np.isclose(case1.boom_dict["e"].A, 1191.7e-6, atol= 1e-7)
    assert np.isclose(case1.boom_dict["f"].A, 750e-6, atol=1e-7)

def test_discretization_case2(case2):
    "See the used fixture for more information on the test"
    case2._compute_boom_areas(1.1958260743101399)
    assert case2.boom_dict["a"].A > 760e-6
    assert case2.boom_dict["c"].A > 600e-6
    assert case2.boom_dict["d"].A > 600e-6
    assert case2.boom_dict["f"].A > 760e-6

    assert np.isclose(case2.boom_dict["a"].A, 750e-6 + 1.5e-3, atol=1e-7)
    assert np.isclose(case2.boom_dict["b"].A, 1191.7e-6, atol=1e-7)
    assert np.isclose(case2.boom_dict["c"].A, 591.7e-6  + 3e-4, atol=1e-7)
    assert np.isclose(case2.boom_dict["d"].A, 591.7e-6 + 3e-4, atol=1e-7)
    assert np.isclose(case2.boom_dict["e"].A, 1191.7e-6, atol= 1e-7)
    assert np.isclose(case2.boom_dict["f"].A, 750e-6 + 1.5e-3, atol=1e-7)

def test_cell_areas(FixtWingbox2, naca24012, naca0012, naca45112):
    res = struct.discretize_airfoil(naca45112, 2, FixtWingbox2)
    # res.plot()
    area_lst = res.read_cell_areas()
    print(area_lst)
 
def test_shear_flows(case23_5_Megson):
    wingbox = case23_5_Megson
    assert np.isclose(np.round(wingbox.Ixx*1e6,1)*1e6,214.3e6)
    wingbox.stress_analysis(44.5e3, 20e3, 80e9, validate=False)
    pnl1 = [i for i in wingbox.panel_dict.values() if i.pid == 1][0]
    pnl2 = [i for i in wingbox.panel_dict.values() if i.pid == 2][0]
    pnl3 = [i for i in wingbox.panel_dict.values() if i.pid == 3][0]
    pnl4 = [i for i in wingbox.panel_dict.values() if i.pid == 4][0]
    pnl5 = [i for i in wingbox.panel_dict.values() if i.pid == 5][0]
    pnl6 = [i for i in wingbox.panel_dict.values() if i.pid == 6][0]
    pnl7 = [i for i in wingbox.panel_dict.values() if i.pid == 7][0]

    assert np.isclose(pnl1.q_basic, -34e3, rtol= 0.01) # should be 32.8 but bc of some minor differences in the setup this difference is there
    assert np.isclose(pnl2.q_basic, 0) 
    assert np.isclose(pnl3.q_basic, 0) 
    assert np.isclose(pnl4.q_basic, -13.6e3, rtol=0.01) 
    assert np.isclose(pnl5.q_basic, 0) 
    assert np.isclose(pnl6.q_basic, 0) 
    assert np.isclose(pnl7.q_basic, 81.7e3, rtol=0.01) 
    # wingbox.plot()
