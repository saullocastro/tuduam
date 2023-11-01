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


#----------------- Testing geometry class --------------------------
def test_x_to_global(FixtGeometry, FixtSingleWing):
    # simple test mimics the behaviour in the function
    coord = 2
    res = FixtGeometry._x_to_global(coord)
    assert res == (coord + FixtSingleWing.x_le_root_global)

def test_perimeter_ellipse( FixtGeometry):
    # Checked for consistency with the output of the following tool:
    # https://www.mathsisfun.com/geometry/ellipse-perimeter.html#tool
    res = FixtGeometry.perimiter_ellipse(10,6)
    assert np.isclose(res,51.05397279)

def test_chord(FixtGeometry, FixtSingleWing):
    #simple test edge cases tests
    res1 = FixtGeometry.chord(FixtSingleWing.span/2)
    res2 = FixtGeometry.chord(0)

    assert np.isclose(res1, FixtSingleWing.chord_tip)
    assert np.isclose(res2, FixtSingleWing.chord_root)

def test_height(FixtGeometry, FixtAirfoil, FixtSingleWing):
    res = FixtGeometry.height(0)

    assert np.isclose(res, FixtAirfoil.thickness_to_chord*FixtSingleWing.chord_root)

def test_l_sk_te(FixtGeometry, FixtSingleWing, FixtAirfoil):
    res = FixtGeometry.l_sk_te(0)
    assert np.isclose(res, ((FixtSingleWing.chord_root*0.25)**2 + (FixtAirfoil.thickness_to_chord*FixtSingleWing.chord_root)**2 )**0.5)

def test_get_area_str(FixtGeometry):
    # Simply checked manually 
    res = FixtGeometry.get_area_str(h_st=0.010, w_st=0.010, t_st=0.002)*1e6
    assert np.isclose(res, 60)

def test_I_st_x(FixtGeometry):
    # Checked for consistency using the following tool
    # https://structx.com/Shape_Formulas_007.html 
    res = FixtGeometry.I_st_x(h_st=0.010, w_st=0.010, t_st=0.002)*1e12
    assert  np.isclose(res, 1620)

def test_I_st_z(FixtGeometry):
    # Checked for consistency using the following tool
    # https://structx.com/Shape_Formulas_007.html 
    res = FixtGeometry.I_st_z(h_st=0.010, w_st=0.010, t_st=0.002)*1e12
    assert  np.isclose(res, 980)

def test_l_sp(FixtGeometry):
    res =  FixtGeometry.l_sp(0)
    assert np.isclose(res, FixtGeometry.height(0))

def test_l_fl(FixtGeometry, FixtSingleWing):
    res = FixtGeometry.l_fl(0)
    assert np.isclose(res, (FixtGeometry.width_wingbox*FixtSingleWing.chord_root))

def test_I_sp_fl_x(FixtGeometry):
    # Checked for equivalency using the following tool
    # https://calcresource.com/moment-of-inertia-rtube.html
    res = FixtGeometry.I_sp_fl_x(0.002,0.002, 0)*1e12
    assert np.isclose(res, 1.77483288903039e8, rtol= 0, atol= 1e-5)

def test_I_sp_fl_z(FixtGeometry):
    # Checked for equivalency using the following tool
    # https://calcresource.com/moment-of-inertia-rtube.html
    res = FixtGeometry.I_sp_fl_z(0.002, 0)*1e12
    assert np.isclose(res, 1.25383088143245e9 , rtol= 0, atol= 1e-3)


def test_get_x_le(FixtGeometry, FixtSingleWing):
    res = FixtGeometry.get_x_le(FixtSingleWing.span/2)
    assert np.isclose(res, FixtSingleWing.x_le_root_global + np.tan(FixtSingleWing.sweep_le)*FixtSingleWing.span/2 )

def test_get_x_start_wb(FixtGeometry, FixtSingleWing):
    res = FixtGeometry.get_x_start_wb(0)
    assert np.isclose(res, FixtSingleWing.x_le_root_global + FixtSingleWing.chord_root*FixtSingleWing.wingbox_start)

def test_get_x_end_wb(FixtGeometry, FixtSingleWing):
    res = FixtGeometry.get_x_end_wb(0)
    assert np.isclose(res, FixtSingleWing.x_le_root_global + FixtSingleWing.chord_root*FixtSingleWing.wingbox_end)

def test_I_xx(FixtGeometry):
    # Checked by performing calculation at the root on paper using the 
    # already tested functions for stringers, flanges et cetera
    res =  FixtGeometry.I_xx((0.003, 0.010, 0.010, 0.02, 0.003))*1e12
    assert isinstance(res, np.ndarray)
    assert np.isclose(res[0], 3.7517e8, rtol=1e-3, atol=1e-3)

def test_I_zz(FixtGeometry):
    # Checked by performing calculation at the root on paper using the 
    # already tested functions for stringers, flanges et cetera
    res =  FixtGeometry.I_zz((0.003, 0.010, 0.010, 0.02, 0.003))*1e12
    assert isinstance(res, np.ndarray)
    assert np.isclose(res[0], 2.8419e9)

# def test_weight_from_tip(FixtGeometry):
#     res =  FixtGeometry.weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.003))*1e12
#     assert isinstance(res, np.ndarray)

def test_moment_y_from_tip(FixtInternalForces, FixtSingleWing):
    res = FixtInternalForces.moment_y_from_tip(np.linspace(0,FixtSingleWing.span,20))
    assert isinstance(res, np.ndarray)


