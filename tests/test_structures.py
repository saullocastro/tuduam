import sys
import pathlib as pl
import numpy as np
from warnings import warn

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
    res = FixtGeometry.I_sp_fl_x(0.002, 0)*1e12
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

def test_str_weight_from_tip(FixtGeometry):
    res =  FixtGeometry.str_weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.003))
    assert isinstance(res, np.ndarray)
    assert res[-1] == 0
    assert np.isclose(res[0],FixtGeometry.get_area_str(0.010,0.010,0.02)*FixtGeometry.wing.span/2*FixtGeometry.material.density*FixtGeometry.wing.n_str*2)

def test_le_wingbox_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res =  FixtGeometry.le_wingbox_weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.003))
    assert isinstance(res, np.ndarray)
    assert res[-1] == 0
    warn("No proper test has been implemted here yet")

def test_te_wingbox_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res1 =  FixtGeometry.te_wingbox_weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.003))
    res2 =  FixtGeometry.te_wingbox_weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.006))

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    assert np.isclose(res1[-1], 0) and np.isclose(res2[-1],0)
    assert (res2[:-1] > res1[:-1]).all()

def test_fl_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res1 =  FixtGeometry.fl_weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.003))
    res2 =  FixtGeometry.fl_weight_from_tip((0.006, 0.010, 0.010, 0.02, 0.003))
    root_flange = ref.width_wingbox*ref.chord(0)*2*ref.material.density*0.003
    tip_flange = ref.width_wingbox*ref.chord(ref.wing.span/2)*2*ref.material.density*0.003
    total_weight_flange = ref.wing.span/4*(root_flange + tip_flange)

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    assert np.isclose(res1[-1], 0) and np.isclose(res2[-1],0)
    assert np.isclose(res1[0], total_weight_flange, rtol= 0.02)
    assert (res2[:-1] > res1[:-1]).all()

def test_spar_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res1 =  FixtGeometry.spar_weight_from_tip((0.003, 0.010, 0.010, 0.02, 0.003))
    res2 =  FixtGeometry.spar_weight_from_tip((0.006, 0.010, 0.010, 0.02, 0.003))
    root_flange = ref.height(0)*2*ref.material.density*0.003
    tip_flange = ref.height(ref.wing.span/2)*2*ref.material.density*0.003
    total_weight_flange = ref.wing.span/4*(root_flange + tip_flange)

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    assert np.isclose(res1[-1], 0) and np.isclose(res2[-1],0)
    assert np.isclose(res1[0], total_weight_flange)
    assert (res2[:-1] > res1[:-1]).all()

def test_rib_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    rib_weight = ref.chord(ref.y) *ref.height(ref.y) *ref.t_rib *ref.material.density
    res = FixtGeometry.rib_weight_from_tip()

    assert isinstance(res, np.ndarray)
    assert res[0] == np.sum(rib_weight)

def test_total_weight(FixtGeometry):
    res = FixtGeometry.weight_from_tip((0.006, 0.010, 0.010, 0.02, 0.003))
    res2 = FixtGeometry.total_weight((0.006, 0.010, 0.010, 0.02, 0.003))

    assert isinstance(res, np.ndarray)
    assert isinstance(res2, float)

def test_moment_y_from_tip(FixtInternalForces, FixtSingleWing):
    res = FixtInternalForces.moment_y_from_tip(np.linspace(0,FixtSingleWing.span,20))
    assert isinstance(res, np.ndarray)


