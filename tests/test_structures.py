import numpy as np
import types
import pdb
import pytest
from scipy.constants import g
from scipy.integrate import quad
from warnings import warn

import tuduam.structures as struct

def test_wing_weight(FixtVTOL, FixtFlightPerformance, FixtSingleWing):
    wing_mass = struct.class2_wing_mass(FixtVTOL, FixtFlightPerformance, FixtSingleWing)
    assert  wing_mass == FixtSingleWing.mass


def test_get_fuselage_weight(FixtVTOL, FixtFlightPerformance, FixtFuselage):
    fuselage_mass = struct.class2_fuselage_mass(FixtVTOL, FixtFlightPerformance, FixtFuselage)
    assert fuselage_mass ==  FixtFuselage.mass


#----------------- Testing geometry class --------------------------
def test_rib_placement(FixtGeometry):
    assert len(FixtGeometry.wing.rib_loc) == FixtGeometry.wing.n_ribs
    assert FixtGeometry.wing.rib_loc[-1] == FixtGeometry.wing.span/2
    assert FixtGeometry.wing.rib_loc[0] == 0

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

def test_ellipse_polar(FixtGeometry):
    res1 = FixtGeometry.rad_ellipse_polar(0, 20,10)
    res2 = FixtGeometry.rad_ellipse_polar(np.pi/2, 15, 3)
    res3 = FixtGeometry.rad_ellipse_polar(np.pi, 24,10)
    res4 = FixtGeometry.rad_ellipse_polar(0.3587706702705722, 10,5)

    assert np.isclose(res1, 20 )
    assert np.isclose(res2, 3 )
    assert np.isclose(res3, 24 )
    assert np.isclose(res4, 8.54400374531753 )

    with pytest.raises(Exception):
        FixtGeometry.rad_ellipse_polar(0,10 ,10.1)

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
    assert np.isclose(res, ((FixtSingleWing.chord_root*(1 - FixtSingleWing.wingbox_end))**2 + (FixtAirfoil.thickness_to_chord*FixtSingleWing.chord_root/2)**2 )**0.5)

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
    res = FixtGeometry.I_sp_fl_z(0.002,0.002, 0)*1e12
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
    res =  FixtGeometry.I_xx((0.003, 0.003, 0.010, 0.010, 0.02, 0.003))*1e12
    assert isinstance(res, np.ndarray)
    assert np.isclose(res[0], 3.7517e8, rtol=1e-3, atol=1e-3)

def test_I_zz(FixtGeometry):
    # Checked by performing calculation at the root on paper using the 
    # already tested functions for stringers, flanges et cetera
    res =  FixtGeometry.I_zz((0.003, 0.003, 0.010, 0.010, 0.02, 0.003))*1e12
    assert isinstance(res, np.ndarray)
    assert np.isclose(res[0], 2.8419e9)

def test_str_weight_from_tip(FixtGeometry):
    res =  FixtGeometry.str_weight_from_tip((0.003, 0.003, 0.010, 0.010, 0.02, 0.003))
    assert isinstance(res, np.ndarray)
    assert res[-1] == 0
    assert np.isclose(res[0],FixtGeometry.get_area_str(0.010,0.010,0.02)*FixtGeometry.wing.span/2*FixtGeometry.material.density*FixtGeometry.wing.n_str*2*g)

def test_le_wingbox_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res =  FixtGeometry.le_wingbox_weight_from_tip((0.003, 0.003, 0.010, 0.010, 0.02, 0.003))
    assert isinstance(res, np.ndarray)
    assert res[-1] == 0
    warn("No proper test has been implemted here yet")

def test_te_wingbox_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res1 =  FixtGeometry.te_wingbox_weight_from_tip((0.003,0.003, 0.010, 0.010, 0.02, 0.003))
    res2 =  FixtGeometry.te_wingbox_weight_from_tip((0.003,0.003, 0.010, 0.010, 0.02, 0.006))

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    assert np.isclose(res1[-1], 0) and np.isclose(res2[-1],0)
    assert (res2[:-1] > res1[:-1]).all()

def test_fl_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res1 =  FixtGeometry.fl_weight_from_tip((0.003,0.003, 0.010, 0.010, 0.02, 0.003))
    res2 =  FixtGeometry.fl_weight_from_tip((0.006, 0.006,0.010, 0.010, 0.02, 0.003))
    root_flange = ref.width_wingbox*ref.chord(0)*2*ref.material.density*0.003*g
    tip_flange = ref.width_wingbox*ref.chord(ref.wing.span/2)*2*ref.material.density*0.003*g
    total_weight_flange = ref.wing.span/4*(root_flange + tip_flange)

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    assert np.isclose(res1[-1], 0) and np.isclose(res2[-1],0)
    assert np.isclose(res1[0], total_weight_flange, rtol= 0.02)
    assert (res2[:-1] > res1[:-1]).all()

def test_spar_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res1 =  FixtGeometry.spar_weight_from_tip((0.003,0.003, 0.010, 0.010, 0.02, 0.003))
    res2 =  FixtGeometry.spar_weight_from_tip((0.006,0.006, 0.010, 0.010, 0.02, 0.003))
    root_flange = ref.height(0)*2*ref.material.density*0.003*g
    tip_flange = ref.height(ref.wing.span/2)*2*ref.material.density*0.003*g
    total_weight_flange = ref.wing.span/4*(root_flange + tip_flange)

    assert isinstance(res1, np.ndarray)
    assert isinstance(res2, np.ndarray)
    assert np.isclose(res1[-1], 0) and np.isclose(res2[-1],0)
    assert np.isclose(res1[0], total_weight_flange)
    assert (res2[:-1] > res1[:-1]).all()

def test_rib_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    rib_weight = ref.chord(ref.rib_loc) *ref.height(ref.rib_loc) *ref.wing.t_rib *ref.material.density
    res = FixtGeometry.rib_weight_from_tip()

    assert isinstance(res, np.ndarray)
    assert np.isclose(res[0], np.sum(rib_weight)*g)

def test_engine_weight_from_tip(FixtGeometry):
    ref = FixtGeometry
    res = FixtGeometry.engine_weight_from_tip()

    assert isinstance(res, np.ndarray)
    assert res[0] == ref.engine.mass*ref.engine.n_engines/2*g

    ref.engine.ignore_loc = [10]

    with pytest.raises(Exception):
        ref.engine_weight_from_tip()

def test_total_weight(FixtGeometry):
    res = FixtGeometry.weight_from_tip((0.006,0.006, 0.010, 0.010, 0.02, 0.003))
    res2 = FixtGeometry.total_weight((0.006,0.006, 0.010, 0.010, 0.02, 0.003))

    assert isinstance(res, np.ndarray)
    assert isinstance(res2, float)


#----------------------- tests for internal forces -----------------------

def test_shear_z_from_tip(FixtInternalForces, FixtFlightPerformance):
    ref = FixtInternalForces
    res = FixtInternalForces.shear_z_from_tip((0.006,0.006, 0.010, 0.010, 0.02, 0.003))
    root_check = ref.lift_func(0)*FixtFlightPerformance.n_ult - ref.total_weight((0.006,0.006, 0.010, 0.010, 0.02, 0.003)) - ref.engine.mass*ref.engine.n_engines/2*g
    tip_check = ref.lift_func(ref.wing.span/2)*FixtFlightPerformance.n_ult - ref.weight_from_tip((0.006,0.006, 0.010, 0.010, 0.02, 0.003))[-1]

    assert isinstance(res, np.ndarray)
    assert all(res <= 100) # Normally should be smaller than 0 however last rib as slighlty more weight than lift
    assert np.isclose(res[0], -root_check) # Check summation and sign at the end
    assert np.isclose(res[-1], -tip_check) # Check summation and sign at the tip

def test_moment_x_from_tip(FixtInternalForces, FixtSingleWing):
    res = FixtInternalForces.moment_x_from_tip((0.006,0.006, 0.010, 0.010, 0.02, 0.003))
    assert isinstance(res, np.ndarray)
    assert len(res) == len(FixtInternalForces.rib_loc)
    assert all(res >= 0)
    assert res[-1] == 0

def test_moment_y_from_tip(FixtInternalForces, FixtSingleWing):
    ref = FixtInternalForces

    res = FixtInternalForces.moment_y_from_tip()
    assert isinstance(res, np.ndarray)
    assert all(res <= 0)
    assert res[-1] == 0
    assert np.isclose(res[0], -4652, rtol=0.01) # Value checked with simple example done by hand

    ref.engine.x_rotor_loc = None
    ref.engine.x_rotor_rel_loc = [0.5, 0.5, 0.5, 0.5]
    res2 =  ref.moment_y_from_tip()

    assert isinstance(res2, np.ndarray)
    assert not all(np.isclose(res, res2))

    ref.engine.x_rotor_loc = [3,3,3,3]
    ref.engine.x_rotor_rel_loc = [0.5, 0.5, 0.5, 0.5]

    with pytest.raises(Exception):
        ref.moment_y_from_tip()

    ref.engine.ignore_loc = [10,3,5,0]
    ref.engine.x_rotor_rel_loc =  None

    with pytest.raises(Exception):
        ref.moment_y_from_tip()

def test_bending_stress_y_from_tip(FixtInternalForces, FixtSingleWing):
    res = FixtInternalForces.bending_stress_y_from_tip((0.006,0.006, 0.010, 0.010, 0.02, 0.003))
    assert isinstance(res, np.ndarray)
    assert all(res >= 0) # Because of tension in the lower side
    assert res[-1] == 0

def test_max_shearflow_no_torque(FixtInternalForces, FixtSingleWing):

    # Prepare the test with the forces required
    ref = FixtInternalForces  # Easy reference for conciseness
    ref.rib_loc = np.array([0]) 
    sample = (0.006,0.006, 0.010, 0.010, 0.02, 0.002)
    virt_shear = 1000
    virt_torq = 0
    virt_Ixx = 1e-4
    virt_height = 0.6
    virt_chord = 2


    #Dynamically overwrite internal properties
    shear_overwrite = lambda x,y: [virt_shear]
    torque_overwrite = lambda x: [virt_torq]
    Ixx_overwrite = lambda x,y: [virt_Ixx]
    height_overwrite = lambda x,y: np.array([virt_height])
    chord_overwrite = lambda x,y: np.array([virt_chord])

    ref.shear_z_from_tip = types.MethodType(shear_overwrite, ref)
    ref.moment_y_from_tip = types.MethodType(torque_overwrite, ref)
    ref.I_xx = types.MethodType(Ixx_overwrite, ref)
    ref.height = types.MethodType(height_overwrite, ref)
    ref.chord = types.MethodType(chord_overwrite, ref)
    shear_max, base_shear_flow_func_lst, cut_shear_lst = ref.shearflow_max_from_tip(sample)

    q01 = cut_shear_lst[0]
    q02 = cut_shear_lst[1]
    q03 = cut_shear_lst[2]

    qt1 = cut_shear_lst[3]
    qt2 = cut_shear_lst[4]
    qt3 = cut_shear_lst[5]

    # Horizontal basic forces
    func_hor_reg1 = lambda x: base_shear_flow_func_lst[0](x)*np.sin(x)*ref.wing.wingbox_start*virt_chord 
    func_hor_reg10 = lambda x: base_shear_flow_func_lst[9](x)*np.sin(x)*ref.wing.wingbox_start*virt_chord 

    force_basic_hor_1 = quad(func_hor_reg1, 0, np.pi/2)[0]
    force_basic_hor_10 = quad(func_hor_reg10, -np.pi/2, 0)[0]
    force_basic_hor_3 = quad(lambda x: base_shear_flow_func_lst[2](x) , 0, ref.width_wingbox*virt_chord)[0]
    force_basic_hor_8 = quad(lambda x: base_shear_flow_func_lst[7](x) ,0, ref.width_wingbox*virt_chord)[0]
    force_basic_hor_5 = quad(lambda x: base_shear_flow_func_lst[4](x), 0 ,  ref.l_sk_te(0))[0]*((1 - ref.wing.wingbox_end)*virt_chord)/(ref.l_sk_te(0))
    force_basic_hor_6 = quad(lambda x: base_shear_flow_func_lst[5](x) ,0, ref.l_sk_te(0))[0]*((1 - ref.wing.wingbox_end)*virt_chord)/(ref.l_sk_te(0))

    # Vertical basic forces
    func_vert_reg1 = lambda x: base_shear_flow_func_lst[0](x)*np.cos(x)*ref.wing.wingbox_start*virt_chord 
    func_vert_reg10 = lambda x: base_shear_flow_func_lst[9](x)*np.cos(x)*ref.wing.wingbox_start*virt_chord 

    force_basic_ver_1 = quad(func_vert_reg1, 0, np.pi/2)[0]
    force_basic_ver_10 = quad(func_vert_reg10, 0, -np.pi/2)[0]
    force_basic_ver_2 = quad(lambda x: base_shear_flow_func_lst[1](x),0, virt_height/2)[0]
    force_basic_ver_9 = quad(lambda x: base_shear_flow_func_lst[8](x),0, -virt_height/2)[0]
    force_basic_ver_4 = quad(lambda x: base_shear_flow_func_lst[3](x),0, virt_height/2)[0]
    force_basic_ver_7 = quad(lambda x: base_shear_flow_func_lst[6](x),0, virt_height/2)[0]
    force_basic_ver_5 = quad(lambda x: base_shear_flow_func_lst[4](x), 0, ref.l_sk_te(0))[0]*virt_height/2/(ref.l_sk_te(0))
    force_basic_ver_6 = quad(lambda x: base_shear_flow_func_lst[5](x), 0, ref.l_sk_te(0))[0]*virt_height/2/(ref.l_sk_te(0))


    # res_hor_force = np.sum([force_reg_hor_1, force_reg_hor_10, force_reg_hor_3,
    #                           -force_reg_hor_8, force_reg_hor_5,force_reg_hor_6])


    assert np.isclose(qt1,0)
    assert np.isclose(qt2,0)
    assert np.isclose(qt3,0)
    #Check the basic shearflows and their contributions
    assert np.isclose(force_basic_hor_1, -270)
    assert np.isclose(force_basic_hor_3, -18360)
    assert np.isclose(force_basic_hor_5, -14985.24896657004, atol = 3 )
    assert np.isclose(force_basic_hor_6, -14985.24896657004, atol = 3 )

    assert np.isclose(force_basic_ver_1, -115.884)
    assert np.isclose(force_basic_ver_2, -270)
    assert np.isclose(force_basic_ver_4, -270)
    assert np.isclose(force_basic_ver_5, -8991.149379942024, atol= 3)
    assert np.isclose(force_basic_ver_6, -8991.149379942024, atol= 3)

    # assert np.isclose(res_vert_force,virt_shear)
    # assert np.isclose(res_hor_force,0)

    #  check whether the resultant force equals vy and vx. Vx should be 0 and vy should be 1000.

