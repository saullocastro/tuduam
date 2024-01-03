import sys
import os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam.propulsion as prop
import pytest
import numpy as np

def test_extract_data():
    test_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "examples", "data"))
    res_arr = prop.extract_data_dir(test_path)
    assert isinstance(res_arr, np.ndarray)
    assert res_arr.shape[1] == 8
    assert all(res_arr[:,-1] > 9e5)
    assert all(res_arr[:,0] < 30)

def test_interpolators():
    test_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "examples", "data"))
    alpha_interp = prop.alpha_xfoil_interp(test_path)
    cl_interp = prop.cl_xfoil_interp(test_path)
    cd_interp = prop.cd_xfoil_interp(test_path)

    # Test whether NearestInterpolator chooses the correct values with random samples from the specified folder
    assert np.isclose(alpha_interp([[0.8038, 42e6]]), 2.8)
    assert np.isclose(cl_interp([[15.4, 39e6]]), 2.01)
    assert np.isclose(cd_interp([[1.7082, 13e6]]), 0.01268)


def test_BEM(FixtPropDSE2021):
    """" 
    Check for equivalence with the original code in the DSE2021 repository, specifically script
    DSE2021/PropandPower/final_blade_design.py. Some relative tolerance is allowed since the manner in which
    the airfoil properties are extracted is vastly differnt. In this repository 
    NearestNDInterpolator is used and in DSE2021 a Reynolds spacing is used where all files are then read out.

    Note:
    ---------------------------------
    Test results are now almost fully identical. However beside bugs in this repository, two bugs were found in the
    original repository which are repeated throughout the source code.

    - A typo mistake was made with the files (see line 992 in final_blade_design.py) where file_up was read twice and file_down none
    - An array was incorrectly copied, creating a pointer instead of a copy. (see e.g line 1047). It is supposed to be .copy() 

    Afterwards identical results were reached
    ---------------------------------
    """
    data_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tests", "Airfoil_test_data"))
    rho = 1.111617926993772
    dyn_vis = 1.757864864661911e-05
    v_cruise = 72.18676185339652
    n_stations = 25
    soundspeed = 336.4029875015975
    thrust = 399.4478198779665
    cruise_BEM = prop.BEM(data_path, FixtPropDSE2021, rho, dyn_vis, v_cruise, n_stations, soundspeed, T=thrust)
    cruise_BEM_power = prop.BEM(data_path, FixtPropDSE2021, rho, dyn_vis, v_cruise, n_stations, soundspeed, P=thrust*v_cruise)
    res = cruise_BEM.optimise_blade(0)

    # Test for exception when raising both thrust and power
    with pytest.raises(Exception):
        except_test = prop.BEM(data_path, FixtPropDSE2021, rho, dyn_vis, v_cruise, n_stations, soundspeed, T=thrust, P=thrust*v_cruise)

    assert np.isclose(res["v_e"], 99.13960660958429)
    assert np.isclose(res["tc"], 0.17356211071732652)
    assert np.isclose(res["zeta"], 0.37340175881265036)
    assert np.isclose(res["pc"], 0.21498749746974827)
    assert all(np.isclose(FixtPropDSE2021.chord_arr, 
                      np.array([0.03465952, 0.04455214, 0.05395549, 0.06279065, 0.07098139,
                        0.07845934, 0.08517839, 0.09109053, 0.09615647, 0.10035978,
                        0.10367558, 0.10609018, 0.10758779, 0.10816815, 0.10781527,
                        0.10651308, 0.10423676, 0.10094716, 0.09658681, 0.09104805,
                        0.08417472, 0.07569153, 0.06508562, 0.05118965, 0.02995261])))
    assert all(np.isclose(np.degrees(FixtPropDSE2021.pitch_arr), 
                                np.array([91.10632547, 89.4159953 , 87.53831421, 85.87598075, 84.23154125,
                                82.40736999, 80.90565315, 79.32837689, 77.57731939, 76.15404688,
                                74.65991317, 73.19606257, 71.66343559, 70.26277712, 68.89464653,
                                67.55942928, 66.25734967, 64.98848416, 63.75277513, 62.55004457,
                                61.38000762, 60.24228566, 59.2364187 , 58.36187723, 57.51807313])))
    assert all(np.isclose(np.degrees(res["alpha_arr"]), 
                        np.array([6.7, 6.7, 6.5, 6.5, 6.5, 6.3, 6.4, 6.4, 6.2, 6.3, 6.3, 6.3, 6.2,
                            6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.2, 6.3, 6.5, 6.7])))
    assert all(np.isclose(res["drag_to_lift_arr"], 
                                np.array([0.01935407, 0.01934073, 0.01745779, 0.01744791, 0.01743913,
       0.01620175, 0.01628944, 0.01628428, 0.01535172, 0.01544255,
       0.01544057, 0.01543958, 0.01477399, 0.01477493, 0.01477682,
       0.01477967, 0.01478349, 0.01478828, 0.01536667, 0.01537369,
       0.01538176, 0.01539087, 0.01623836, 0.01748357, 0.01937263])))
    

def test_offdesign(DSE2021OffDesignAnalysis):
    """"
    Check for equivalence with the off design analysis code with the original code. Please see :test_BEM: for further explanation and where to 
    find the original sample. The bugfixes specified in :test_BEM:  were implented as well, also the initial estimate of the Reynolds
    number was made using 1090 as this is internally done in this reposiotry. I don't think it would matter
    for the end result however.
    """
    data_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tests", "Airfoil_test_data"))
    rho = 1.111617926993772
    dyn_vis = 1.757864864661911e-05
    v_cruise = 72.18676185339652
    soundspeed = 336.4029875015975
    rpm = 1090
    blade_analysis = prop.OffDesignAnalysisBEM(data_path, DSE2021OffDesignAnalysis, v_cruise, rpm, rho, dyn_vis, soundspeed)
    res = blade_analysis.analyse_propeller(delta_pitch=0)
    

    assert np.isclose(res["thrust"], 159.9499511)
    assert np.isclose(res["torque"], 130.9214)
    assert np.isclose(res["eff"], 0.77263)
    assert np.isclose(res["thrust_coeff"], 0.425920857)
    assert np.isclose(res["power_coeff"], 2.177700363)
    assert all(np.isclose(res["AoA"], np.array([0.04744176, 0.02797871, 0.02570013, 0.02553307, 0.02510486,
                                            0.02350286, 0.02441464, 0.02501597, 0.02300074, 0.02425103,
                                            0.02516243, 0.02511891, 0.02447518, 0.02553313, 0.02675604,
                                            0.02701844, 0.02828004, 0.02866974, 0.0300271 , 0.03141794,
                                            0.03194067, 0.03291112, 0.03525129, 0.03906602, 0.0424161 ])))
    assert all(np.isclose(res["lift_coeff"] ,np.array([0.71119423, 0.59582504, 0.58529195, 0.59413737, 0.58354448,
                                                    0.57891257, 0.58988066, 0.59011234, 0.58411223, 0.59537095,
                                                    0.59567611, 0.59600843, 0.59636288, 0.60764481, 0.60805522,
                                                    0.60849354, 0.61988117, 0.62037903, 0.63184504, 0.64335694,
                                                    0.64395399, 0.65047667, 0.66201734, 0.67775645, 0.69012942])))
    assert all(np.isclose(res["drag_coeff"] , np.array([0.01708873, 0.01648528, 0.01643733, 0.01518123, 0.01513424,
                                                        0.01427807, 0.01432406, 0.01432969, 0.01368949, 0.01373696,
                                                        0.013744  , 0.01375167, 0.01375985, 0.01380964, 0.01381897,
                                                        0.01382893, 0.01387024, 0.01388138, 0.01393426, 0.01398782,
                                                        0.0140008 , 0.01467716, 0.01474375, 0.01571399, 0.01715453])))



def test_plotblade(DSE2021OffDesignAnalysis, naca24012):
    tst = True
    DSE2021OffDesignAnalysis.tc_ratio = 0.12
    plotblade = prop.PlotBlade(DSE2021OffDesignAnalysis, naca24012)
    plotblade.plot_blade(tst)
    plotblade.plot_3D(tst)
    plotblade.plot_3D_plotly(tst)
