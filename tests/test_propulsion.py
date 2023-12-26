import sys
import os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam.propulsion as prop
import tuduam as tud
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
    res = cruise_BEM.optimise_blade(0)

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
    find the original sample.
    """
    data_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tests", "Airfoil_test_data"))
    rho = 1.111617926993772
    dyn_vis = 1.757864864661911e-05
    v_cruise = 72.18676185339652
    soundspeed = 336.4029875015975
    rpm = 1090
    blade_analysis = prop.OffDesignAnalysisBEM(data_path, DSE2021OffDesignAnalysis, v_cruise, rpm, rho, dyn_vis, soundspeed)
    res = blade_analysis.analyse_propeller()

    assert np.isclose(res["thrust"], 157.8187, rtol =0.01)
    assert np.isclose(res["torque"], 128.928187)
    assert np.isclose(res["eff"], 0.774122895)
    assert np.isclose(res["thrust_coeff"], 0.4202)
    assert np.isclose(res["power_coeff"], 2.1445)
    assert all(np.isclose(res["AoA"], np.array([0.04046394, 0.02295662, 0.02008227, 0.01985122, 0.02087762,
       0.01916235, 0.01963098, 0.018492  , 0.01862655, 0.02004781,
       0.02080169, 0.02009016, 0.02117429, 0.02086136, 0.02200812,
       0.02329413, 0.02358909, 0.02526579, 0.0256387 , 0.02782584,
       0.02835683, 0.03022875, 0.03168534, 0.03458972, 0.03868065])))
    assert all(np.isclose(res["lift_coeff"] ,np.array([0.66951785, 0.56388127, 0.56174527, 0.55101608, 0.56202493,
       0.55708196, 0.55727841, 0.55749745, 0.56225514, 0.56252254,
       0.5737981 , 0.57411611, 0.57445696, 0.57842678, 0.58983142,
       0.59025406, 0.60173806, 0.59849811, 0.60994828, 0.621443  ,
       0.62201972, 0.62872631, 0.64024739, 0.65616928, 0.66882847])))
    assert all(np.isclose(res["drag_coeff"] , np.array([0.01685313, 0.01634201, 0.01505498, 0.01501747, 0.01506247,
       0.01420636, 0.01421137, 0.01421695, 0.01361768, 0.01362415,
       0.01367213, 0.01367971, 0.01368783, 0.01322324, 0.01327327,
       0.01328278, 0.01332383, 0.01380911, 0.01386199, 0.01390519,
       0.01391809, 0.01459432, 0.01465052, 0.01561023, 0.01704022])))



if __name__ == "__main__":
    def FixtPropDSE2021():
        data_dict = {
            "r_prop": 0.50292971,
            "n_blades": 6,
            "rpm_cruise": 1350,
            "xi_0": 0.1
        }
        return tud.Propeller(**data_dict)

    data_path =  os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "tests", "Airfoil_test_data"))
    rho = 1.111617926993772
    dyn_vis = 1.757864864661911e-05
    v_cruise = 72.18676185339652
    n_stations = 25
    soundspeed = 336.4029875015975
    thrust = 399.4478198779665
    cruise_BEM = prop.BEM(data_path, FixtPropDSE2021(), rho, dyn_vis, v_cruise, n_stations, soundspeed, T=thrust)
    res = cruise_BEM.optimise_blade(0)
