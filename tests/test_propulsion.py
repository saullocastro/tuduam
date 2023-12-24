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
    the airfoil properties were chosen base on the reynolds number is slightly different. In this repository 
    NearestNDInterpolator is used and in DSE2021 a Reynolds spacing is used.

    Note:
    ---------------------------------
    For now test results are acceptable however in some case like the drag to lift the difference is quite big
    and should be verified that these changes are truly the result of the different I/O mechanicsm with
    new interpolators.
    ---------------------------------
    """
    data_path = r"C:\Users\damie\OneDrive\Desktop\Damien\UAM_course\tuduam\tests\Airfoil_test_data"
    rho = 1.111617926993772
    dyn_vis = 1.757864864661911e-05
    v_cruise = 72.18676185339652
    n_stations = 25
    soundspeed = 336.4029875015975
    thrust = 399.4478198779665
    cruise_BEM = prop.BEM(data_path, FixtPropDSE2021, rho, dyn_vis, v_cruise, n_stations, soundspeed, T=thrust)
    res = cruise_BEM.optimise_blade(0)

    assert np.isclose(res["v_e"], 99.13960660958429, rtol= 0.01)
    assert np.isclose(res["tc"], 0.17356211071732652, rtol= 0.01)
    assert np.isclose(res["zeta"], 0.3734085378029197, rtol= 0.03)
    assert np.isclose(res["pc"], 0.21499231530433627, rtol= 0.03)
    assert all(np.isclose(FixtPropDSE2021.chord_arr, 
                      np.array([0.03574108, 0.04591601, 0.05557541, 0.06463885, 0.073029,   0.08067707,
                                0.08753596, 0.09355469, 0.09870661, 0.10296327, 0.10630526, 0.10871533,
                                0.11019308, 0.1107256,  0.11030286, 0.10891001, 0.10652333, 0.10310445,
                                0.0985919,  0.09289188, 0.08583214, 0.07713958, 0.06629445, 0.05211147,0.03047527]), rtol=0.1))
    assert all(np.isclose(np.degrees(FixtPropDSE2021.pitch_arr), 
                                np.array([90.70632482, 89.01599445, 87.13831317, 85.47597952, 83.83153985, 82.10736842,
                        80.50565142, 78.82837499, 77.27731735, 75.85404469, 74.35991086, 72.79606013,
                        71.36343304, 69.96277447, 68.59464377, 67.25942644, 65.95734675, 64.68848117,
                        63.45277207, 62.35004145, 61.18000446, 60.04228245, 59.03641545, 58.06187395,
                        57.21806982]), rtol=0.01))
    assert all(np.isclose(np.degrees(res["alpha_arr"]), 
                        np.array([6.3, 6.3, 6.1, 6.1, 6.1, 6.,  6.,  5.9, 5.9, 6.,  6.,  5.9, 5.9, 5.9, 5.9, 5.9, 5.9, 5.9,
                                5.9, 6.,  6.,  6.,  6.1, 6.2, 6.4]),rtol=0.07))
    assert all(np.isclose(res["drag_to_lift_arr"], 
                                np.array([0.01943381, 0.01940926, 0.01755582, 0.01753586, 0.01751703, 0.01637003,
                                            0.01635453, 0.01551153, 0.01549882, 0.0155737,  0.01556295, 0.01489036,
                                            0.01488201, 0.01487465, 0.01486827, 0.01486287, 0.01485847, 0.01485506,
                                            0.01485266, 0.0155124,  0.01551201, 0.01551269, 0.01634816, 0.01748214,
                                            0.01931771]),atol=0.01))
    


