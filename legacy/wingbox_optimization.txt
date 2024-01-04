import sys
import os

sys.path.insert(0,os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))


import tuduam as tud
from legacy.legacy_structures  import wingbox_optimization
from warnings import warn

warn("This script is not supported anymore and might fail. \
Script is likely be deprecated in the future", DeprecationWarning)





aero_dict = {
    "spanwise_points": 29,
    "alpha_zero_lift": 0,
 }

engine_dict  = {
    "n_engines": 2,
    "mass": 100,
    "diameter": 1.4,
    "y_rotor_loc": [ 2],
    "x_rotor_loc": [3],
    "thrust": 3e3,
 }


material_dict  = {
    "young_modulus":3e9,
    "shear_modulus":4000000000.0,
    "poisson":0.3,
    "density":1600,
    "beta_crippling":1.42,
    "sigma_ultimate":407000000.0,
    "sigma_yield":407000000.0,
    "g_crippling":5,
    "safety_factor":1.2,
 }

flight_perf_dict = {
    "n_ult":3.5,
    "cL_cruise":0.3,
    "wingloading":1000,
    "v_cruise":80,
    "h_cruise":1000,
}

wingclass_dict  = {
    "aspect_ratio": 9.4,
    "taper":0.45, 
    "quarterchord_sweep":0.01, 
    "washout":0, 
    "x_le_root_global":3, 
    "wingbox_start":0.15, 
    "wingbox_end":0.75, 
    "n_ribs":10, 
    "n_str":4, 
    "spar_thickness":0.008, 
    "fl_thickness":0.008, 
    "stringer_height":0.010, 
    "stringer_width":0.005, 
    "stringer_thickness":0.002, 
    "skin_thickness":0.003, 
 }

AirfoilStruct = tud.Airfoil(thickness_to_chord=0.17, cl_alpha=6)
AeroStruct = tud.Aerodynamics(**aero_dict)
EngStruct = tud.Engine(**engine_dict)
FlightperfStruct = tud.FlightPerformance(**flight_perf_dict)
MatStruct = tud.Material(**material_dict)
VTOLStruct = tud.VTOL(mtom=2300)
WingStruct = tud.SingleWing(**wingclass_dict)


tud.planform_geometry(FlightperfStruct, VTOLStruct, WingStruct)
tud.lift_curve_slope(AeroStruct, WingStruct)
res = wingbox_optimization(AeroStruct, AirfoilStruct, EngStruct, FlightperfStruct,  MatStruct, WingStruct )



