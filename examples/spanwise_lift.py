import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import tuduam.aerodynamics as aero
from tuduam import *

dict_wing ={
    "aspect_ratio": 10,
    "taper": 0.45,
    "quarterchord_sweep": np.radians(10),
    "washout":np.radians(2)
}

dict_vtol ={
    "mtom": 2200,
}

dict_flightperf ={
    "wingloading_cruise": 1000,
}


Airfoilstruct = Airfoil(cl_alpha=5.52)
WingStruct  = SingleWing(**dict_wing)
VTOLStruct  = VTOL(**dict_vtol)
FlightPerfStruct  = FlightPerformance(**dict_flightperf)
AeroStruct = Aerodynamics()

aero.wing_planform(FlightPerfStruct, VTOLStruct, WingStruct)
aero.weissinger_l(Airfoilstruct, WingStruct, np.radians(20), 20 )
aero.get_aero_planform(AeroStruct, Airfoilstruct, WingStruct, 20, plot= True )
