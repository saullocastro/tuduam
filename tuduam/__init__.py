import sys
import os

sys.path.append(os.path.dirname(__file__))

from data_structures import *
from aerodynamics import lift_curve_slope, wing_geometry
from structures import wingbox_optimization

__all__ = ["aerdynamics", "data_structures", "structures"]
