"""This library contains methods to size eVTOLs in various areas of design. The package has its origin
in two libraries namely Aetheria and Wigeon. """


from tuduam.data_structures import *
from tuduam.performance import ISA
from tuduam.aerodynamics import lift_curve_slope, planform_geometry
from tuduam.propulsion import BEM, OffDesignAnalysisBEM, PlotBlade

