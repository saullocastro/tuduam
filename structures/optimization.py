import multiprocessing
from copy import deepcopy
import warnings
from typing import List

import numpy as np
import scipy.optimize as sop
from pymoo.core.problem import Problem, ElementwiseProblem

# StarmapParallelization moved between pymoo versions. Try the current location first,
# then the older location, and finally provide a small local wrapper so the rest of
# the file can keep using `StarmapParallelization`.
try:
    # newer pymoo versions
    from pymoo.core.parallelization import StarmapParallelization
except Exception:
    try:
        # older pymoo versions
        from pymoo.core.problem import StarmapParallelization
    except Exception:
        # Fallback wrapper: has the same construction pattern used in this file:
        # `StarmapParallelization(pool.starmap)`
        class StarmapParallelization:  # noqa: N801
            def __init__(self, starmap_func):
                # The real pymoo class exposes the mapping function; tests in this module
                # only need the presence of `.starmap` and isinstance checks to work.
                self.starmap = starmap_func

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling


from ..data_structures import *
from .wingbox import *
from .constraints import IsotropicWingboxConstraints


class ProblemFreePanel(ElementwiseProblem):
    """
    The following problem sets up a discrete variable problem which optimizes the amount of stringers per cell. This is done 
    in a separate problem as changing the amount of stringers changes the amount of panels and hence the amount of constraints. 
    Most optimizers do not allow this, hence the following problem uses the output of the :class:`WingboxFixedPanel` to abide 
    by the constraints.

    

    Parameters
    ----------
    shear_y : float
        Shear force applied in the y-direction.
    shear_x : float
        Shear force applied in the x-direction.
    moment_y : float
        Bending moment around the y-axis.
    moment_x : float
        Bending moment around the x-axis.
    applied_loc : float
        Location at which the load is applied.
    chord : float
        Chord length of the wingbox section.
    len_sec : float
        Length of the wingbox section.
    upper_bnds: list
        A list of N + 4 long containing the upper bounds, where N is the amount of cells. The bounds are presented in the following order t_sk_cell, t_sp, t_st, w_st, h_st. 
    lower_bnds: list
        A list of N + 4 long containing the lower bounds, where N is the amount of cells. The bounds are presented in the following order t_sk_cell, t_sp, t_st, w_st, h_st. 
    box_struct : Wingbox
        Wingbox structure, which includes details about geometry, materials, and discretization.
    mat_struct : Material
        Material properties used in the wingbox.
    path_coord : str
        Path to the file containing the airfoil coordinates.
    kwargs_intern : dict
        Additional keyword arguments to pass settings and options to the genetic algorithm optimization functions.

    Attributes
    ----------
    box_params : list
        List to store parameters of the wingbox for optimization.
    shear_y : float
        Shear force applied in the y-direction.
    shear_x : float
        Shear force applied in the x-direction.
    moment_y : float
        Bending moment around the y-axis.
    moment_x : float
        Bending moment around the x-axis.
    applied_loc : float
        Location at which the load is applied.
    chord : float
        Chord length of the wingbox section.
    len_sec : float
        Length of the wingbox section.
    box_struct : Wingbox
        Wingbox structure, which includes details about geometry, materials, and discretization.
    mat_struct : Material
        Material properties used in the wingbox.
    path_coord : str
        Path to the file containing the airfoil coordinates.
    kwargs_intern : dict
        Additional keyword arguments for the optimization functions. Mostly used to pass any keyword arguments to ProblemFixedPanel
    ``` 


    def __init__(self,
        shear_y: float,
        shear_x: float,
        moment_y: float,
        moment_x: float,
        applied_loc: float,
        chord: float,
        len_sec: float,
        upper_bnds: list,
        lower_bnds: list,
        box_struct: Wingbox,
        mat_struct: Material,
        path_coord: str,
        **kwargs_intern,
    ):
        # Initialization logic here...
        super().__init__()  # Call to superclass constructor

# Remaining class implementation ...