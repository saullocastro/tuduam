import multiprocessing
from copy import deepcopy
import warnings
from typing import List

import numpy as np
import scipy.optimize as sop
from pymoo.core.problem import Problem, ElementwiseProblem
# StarmapParallelization was moved between pymoo versions – try the new location first,
# fall back to the old location, otherwise disable parallelization support.
try:
    from pymoo.core.parallelization import StarmapParallelization
except Exception:
    try:
        from pymoo.core.problem import StarmapParallelization
    except Exception:
        StarmapParallelization = None

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
    """


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
        """
        Initializes an instance of the optimization problem class, setting up all the necessary parameters and structures needed for structural analysis and optimization of an aircraft wingbox.

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

        """


        self.box_params: list = list()
        self.shear_y = shear_y
        self.shear_x = shear_x
        self.moment_y = moment_y
        self.moment_x = moment_x
        self.applied_loc = applied_loc
        self.chord = chord
        self.len_sec = len_sec
        self.upper_bnds = upper_bnds
        self.lower_bnds = upper_bnds
        self.box_struct = box_struct
        self.mat_struct = mat_struct
        self.path_coord = path_coord
        self.kwargs_intern = (
            kwargs_intern  #  Used to pass any settings to GA_optimize functions
        )

        super().__init__(
            n_var=box_struct.n_cell,
            n_obj=1,
            n_ieq_constr=0,
            xl=5,
            xu=40,
            vtype=int,
            **kwargs_intern,
        )

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """
        This function gets evaluated for each 'element', where an element in this case is a single sequence of n_var long. 
        For more information, please see the documentation of the pymoo library.

        Parameters
        ----------
        x : np.ndarray
            A list of n_var long containing the amount of stringers per cell.
        out : dict
            A dictionary containing the constraints and objective.
        """

        n_cell = self.box_struct.n_cell
        copy_struct = self.box_struct.model_copy()
        copy_struct.str_cell = x

        if len(x) != n_cell:
            raise RuntimeError(
                "The flattened  design vector received does not match the wingbox properties"
            )

        sec_opt = SectionOpt(
            self.path_coord, self.chord, self.len_sec, copy_struct, self.mat_struct
        )

        # Elementwise runner is removed otherwise it clashes in the GA_optimize function
        try:
            self.kwargs_intern.pop("elementwise_runner")
        except KeyError:
            pass

        GA_res = sec_opt.GA_optimize(
            self.shear_y,
            self.shear_x,
            self.moment_y,
            self.moment_x,
            self.applied_loc,
            self.upper_bnds, 
            self.lower_bnds,
            **self.kwargs_intern,
        )

        if GA_res.X is None:
            warnings.warn(
                f"Internal optimization for stringers {x} was not successful, no solution found "
            )

        # Discretize airfoil from new given parameters
        wingbox_obj = discretize_airfoil(self.path_coord, self.chord, copy_struct)

        out["F"] = wingbox_obj.get_total_area()


class ProblemFixedPanel(ElementwiseProblem):
    """
    The following problem sets up an exploration of a wingbox for a fixed amount of stringers per cell. Since it is difficult for an optimizer
    like COBYLA to find a global minimum, a genetic algorithm is set up first to explore the design space. This problem sets up the problem as 
    defined in (`PYMOO <https://pymoo.org/problems/definition.html#ElementwiseProblem-(loop)>`_).

    .. warning::

        This optimization problem is **not** compatible with stringer area because of the constraints requiring stringer geometry.

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
    box_struct : Wingbox
        Wingbox structure, which includes details about geometry, materials, and discretization.
    mat_struct : Material
        Material properties used in the wingbox.
    path_coord : str
        File path or identifier for coordinates related to the section.
    upper_bnds : list, optional
        A list containing the upper bounds of the optimization.
    lower_bnds : list, optional
        A list containing the lower bounds of the optimization.

    Attributes
    ----------
    shear_y : float
        Vertical shear force applied to the section.
    shear_x : float
        Horizontal shear force applied to the section.
    moment_y : float
        Bending moment around the Y-axis applied to the section.
    moment_x : float
        Bending moment around the X-axis applied to the section.
    applied_loc : float
        The location along the chord where the load is applied.
    chord : float
        The length of the wing chord at the section where loads are applied.
    len_sec : float
        The length of the wing section being analyzed in the spanwise direction, i.e., the distance to the next rib.
    box_struct : Wingbox
        Data structure describing the structural configuration of the wingbox at the section.
    mat_struct : Material
        The material properties used in the wing section.
    path_coord : str
        File path or identifier for coordinates related to the section.
    wingbox_obj : object
        The object used to avoid repetitive computations.
    """
   

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
        **kwargs,
    ):
        """
        Initializes the class with the correct data.

        Parameters
        ----------
        shear_y : float
            Vertical shear force applied to the section.
        shear_x : float
            Horizontal shear force applied to the section.
        moment_y : float
            Bending moment around the Y-axis applied to the section.
        moment_x : float
            Bending moment around the X-axis applied to the section.
        applied_loc : float
            The location along the chord where the load is applied.
        chord : float
            The length of the wing chord at the section where loads are applied.
        len_sec : float
            The length of the wing section being analyzed in the spanwise direction, i.e., the distance to the next rib.
        upper_bnds: list
            A list of N + 4 long containing the upper bounds, where N is the amount of cells. The bounds are presented in the following order t_sk_cell, t_sp, t_st, w_st, h_st. 
        lower_bnds: list
            A list of N + 4 long containing the lower bounds, where N is the amount of cells. The bounds are presented in the following order t_sk_cell, t_sp, t_st, w_st, h_st. 
        box_struct : Wingbox
            Data structure describing the structural configuration of the wingbox at the section.
        mat_struct : Material
            The material properties used in the wing section.
        path_coord : str
            File path or identifier for coordinates related to the section.
        upper_bnds : list
            Upper bounds for optimization or design constraints, defaults to 0.01 for all parameters.
        lower_bnds : list
            Lower bounds for optimization or design constraints, defaults to 1e-8 for all parameters.

        Raises
        ------
        RuntimeError
            If the optimization is not compatible with stringer area as the constraints rely on the string geometry.
        """  

        if box_struct.area_str is not None:
            raise RuntimeError(
                "The optimization is not compatible with stringer area as the constraints rely on the string geometry"
            )

        self.shear_y = shear_y
        self.shear_x = shear_x
        self.moment_y = moment_y
        self.moment_x = moment_x
        self.applied_loc = applied_loc
        self.chord = chord
        self.len_sec = len_sec
        self.box_struct = box_struct
        self.mat_struct = mat_struct
        self.path_coord = path_coord

        # Stops discretize airfoil from complaining (just for getting the right amount of constraints)
        self.box_struct.area_str = 1e-5
        self.box_struct.t_sk_cell = 0.001 * np.ones(self.box_struct.n_cell)
        self.box_struct.t_sp = 0.001
        # The following object is used to avoid repetitive computations
        self.wingbox_obj = discretize_airfoil(
            self.path_coord, self.chord, self.box_struct
        )

        # Remove from data struct again to stop from interferitg
        self.box_struct.area_str = None
        self.box_struct.t_sk_cell = None
        self.box_struct.t_sp = None

        # The number of constraint depends on the amount of booms + 2 constraints for stringer geometry
        super().__init__(
            n_var=box_struct.n_cell + 4,
            n_obj=1,
            n_ieq_constr=7 * len(self.wingbox_obj.panel_dict) + 2,
            xl=lower_bnds,
            xu=upper_bnds,
            **kwargs,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        The function is executed at each element in the optimization. The arguments that are passed should be in the following 
        order: t_sk_cell, t_sp, t_st, w_st, h_st. Together, they give a total length of N + 4, where N is the number of cells.
        This is internally checked, and if it is not the case, a runtime error will be raised.

        Parameters
        ----------
        x : list
            The design variable vector which is in the following format [t_sk_cell, t_sp, t_st, w_st, h_st].
        shear : float
            The shear force for which we're optimizing.
        moment : float
            The moment for which we're optimizing.
        applied_loc : float
            The applied location of the shear force on the wingbox.
        str_lst : list
            A list with the amount of stringers per cell.

        Raises
        ------
        RuntimeError
            If the flattened design vector does not match the specified wingbox properties.
        """


        n_cell = self.box_struct.n_cell

        # check whether we are using multiprocess
        if StarmapParallelization is not None and isinstance(getattr(self, "elementwise_runner", None), StarmapParallelization):
            # If so then make a copy of wingbox object
            box_copy = deepcopy(
                self.wingbox_obj
            )  # Create copy (required for multiprocessing)
        else:
            # If not then simpy make a reference
            box_copy = self.wingbox_obj

        if len(x) != n_cell + 4:
            raise RuntimeError(
                "The flattened  design vector received does not match the wingbox properties"
            )

        # Assing new properties to the wingbox struct assuming the specified length
        t_sk_cell = x[:n_cell]
        t_sp = x[n_cell]
        t_st = x[n_cell + 1]
        w_st = x[n_cell + 2]
        h_st = x[n_cell + 3]

        box_copy.load_new_gauge(t_sk_cell, t_sp, t_st, w_st, h_st)

        # Perform stress analysis
        box_copy.stress_analysis(
            self.shear_y,
            self.shear_x,
            self.moment_y,
            self.moment_x,
            self.applied_loc,
            self.mat_struct.shear_modulus,
        )

        # ================ Get constraints ====================================== 
        constr_cls = IsotropicWingboxConstraints(
            box_copy, self.mat_struct, self.len_sec
        )
        # Add all constraints from a structural perspective
        struct_constr = np.negative(
            np.concatenate(
                (
                    constr_cls.global_skin_buckling(),
                    constr_cls.interaction_curve(),
                    constr_cls.von_Mises(),
                    constr_cls.column_str_buckling(),
                    constr_cls.stringer_flange_buckling(),
                    constr_cls.stringer_web_buckling(),
                    constr_cls.crippling(),
                )
            )
        )  # negative is necessary because pymoo handles inequality constraints differently
        # Add constraints requried from a geometrical perspective
        geom_constr = [t_st - w_st, t_st - h_st]
        out["F"] = box_copy.get_total_area()
        out["G"] = np.concatenate((struct_constr, geom_constr))


class SectionOpt:
    """
    The following class allows for the optimization of a single airfoil section. The class assumes that the correct material properties
    have been embedded in :class:`Material`, and an initial estimate should be embedded in the :class:`Wingbox` class.

    Parameters
    ----------
    path_coord : str
        The path to the coordinate file of the airfoil that is used.
    chord : float
        The chord size.
    len_sec : float
        The length of the section, that is the length to the next rib.
    wingbox_struct : Wingbox
        The wingbox structure.
    material_struct : Material
        The material data structure.

    Attributes
    ----------
    path_coord : str
        The path to the coordinate file of the airfoil that is used.
    chord : float
        The chord size.
    len_sec : float
        The length of the section, that is the length to the next rib.
    box_struct : Wingbox
        The wingbox data structure.
    mat_struct : Material
        The material data structure.
    _wingbox_obj : IdealWingbox or None
        The wingbox object used for computations.
    """


    def __init__(self,
        path_coord: str,
        chord: float,
        len_sec: float,
        wingbox_struct: Wingbox,
        material_struct: Material,
    ) -> None:
        """
        Constructor method.

        Parameters
        ----------
        path_coord : str
            The path to the coordinate file of the airfoil that is used.
        chord : float
            The chord size.
        len_sec : float
            The length of the section, that is the length to the next rib.
        wingbox_struct : Wingbox
            The wingbox data structure.
        material_struct : Material
            The material data structure.
        """ 

        # Perform checks on whether correct data was loaded in

        self.path_coord = path_coord
        self.chord = chord
        self.len_sec = len_sec
        self.box_struct: Wingbox = wingbox_struct
        self.mat_struct: Material = material_struct

        # Required Overhead
        self._wingbox_obj: None | IdealWingbox = None

    def GA_optimize(
        self,
        shear_y: float,
        shear_x: float,
        moment_y: float,
        moment_x: float,
        applied_loc: float,
        upper_bnds: list,
        lower_bnds: list, 
        n_gen: int = 50,  # Possible keywords
        pop: int = 100,
        verbose: bool = True,
        seed: int = 1,
        multiprocess: bool = False,
        cores: int = multiprocessing.cpu_count(),
        save_hist: bool = True,
    ):
        """
        The following function executes the Genetic Algorithm (`GA <https://pymoo.org/algorithms/soo/ga.html>`_) to optimize the wingbox given to the overarching class
        with the loads fed to the function. For more information on the loads specification please see :meth:`stress_analysis <tuduam.structures.wingbox.IdealWingbox.stress_analysis>`. Additionally, t[...] 

        Parameters
        ----------
        shear : float
            The internal shear force acting at the section.
        moment : float
            The internal moment acting at the section.
        applied_loc : float
            The position of the internal shear force given as a ratio to the chord.
        upper_bnds: list
            A list of N + 4 long containing the upper bounds, where N is the amount of cells. The bounds are presented in the following order t_sk_cell, t_sp, t_st, w_st, h_st. 
        lower_bnds: list
            A list of N + 4 long containing the lower bounds, where N is the amount of cells. The bounds are presented in the following order t_sk_cell, t_sp, t_st, w_st, h_st. 
        n_gen : int, optional
            The number of generations allowed before termination, defaults to 50.
        pop : int, optional
            The generation size, defaults to 50.
        verbose : bool, optional
            If true, will print the results of all generations, defaults to True.
        seed : int, optional
            The seed for the random generations of the samples, defaults to 1.
        cores : int, optional
            The number of cores used for the parallelization of the evaluations, defaults to multiprocessing.cpu_count().
        save_hist : bool, optional
            Saves the history in the result object if true, defaults to True.

        Returns
        -------
        pymoo.core.result.Result
            Returns the result class from pymoo, which will also contain the history if specified true. Please see the example for use cases (`example <https://pymoo.org/getting_started/part_4.html>`_[...]
        """  

        # initialize the thread pool and create the runner
        if multiprocess and StarmapParallelization is not None:
            n_proccess = cores
            pool = multiprocessing.Pool(n_proccess)
            runner = StarmapParallelization(pool.starmap)

            problem = ProblemFixedPanel(
                shear_y,
                shear_x,
                moment_y,
                moment_x,
                applied_loc,
                self.chord,
                self.len_sec,
                upper_bnds,
                lower_bnds,
                self.box_struct,
                self.mat_struct,
                self.path_coord,
                elementwise_runner=runner,
            )
        else:
            if multiprocess and StarmapParallelization is None:
                warnings.warn(
                    "pymoo StarmapParallelization not available in this pymoo version; running without multiprocessing."
                )
            problem = ProblemFixedPanel(
                shear_y,
                shear_x,
                moment_y,
                moment_x,
                applied_loc,
                self.chord,
                self.len_sec,
                upper_bnds,
                lower_bnds,
                self.box_struct,
                self.mat_struct,
                self.path_coord,
            )

        method = GA(pop_size=pop, eliminate_duplicates=True)
        resGA = minimize(
            problem,
            method,
            termination=("n_gen", n_gen),
            seed=seed,
            save_history=save_hist,
            verbose=verbose,
        )
        return resGA

    def _obj_func_cobyla(
        self,
        x: list,
        shear_y: float,
        shear_x: float,
        moment_y: float,
        moment_x: float,
        applied_loc: float,
        str_lst: list,
    ):
        """
        This function is not intended to be used by the user; hence it is hinted to be a private method. The function is passed to the scipy.optimize.minimize function. The arguments that are passed s[...] 
        order: t_sk_cell, t_sp, t_st, w_st, h_st. Together, they give a total length of N + 4, where N is the number of cells.
        This is internally checked, and if it is not the case, a runtime error will be raised.

        Parameters
        ----------
        x : list
            The design variable vector which is in the following format [t_sk_cell, t_sp, t_st, w_st, h_st].
        shear : float
            The shear force for which we're optimizing.
        moment : float
            The moment for which we're optimizing.
        applied_loc : float
            The applied location of the shear force on the wingbox.
        str_lst : list
            A list with the amount of stringers per cell.

        Raises
        ------
        RuntimeError
            If the flattened design vector does not match the specified wingbox properties.
        """


        n_cell = self.box_struct.n_cell

        if len(x) != n_cell + 4:
            raise RuntimeError(
                "The flattened  design vector received does not match the wingbox properties"
            )

        # Assing new properties to the wingbox struct assuming the specified length
        t_sk_cell = x[:n_cell]
        t_sp = x[n_cell]
        t_st = x[n_cell + 1]
        w_st = x[n_cell + 2]
        h_st = x[n_cell + 3]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.t_st = t_st
        self.box_struct.w_st = w_st
        self.box_struct.h_st = h_st
        self.box_struct.str_cell = str_lst

        # Discretize airfoil from new given parameters
        self._wingbox_obj.load_new_gauge(t_sk_cell, t_sp, t_st, w_st, h_st)

        # Perform stress analysis
        self._wingbox_obj.stress_analysis(
            shear_y,
            shear_x,
            moment_y,
            moment_x,
            applied_loc,
            self.mat_struct.shear_modulus,
        )

        return self._wingbox_obj.get_total_area()

    def optimize_cobyla(
        self,
        shear_y: float,
        shear_x: float,
        moment_y: float,
        moment_x: float,
        applied_loc: float,
        str_lst: list,
        bnd_mult: int = 1e3,
        x0: list | None = None,
    ) -> sop._optimize.OptimizeResult:
        """
        Optimizes the design using the COBYLA optimizer with the constraints defined in :class:`IsotropicWingboxConstraints`. The optimization parameters
        are the skin thickness in each cell, the spar thickness, and the area of the stringers. Hence the resulting design vector is x = []. The amount of stringers is not an optimization parameter he[...]
        as this would result in a varying amount of constraints which is not supported by COBYLA. Hence, the result of this will be fed to a different 
        optimizer.

        Parameters
        ----------
        shear : float
            The shear force for which we're optimizing.
        moment : float
            The moment for which we're optimizing.
        applied_loc : float
            The applied location of the shear force on the wingbox.
        str_list : list
            List of the amount of stringers per cell.
        bnd_mult : int, optional
            A multiplier to increase the enforcement of the bounds on the variables, defaults to 1000.
        x0 : list, optional
            A more explicit way to define the initial estimate instead of taking it from the wingbox structure, defaults to None.

        Returns
        -------
        scipy.optimize.OptimizeResult
            The result of the optimization process.
        """  

        n = self.box_struct.n_cell  # quick reference to number of cells
        # Whatever parameters were given in the datastrucrtre are used as inital estimate
        if x0 is None:
            x0 = (
                self.box_struct.t_sk_cell
                + [self.box_struct.t_sp]
                + [self.box_struct.t_st]
                + [self.box_struct.w_st]
                + [self.box_struct.h_st]
            )

        constr_lst: List[dict] = [
            {
                "type": "ineq",
                "fun": self._get_constraint_vector,
                "args": [shear_y, shear_x, moment_y, moment_x, applied_loc],
            },
        ]

        lb_lst = []
        ub_lst = []

        for i in range(len(x0)):
            if i <= n - 1:
                lb_lst.append(1e-5)
                ub_lst.append(0.2)
            elif i == n:
                lb_lst.append(1e-5)
                ub_lst.append(0.3)
            elif i >= n + 1:
                lb_lst.append(1e-8)
                ub_lst.append(0.2)

        # Apply bounds through constraints manually so we can penalize them with a greater quantity
        # Otherwise they migt be ignored compared to other constraints
        for i in range(len(lb_lst)):

            def upper_constraint(x, idx):
                return 1e3 * (ub_lst[idx] - x[idx])

            def lower_constraint(x, idx):
                return 1e3 * (x[idx] - lb_lst[idx])

            constr_lst.append({"type": "ineq", "args": [i], "fun": upper_constraint})
            constr_lst.append({"type": "ineq", "args": [i], "fun": lower_constraint})

        self._wingbox_obj = discretize_airfoil(
            self.path_coord, self.chord, self.box_struct
        )
        res = sop.minimize(
            self._obj_func_cobyla,
            x0,
            args=(shear_y, shear_x, moment_y, moment_x, applied_loc, str_lst),
            method="COBYLA",
            constraints=constr_lst,
        )
        return res

    def full_section_opt(
        self,
        shear_y: float,
        shear_x: float,
        moment_x: float,
        moment_y: float,
        applied_loc: float,
        n_gen_full: int = 50,  # Possible keywords
        pop_full: int = 100,
        verbose_full: bool = True,
        cores: int = multiprocessing.cpu_count(),
        seed: int = 1,
        multiprocess_full: bool = True,
        save_hist_full: bool = True,
        **kwargs,
    ):
        """
        The following function handles the full optimization of the wingbox. Compared to the :func:`GA_optimize`, this function also varies
        the amount of stringers per cell to find the optimum design. In order to do this, it utilizes all previously defined optimizations in combination 
        with a discrete Genetic Algorithm variable optimization.

        Parameters
        ----------
        shear : float
            The shear force for which we're optimizing.
        moment : float
            The moment for which we're optimizing.
        applied_loc : float
            The applied location of the shear force on the wingbox.
        n_gen_full : int, optional
            The number of generations for the full optimization, defaults to 50.
        verbose_full : bool, optional
            If true, will print the results of all generations for the full optimization, defaults to True.
        seed : int, optional
            The seed for the random generations of the samples, defaults to 1.
        save_hist_full : bool, optional
            Saves the history in the result object if true, defaults to True.
        """  

        pool = None
        runner = None
        if multiprocess_full and StarmapParallelization is not None:
            n_proccess = cores
            pool = multiprocessing.Pool(n_proccess)
            runner = StarmapParallelization(pool.starmap)

        if multiprocess_full and StarmapParallelization is None:
            warnings.warn(
                "pymoo StarmapParallelization not available in this pymoo version; running without multiprocessing."
            )

        if runner is not None:
            prob = ProblemFreePanel(
                shear_y,
                shear_x,
                moment_y,
                moment_x,
                applied_loc,
                self.chord,
                self.len_sec,
                self.box_struct,
                self.mat_struct,
                self.path_coord,
                elementwise_runner=runner,
                **kwargs,
            )
        else:
            prob = ProblemFreePanel(
                shear_y,
                shear_x,
                moment_y,
                moment_x,
                applied_loc,
                self.chord,
                self.len_sec,
                self.box_struct,
                self.mat_struct,
                self.path_coord,
                **kwargs,
            )

        method = GA(
            pop_size=pop_full,
            sampling=IntegerRandomSampling(),
            crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
            eliminate_duplicates=True,
        )

        res = minimize(
            prob,
            method,
            termination=("n_gen", n_gen_full),
            seed=seed,
            save_history=save_hist_full,
            verbose=verbose_full,
        )
        # best effort: if we created a pool, close it
        if pool is not None:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass
        return res

    def _get_constraint_vector(
        self,
        x: list,
        shear_y: float,
        shear_x: float,
        moment_y: float,
        moment_x: float,
        applied_loc: float,
    ) -> list:

        r"""
        The following function utilizes all other constraints and wraps their results into one vector, where each 
        element represents one of the inequality constraints. The reason we utilize this method instead of just passing each function as their own constraint
        has to do with the overhead that would be incurred. Since scipy.optimize.minimize can only take a flattened array, I cannot pass the actual discretized
        airfoil around. Hence, if I want to get the discretized airfoil from the flattened array I have to run the :func:`discretize_airfoil` again. To avoid
        doing this multiple times, all constraints are wrapped into one function.

        Parameters
        ----------
        x : list
            The flattened array received from the scipy.optimize.minimize function.
        wingbox_struct : Wingbox
            The wingbox struct from the data structures module.
        shear : float
            The applied shear force on the wingbox.
        moment : float
            The applied moment on the wingbox.
        applied_loc : float
            The applied location of the shear force on the wingbox.

        Returns
        -------
        list
            All constraints appended to each other into one list of constraints.
        """

        n_cell = self.box_struct.n_cell
        t_sk_cell = x[:n_cell]
        t_sp = x[n_cell]
        t_st = x[n_cell + 1]
        w_st = x[n_cell + 2]
        h_st = x[n_cell + 3]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.t_st = t_st
        self.box_struct.w_st = w_st
        self.box_struct.h_st = h_st

        ideal_wingbox: IdealWingbox = discretize_airfoil(
            self.path_coord, self.chord, self.box_struct
        )
        ideal_wingbox.stress_analysis(
            shear_y,
            shear_x,
            moment_y,
            moment_x,
            applied_loc,
            self.mat_struct.shear_modulus,
        )
        constr_cls = IsotropicWingboxConstraints(
            ideal_wingbox, self.mat_struct, self.len_sec
        )
        return np.concatenate((constr_cls.interaction_curve(), constr_cls.von_Mises()))
