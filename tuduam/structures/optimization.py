import multiprocessing

import numpy as np
import scipy.optimize as sop
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.problem import StarmapParallelization
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling


from ..data_structures import *
from .wingbox import *
from .constraints import IsotropicWingboxConstraints

class ProblemFreePanel(ElementwiseProblem):
    """ The following problem sets up Discrete variable problem which optimises the amount of stringers per cell. This was done 
    in a separate problem as changing the amount of stringers changes the amount of panels and hence the amount of constraints. Most optimizers
    do not allow this, hence the following problem uses the output of the :class:`WingboxFixedPanel` to abide by the constraints.
    """    

    def __init__(self, shear_y: float, shear_x: float, moment_y: float, moment_x: float, applied_loc: float, chord: float, len_sec: float, box_struct: Wingbox, mat_struct: Material,
                 path_coord: str, **kwargs_intern ):

        self.box_params: list =  list()
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
        self.kwargs_intern = kwargs_intern #  Used to pass any settings to GA_optimize functions

        super().__init__(n_var= box_struct.n_cell, n_obj=1, n_ieq_constr=0, xl=5, xu=40, vtype= int, **kwargs_intern)

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        """ This function get evaluated for each 'element', where an element in this case is a single sequence of n_var long. For more info
        please see the documentation of the pymoo library.

        :param x: A list of n_var long containing the amount of stringers per cell
        :type x: np.ndarray
        :param out: A dictionary containing the constraitns and objective
        :type out: dict
        """        

        
        n_cell = self.box_struct.n_cell
        copy_struct = self.box_struct.model_copy()
        copy_struct.str_cell = x

        if len(x) != n_cell:
            raise RuntimeError("The flattened  design vector received does not match the wingbox properties")
        
        sec_opt =  SectionOptimization(self.path_coord, self.chord, self.len_sec, copy_struct, self.mat_struct)

        # Elementwise runner is removed otherwise it clashes in the GA_optimize function
        try:
            self.kwargs_intern.pop("elementwise_runner")
        except KeyError:
            pass

        GA_res = sec_opt.GA_optimize(self.shear_y, self.shear_x, self.moment_y, self.moment_x, self.applied_loc, **self.kwargs_intern)

        if GA_res.X is None:
            raise ValueError(f"Internal optimization for stringers {x} was not successful and the outer optimization could not continue")

        # Discretize airfoil from new given parameters
        wingbox_obj = discretize_airfoil(self.path_coord, self.chord, copy_struct)

        out["F"] = wingbox_obj.get_total_area() 

class ProblemFixedPanel(ElementwiseProblem):
    """
        The followinng problem sets up an exploration of a wingbox for a fixed amount of stringers per cell. Since it is difficult for an optimizer
        like COBYLA to find a global minimum a genetica algorith is set up first to explore the design space. This problem sets up this search.

        ---------------------------
        Important notes
        ---------------------------
        #. This optimization is *not* compatible with stringer area because of the constraints requiring stringer geometry
    """    

    def __init__(self, shear_y: float, shear_x: float, moment_y: float, moment_x: float, applied_loc: float, chord: float, len_sec: float,
                 box_struct: Wingbox, mat_struct: Material, path_coord: str, **kwargs):

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

        self.box_struct.area_str = 1e-5 # Stops discretize airfoil from complaining
        # sample to get the right number of constraints
        sample = discretize_airfoil(self.path_coord, self.chord, self.box_struct) 
        self.box_struct.area_str = None # Remove from data struct again to stop from interferitg

        super().__init__(n_var= box_struct.n_cell + 4, n_obj=1, n_ieq_constr= 2*len(sample.panel_dict), xl=np.ones(self.box_struct.n_cell + 4)*1e-8, xu=np.ones(self.box_struct.n_cell + 4), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        The function is excecuted at each element in the optimization. The arguments that are passed should be in the following 
        order t_sk_cell, t_sp, t_st, w_st, h_st . Together given a total length of N + 2 where N is the amount of cells.
        This interally checked if it is not the case a runtime error will be raised

        :param x: The design variable vector which is in the following format [t_sk_cell, t_sp, t_st, w_st, h_st]
        :type x: list
        :param shear: The shear force for which we're optimizing
        :type shear: float
        :param moment: The moment for which we're optimizing
        :type moment: float
        :param applied_loc: The applied location of the shear force on the wingbox.
        :type applied_loc: float
        :param str_lst: A list with the amount of stringers per cell.
        :type str_lst: list
        :raises RuntimeError: when the flattened design vector does not match the specified wingbox properties.
        """        
        
        n_cell = self.box_struct.n_cell

        if len(x) != n_cell + 4:
            raise RuntimeError("The flattened  design vector received does not match the wingbox properties")
        
        
        # Assing new properties to the wingbox struct assuming the specified length
        t_sk_cell =  x[:n_cell]
        t_sp = x[n_cell]
        t_st = x[n_cell + 1]
        w_st = x[n_cell + 2]
        h_st = x[n_cell + 3]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.t_st = t_st
        self.box_struct.w_st = w_st
        self.box_struct.h_st = h_st

        # Discretize airfoil from new given parameters
        wingbox_obj = discretize_airfoil(self.path_coord, self.chord, self.box_struct)

        # Perform stress analysis
        wingbox_obj.stress_analysis(self.shear_y,self.shear_x, self.moment_y, self.moment_x, self.applied_loc, self.mat_struct.shear_modulus)

        #================ Get constraints ======================================
        constr_cls = IsotropicWingboxConstraints(wingbox_obj, self.mat_struct, self.len_sec)
        out["F"] = wingbox_obj.get_total_area() 
        out["G"] = np.negative(np.concatenate((constr_cls.interaction_curve(), constr_cls.von_Mises()))) # negative is necessary because pymoo handles inequality constraints differently

class SectionOptimization:
    """
    The following class allows for the optimization of a single airfoil section. The clas assumes that the correct material properties
     have been embedded in :class:`Material` has been chosen, an initial estimate should be embedded in the :class:`Wingbox` class.

    **Attributes**

    :param path_coord: The path to the coordinate file of the airfoil that is used
    :type path_coord: string
    :param chord: The chord size
    :type chord: float
    :param len_sec: The length of the section, that is the length to the next rib
    :type len_sec: float
    :param box_struct: The wingbox structure
    :type box_struct: wingbox
    :param mat_struct: The Material  data structure
    :type box_struct: Material
    """    
    def __init__(self, path_coord: str, chord: float, len_sec: float, wingbox_struct: Wingbox, material_struct: Material) -> None:
        """Constructor method

        :param path_coord: The path to the coordinate file of the airfoil that is used
        :type path_coord: str
        :param chord: The chord size
        :type chord: float
        :param len_sec: The length of the section, that is the lengt to the next rib
        :type len_sec: float
        :param wingbox_struct: THe wingbox datastructure
        :type wingbox_struct: Wingbox
        :param material_struct: _description_
        :type material_struct: Material
        """        

        # Perform checks on whether correct data was loaded in

        self.path_coord = path_coord
        self.chord = chord
        self.len_sec =  len_sec
        self.box_struct: Wingbox = wingbox_struct
        self.mat_struct: Material  = material_struct

        # Required Overhead
        self.wingbox_obj: None | IdealWingbox = None

    def GA_optimize(self, shear_y: float, shear_x: float, moment_y: float, moment_x: float, applied_loc: float,
                     n_gen: int = 50, # Possible keywords
                     pop: int = 100,
                     verbose: bool = True,
                     seed: int = 1,
                     multiprocess: bool =  False,
                     cores: int = multiprocessing.cpu_count(),
                     save_hist: bool = True):
        
        """ The following function executes the Genetic Algorithm (`GA <https://pymoo.org/algorithms/soo/ga.html>`_) to optimize the wingbox given to the overarching class
        and with the loads fed to the function. For more information on the loads specification please see :meth:`stress_analysis <tuduam.structures.wingbox.IdealWingbox.stress_analysis>`. Additionally there are some keywords which are explained below.

        :param shear: The internal shear force acting at the section
        :type shear: float
        :param moment: The internal moment acting at the section
        :type moment: float
        :param applied_loc: The position of the internal shear force given as a ratio to the chord.
        :type applied_loc: float
        :param n_gen: The amount of generations allowed before termination, defaults to 50
        :type n_gen: int, optional
        :param pop: The generation size, defaults to 50
        :type pop: int, optional
        :param verbose: If true will print the results of all generations, defaults to True
        :type verbose: bool, optional
        :param seed: The seed for the random generations of the samples, defaults to 1
        :type seed: int, optional
        :param cores: The amount of cores used for the parallelization of the evaluations , defaults to multiprocesing.cpu_count()
        :type cores: int, optional 
        :param save_hist: Saves the history in the result object if true, defaults to True
        :type save_hist: bool, optional
        :return: Returns the result class from pymoo, which will also contain the history if specified true. Please see the example for use cases (`example <https://pymoo.org/getting_started/part_4.html>`_) .
        :rtype: pymoo.core.result.Result
        """        

        # initialize the thread pool and create the runner
        if multiprocess:
            n_proccess =  cores
            pool = multiprocessing.Pool(n_proccess)
            runner = StarmapParallelization(pool.starmap)

            problem = ProblemFixedPanel(shear_y, shear_x, moment_y, moment_x, applied_loc, self.chord,  self.len_sec, 
                                        self.box_struct, self.mat_struct, self.path_coord, elementwise_runner=runner)
        else:
            problem = ProblemFixedPanel(shear_y, shear_x, moment_y, moment_x, applied_loc, self.chord,  self.len_sec, 
                                        self.box_struct, self.mat_struct, self.path_coord)

        method = GA(pop_size=pop, eliminate_duplicates=True)
        resGA = minimize(problem, method, termination=('n_gen', n_gen), seed= seed,
                        save_history=save_hist, verbose=verbose)
        return resGA

    def _obj_func_cobyla(self, x: list, shear_y: float, shear_x: float, moment_y: float,  moment_x: float, applied_loc: float, str_lst: list):
        """
        This function is not to be intended by the user hence it is hinted to be a private method. The function passed to the scip.optimize.minmize function. The arguments that are passed should be in the following 
        order t_sk_cell, t_sp, area_str. Together given a total length of N + 2 where N is the amount of cells.
        This interally checked if it is not the case a runtime error will be raised

        :param x: The design variable vector which is in the following format [t_sk_cell, t_sp, t_st, w_st, h_st ]
        :type x: list
        :param shear: The shear force for which we're optimizing
        :type shear: float
        :param moment: The moment for which we're optimizing
        :type moment: float
        :param applied_loc: The applied location of the shear force on the wingbox.
        :type applied_loc: float
        :param str_lst: A list with the amount of stringers per cell.
        :type str_lst: list
        :raises RuntimeError: when the flattened design vector does not match the specified wingbox properties.
        """        
        
        n_cell = self.box_struct.n_cell

        if len(x) != n_cell + 4:
            raise RuntimeError("The flattened  design vector received does not match the wingbox properties")
        
        
        # Assing new properties to the wingbox struct assuming the specified length
        t_sk_cell =  x[:n_cell]
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
        self.wingbox_obj = discretize_airfoil(self.path_coord, self.chord, self.box_struct)

        # Perform stress analysis
        self.wingbox_obj.stress_analysis(shear_y, shear_x, moment_y, moment_x, applied_loc, self.mat_struct.shear_modulus)

        return self.wingbox_obj.get_total_area() 


    def optimize_cobyla(self, shear_y: float, shear_x: float, moment_y: float, moment_x: float, applied_loc: float, str_lst: list,
                        bnd_mult: int = 1e3, 
                        x0: list | None = None) -> sop._optimize.OptimizeResult:

        """ Optimizes the design using the COBYLA optimizers with the constraints defined in :class:` IsotropicWingboxConstraints`.  The optimization parameters
        are the skin thickness in each cell, the spar thickness and the area of the stringers. Hence the resulting design vector is x = [] The amount of stringers is not a optimization parameter here
        as this would results in a varying amount of constraints which is not supported by COBYLA. Hence, the result of this will be fed to a different 
        optimizer.

        :param shear: _description_
        :type shear: float
        :param moment: _description_
        :type moment: float
        :param applied_loc: _description_
        :type applied_loc: float
        :param str_list: List of the amount of stringers per cell
        :type str_list: list
        :param bnd_mult: A multiplier to increase the enforcement of the boudns on the variables, default to 1000
        :type bnd_mult: int, optional
        :param x0: A more explicit way to define the intial estimate instead of taking it from the wingbox struct, default to None
        :type x0: list, optional
        :return: _description_
        :rtype: sop._optimize.OptimizeResult
        """        

        n = self.box_struct.n_cell # quick reference to number of cells
        # Whatever parameters were given in the datastrucrtre are used as inital estimate
        if x0 is None:
            x0 = self.box_struct.t_sk_cell + [self.box_struct.t_sp] + [self.box_struct.t_st] + [self.box_struct.w_st] + [self.box_struct.h_st]  
        else:
            pass

        constr_lst: List[dict] = [
            {'type': 'ineq', 'fun': self._get_constraint_vector, "args": [shear_y, shear_x, moment_y, moment_x, applied_loc]},
                ]

        lb_lst = []
        ub_lst = []

        for i in range(len(x0)):
            if i <= n-1:
                lb_lst.append(1e-5)
                ub_lst.append(0.3)
            elif i == n :
                lb_lst.append(1e-5)
                ub_lst.append(0.5)
            elif i  ==  n + 1: 
                lb_lst.append(1e-8)
                ub_lst.append(1)
        
        # Apply bounds through constraints manually so we can penalize them with a greater quantity
        # Otherwise they migt be ignored compared to other constraints
        for i in range(len(lb_lst)):
            def upper_constraint(x, idx):
                return 1e3*(ub_lst[idx] - x[idx])

            def lower_constraint(x, idx):
                return 1e3*(x[idx] - lb_lst[idx])

            constr_lst.append({'type':'ineq', 'args': [i], 'fun':upper_constraint})
            constr_lst.append({'type':'ineq', 'args':[i], 'fun':lower_constraint})

        res = sop.minimize(self._obj_func_cobyla, x0, args=(shear_y, shear_x, moment_y, moment_x, applied_loc, str_lst), method="COBYLA" , constraints= constr_lst)
        return res
    
    def full_section_opt(self, shear_y: float, shear_x: float, moment_x: float, moment_y: float, applied_loc: float,
                     n_gen_full: int = 50, # Possible keywords
                     pop_full: int = 100,
                     verbose_full: bool = True,
                     cores: int = multiprocessing.cpu_count(),
                     seed: int = 1,
                     save_hist_full: bool = True, **kwargs):
        """ 
        The following functions handles the full optimization of the wingbox. Compared to the func:`GA_optimize` the following functions also varies
        the amount of stringers per cell to find the optimum design. In order to do this it utlizes all previously defined optimization in combination 
        with a discrete Geentic algoirth variabled optimization. 

        :param shear: _description_
        :type shear: float
        :param moment: _description_
        :type moment: float
        :param applied_loc: _description_
        :type applied_loc: float
        :param n_gen_full: _description_, defaults to 50
        :type n_gen_full: int, optional
        :param verbose_full: _description_, defaults to True
        :type verbose_full: bool, optional
        :param seed: _description_, defaults to 1
        :type seed: int, optional
        :param save_hist_full: _description_, defaults to True
        :type save_hist_full: bool, optional
        """        

        n_proccess =  cores
        pool = multiprocessing.Pool(n_proccess)
        runner = StarmapParallelization(pool.starmap)
        prob = ProblemFreePanel(shear_y, shear_x, moment_y, moment_x, applied_loc, self.chord, self.len_sec, self.box_struct, self.mat_struct, 
                                self.path_coord, elementwise_runner=runner, **kwargs)
        

        method = GA(pop_size=pop_full,
                    sampling=IntegerRandomSampling(),
                    crossover=SBX(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                    mutation=PM(prob=1.0, eta=3.0, vtype=float, repair=RoundingRepair()),
                    eliminate_duplicates=True,
                    )

        res = minimize(prob,
                    method,
                    termination=('n_gen', n_gen_full),
                    seed=seed,
                    save_history= save_hist_full,
                    verbose= verbose_full
                    )
        return res





    def _get_constraint_vector(self, x: list, shear_y: float, shear_x: float, moment_y: float, moment_x: float, applied_loc: float) -> list:
        r""" 
        The following function utilizes all other constraints and wraps their results into one vector. Where each 
        element represents one  of the inequality constraints. The reasons we utilize this method instead of just passing each function as their own constraint
        has to do with the overhead that would be incurred. Since scipy.optimize.minimize can only take a flattened array, I can not pass the actual discretized
        airfoil around. Hence, if I want to get the discretize airfoil from the flattened array I have to run the  :func:`discretize_airfoil` again.  To avoid
        doing this a multitude times all constraints are wrapped into one function.

        :param x: The flattened array received from the scipy.optimize.minimize function.
        :type x: list
        :param wingbox_struct: The wingbox struct from the  data structures module        
        :param shear: The applied shear force on the wingbox.
        :type shear: float
        :param moment: The applied moment on the wingbox
        :type moment: float
        :param applied_loc: The applied location of the shear force on the wingbox
        :type applied_loc: float
        :return: All constraints appended to each other into one constraints
        :rtype: list
        """ 

        n_cell = self.box_struct.n_cell
        t_sk_cell =  x[:n_cell]
        t_sp = x[n_cell]
        t_st = x[n_cell + 1]
        w_st = x[n_cell + 2]
        h_st = x[n_cell + 3]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.t_st = t_st
        self.box_struct.w_st = w_st
        self.box_struct.h_st = h_st 

        ideal_wingbox: IdealWingbox = discretize_airfoil(self.path_coord, self.chord, self.box_struct)
        ideal_wingbox.stress_analysis(shear_y,shear_x, moment_y, moment_x, applied_loc, self.mat_struct.shear_modulus)
        constr_cls = IsotropicWingboxConstraints(ideal_wingbox, self.mat_struct, self.len_sec)
        return np.concatenate((constr_cls.interaction_curve(), constr_cls.von_Mises()))
