import numpy  as np
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.problem import StarmapParallelization
import multiprocessing
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
import random
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.optimize as sop
import scipy.constants as const
from .data_structures import Wingbox, Material
from math import ceil
from shapely.geometry import Polygon
from warnings import warn
import pdb

class Boom:
    """ A class to represent an idealized boom
    """
    def __init__(self) -> None:
        self.bid = None #  Boom ID
        self.A = None  # Area boom
        self.x = None # X location
        self.y = None # Y location
        self.sigma = None

    def get_cell_idx(self, wingbox_struct:Wingbox, chord: float) -> int:
        """ Returns the cell index of where the panel is located.

            :param wingbox_struct: The wingbox data structure containing the locations of the spars
            :type wingbox_struct: Wingbox
            :param chord: The local chord of the wing section
            :type chord: float
            :return: The cell index of where the panel is located
            :rtype: int
        """        
        cell_idx = np.asarray(self.x >= np.insert(wingbox_struct.spar_loc_nondim, 0, 0)*chord) # Get index of the cell
        if  not any(cell_idx):
            cell_idx = 0
        else:
            cell_idx = cell_idx.nonzero()[0][-1]
        return cell_idx


class IdealPanel:
    """
    A class representing a panel in an idealized wingbox
    """
    def __init__(self):
        self.pid = None # Panel ID
        self.bid1 = None  # The boom id of its corresponding first boom id
        self.bid2 = None # The boom id of its corresponding second boom id
        self.b1 = None
        self.b2  = None
        self.t_pnl = None
        self.q_basic = None
        self.q_tot = None
        self.tau = None
        self.dir_vec = None
    
    def get_cell_idx(self, wingbox_struct:Wingbox, chord: float) -> int:
        """ Returns the cell index of where the panel is located.

        :param wingbox_struct: The wingbox data structure containing the locations of the spars
        :type wingbox_struct: Wingbox
        :param chord: The local chord of the wing section
        :type chord: float
        :return: The cell index of where the panel is located
        :rtype: int
        """        
        cell_idx = np.asarray((self.b1.x + self.b2.x)/2 >= np.insert(wingbox_struct.spar_loc_nondim, 0, 0)*chord) # Get index of the cell
        if  not any(cell_idx):
            cell_idx = 0
        else:
            cell_idx = cell_idx.nonzero()[0][-1]
        return cell_idx
    
    def length(self) -> float:
        """ Length of the panel based on the coordinates of the boom. Boom center is used as the 
        assumption is that the booms are infitestimally small.

        :return: The length of the panel
        :rtype: float
        """        
        return np.sqrt((self.b2.x - self.b1.x)**2 + (self.b2.y - self.b1.y)**2)

    def set_b1_to_b2_vector(self) -> tuple:
        try:
            x_comp = (self.b2.x - self.b1.x)/self.length()
            y_comp = (self.b2.y - self.b1.y)/self.length()
        except AttributeError as err:
            raise err("The boom instance has not been assigned yet or is missing the attribute x and y")
        self.dir_vec = [x_comp, y_comp]

    def set_b2_to_b1_vector(self) -> tuple:
        try:
            x_comp = (self.b1.x - self.b2.x)/self.length()
            y_comp = (self.b1.y - self.b2.y)/self.length()
        except AttributeError as err:
            raise err("The boom instance has not been assigned yet or is missing the attribute x and y")
        self.dir_vec =  [x_comp, y_comp]

class IdealWingbox():
    """ A class representing an idealized wingbox, containing methods to perform computations
    on that instants and some accessed methods. It is not recommended to use as a standalone tool but should 
    be using in conjunction with the :func:`discretize_airfoil` function.

    **Assumptions**
    1. The x datum of the coordinate system should be attached to the leading edge of the 
    wingbox
    2. Some methods such as the read_cell_area expect the first and last to begin and end on a vertex.


    """    
    def __init__(self, wingbox: Wingbox, chord:float) -> None:
        self.wingbox_struct = wingbox
        self.chord = chord
        self.x_centroid = None # datum attached to leading edge
        self.y_centroid = None
        self.panel_dict = {}
        self.boom_dict = {}
        pass

    @property
    def Ixx(self):
        Ixx = 0
        for boom in self.boom_dict.values():
            Ixx += boom.A*(boom.y - self.y_centroid)**2
        return Ixx

    @property
    def _read_skin_panels_per_cell(self) -> List[int]:
        """ Returns a list with the amount of panels on the skin per cell. That is ignoring the panels
        which are part of one the spars. This functions requires a fully filled out boom and panel dictionary

        :return: An n x m 2d list where n is the amount cells and m the amount of panels (might not be identical for each cell)
        :rtype: list
        """        
        panel_lst =  []

        spar_loc_arr = np.insert(self.wingbox_struct.spar_loc_nondim, 0,0)*self.chord
        for idx, spar_loc in enumerate(spar_loc_arr):
            if idx != len(spar_loc_arr) - 1:
                panel_lst.append([i for i in self.panel_dict.values() if (spar_loc <= (i.b1.x + i.b2.x)/2 < spar_loc_arr[idx + 1]) and i.b1.x != i.b2.x])
            else:
                panel_lst.append([i for i in self.panel_dict.values() if  (i.b1.x + i.b2.x)/2 >= spar_loc and i.b1.x != i.b2.x])
        return panel_lst

    def get_total_area(self) -> float:
        """ Returns the total area of all the booms. These booms also contain the addition of the skin thicknesses. This function in the library is used 
        for the optimizatio of a wingbox.

        :return: The total area of all the booms combined
        :rtype: float
        """        

        tot_area = 0
        for boom in self.boom_dict.values():
            tot_area += boom.A
        return tot_area


    def get_polygon_cells(self, validate=False) -> List[float]:
        """ 
        Compute the area of each cell with the help of the shapely.geometry.Polygon class. The function expects a fully loaded airfoil to be in the 
        class using the idealized_airfoil function. Errenous results or an error will be given in case this is not the case! When using this function for the first time
        with a new airfoil it is advised to run it once with validate= True to see if the resulting areas are trustworthy. This will
        show you n plots of the cell polygon where n is the amount of cells.

        **Assumptions**

        1. Function is built for a object built with the discretize airfoil, that is cell 0 has a singular point as a leading edge, that is one point is the furthest ahead. The same goes for cell n but with the trailing edge

        :param validate: When True will show the 3 plots described above, defaults to False
        :type validate: bool, optional
        :return: A list whic is m long where m is the amount of cells. Each element is the area of the respective cell.
        :rtype: List[float]
        """        
        bm_per_cell_lst =  []
        polygon_lst =  []

        # Get all the booms per cell
        spar_loc_arr = np.insert(self.wingbox_struct.spar_loc_nondim, 0,0)*self.chord
        for idx, spar_loc in enumerate(spar_loc_arr):
            if idx != len(spar_loc_arr) - 1:
                bm_per_cell_lst.append([i for i in self.boom_dict.values() if (spar_loc <= i.x <= spar_loc_arr[idx + 1]) or np.isclose(i.x, spar_loc) or np.isclose(i.x, spar_loc_arr[idx + 1])])
            else:
                bm_per_cell_lst.append([i for i in self.boom_dict.values() if  (i.x >= spar_loc) or np.isclose(i.x, spar_loc)])
        
        # The code in this for loop is required to correctly sort the coordinates 
        # so a polygon can be created from which the area is computed
        for idx, cell in enumerate(bm_per_cell_lst):
            x_lst = [i.x for i in cell]
            y_lst = [i.y for i in cell]
            coord_arr = np.vstack((x_lst, y_lst)).transpose()
            if idx == 0:
                # The boundary here is always chosen to be the leading edge
                bnd = y_lst[x_lst.index(np.min(x_lst))] 
            elif idx == self.wingbox_struct.n_cell - 1:
                # The boundary here is always chosen to be the trailing edge
                bnd = y_lst[x_lst.index(np.max(x_lst))] 
            else:
                # Get the vertices of the spar in order to always choose 
                # the correct horizontal boundary to split from
                idx_xmax = np.where(x_lst == np.max(x_lst))
                idx_xmin = np.where(x_lst == np.min(x_lst))

                y_right_max = np.max(np.array(y_lst)[idx_xmax])
                y_right_min = np.min(np.array(y_lst)[idx_xmax])

                y_left_max = np.max(np.array(y_lst)[idx_xmin])
                y_left_min = np.min(np.array(y_lst)[idx_xmin])

                # Select the limitng vertices and select the boundary by choosing the half way
                # point between them.
                bnd_top = np.min([y_right_max, y_left_max])
                bnd_bot = np.max([y_right_min, y_left_min])
                bnd = (bnd_top + bnd_bot)/2

            # Get all upper  coords and sort them from low to high based on the x location
            upper_coord = coord_arr[coord_arr[:,1] >= bnd,:]
            upper_coord = upper_coord[upper_coord[:,0].argsort(),:]
            
            # Sort the upper coords of the left spar and right spar 
            if idx != 0: # The first cell does not have a left spar
                #left spar
                spar_sort_idx = np.where(upper_coord[:,0] == np.min(upper_coord[:,0]))[0] # Correct the sorting of the nodes on the left spar
                left_spar =  upper_coord[spar_sort_idx,:]
                left_spar = left_spar[left_spar[:,1].argsort(),:]
                upper_coord[spar_sort_idx,:] = left_spar

                # right spar
                spar_sort_idx = np.where(upper_coord[:,0] == np.max(upper_coord[:,0]))[0] # Correct the sorting of the nodes on the left spar
                right_spar =  upper_coord[spar_sort_idx,:]
                right_spar = right_spar[np.flip(right_spar[:,1].argsort()),:]
                upper_coord[spar_sort_idx,:] = right_spar

            # Get all lower coords and sort them from high to low based on the x location
            lower_coord = coord_arr[coord_arr[:,1] < bnd,:]
            lower_coord = np.flip(lower_coord[lower_coord[:,0].argsort(),:], axis=0)

            # Sort the lower coords of the left spar internally 
            if idx != 0: # The first cell does not have a left spar
                # left spar
                spar_sort_idx = np.where(lower_coord[:,0] == np.min(lower_coord[:,0]))[0] # Correct the sorting of the nodes on the left spar
                left_spar = lower_coord[spar_sort_idx,:]
                left_spar = left_spar[left_spar[:,1].argsort(),:]
                lower_coord[spar_sort_idx,:] = left_spar

                # right spar
                spar_sort_idx = np.where(lower_coord[:,0] == np.max(lower_coord[:,0]))[0] # Correct the sorting of the nodes on the left spar
                right_spar = lower_coord[spar_sort_idx,:]
                right_spar = right_spar[np.flip(right_spar[:,1].argsort()),:]
                lower_coord[spar_sort_idx,:] = right_spar

            # An correct set of coordinates is now achieved
            coord_arr = np.vstack((upper_coord, lower_coord)) 
            coord_arr[0,:] = coord_arr[-1:] # Close the loop so the polygon can compute the area
            poly = Polygon(coord_arr) # Create the actual polygon
            polygon_lst.append(poly) # Get the area

            if validate:
                x,y = poly.exterior.xy
                plt.hlines([bnd], np.min(x_lst), np.max(x_lst), "r")
                plt.plot(x,y)
                plt.show()

        return polygon_lst

    def get_cell_areas(self, validate= False) -> List[float]:
        polygon_lst = self.get_polygon_cells(validate)
        return [i.area for i in polygon_lst]


    def _compute_boom_areas(self, chord) -> None:
        """ Function that creates all boom areas, program assumes a fully functional panel and boom
        dictionary where all values have the full classes assigned. Function can be used by user manually
        but it is generall advised to use the discretize_airfoil function to create a wingbox.

        **Assumptions**

        #. The idealizatin only takes into account a force in the vertical direction (that is tip-deflection path)
        #. 
        """        

        # Find stringer area to add per cell
        # str_contrib = []
        # for idx, n_str in enumerate(self.wingbox_struct.str_cell):
        #     str_contrib.append(n_str*self.wingbox_struct.area_str/len(self._read_skin_panels_per_cell[idx]))

        # Define absolute spar location for use within the loop
        spar_loc_abs = np.array(self.wingbox_struct.spar_loc_nondim)*chord
        # Per boom find all the panel in which the boom is found and add skin contribution
        for boom in self.boom_dict.values():
            boom_area = boom.A =  0
            # Retrieve all panel where boom is a part of
            pnl_lst = [pnl for pnl in self.panel_dict.values() if pnl.bid1 == boom.bid or pnl.bid2 == boom.bid]
            for pnl in pnl_lst:
                if boom.bid == pnl.bid1:
                    boom_area += pnl.t_pnl*pnl.length()/6*(2 + (pnl.b2.y - self.y_centroid )/(boom.y - self.y_centroid))
                else:
                    boom_area += pnl.t_pnl*pnl.length()/6*(2 + (pnl.b1.y - self.y_centroid)/(boom.y - self.y_centroid))

            boom.A = boom_area
            # check if it is not a spar boom, this will get a flange addition in the future maybe
            if not any(np.isclose(boom.x, spar_loc_abs )):
                boom.A += self.wingbox_struct.area_str
            if boom.A < 0: 
                warn("Negative boom areas encountered this is currently a bug, temporary fix takes the absolute value")
                boom.A = np.abs(boom.A)


    def stress_analysis(self,  intern_shear:float, internal_mz:float, shear_centre_rel : float, shear_mod: float, validate=False) ->  Tuple[list,list]:
        """ 
        Perform stress analysis on  a wingbox section 

        List of assumptions (Most made in Megson, some for simplificatoin of code)
        ---------------------------------------------------------------------------
        - The effect of taper are not included see 21.2 (See megson) TODO: future implementation
        - Lift acts through the shear centre (no torque is created) TODO: future implementation
        - Stresses due to drag are not considered. 



        :param intern_shear: _description_
        :type intern_shear: float
        :param internal_mz: _description_
        :type internal_mz: float
        :param shear_mod: shear_modulus
        :type shear_mod: float
        :return: _description_
        :rtype: Tuple[float, dict]
        """    

        # Ensure all shear flows are set to none because the function relies on it
        for pnl in self.panel_dict.values():
            pnl.q_basic = None
            pnl.q_total = None

        # First compute all the direct stresses
        for boom in self.boom_dict.values():
            boom.sigma = internal_mz*(boom.y - self.y_centroid)/self.Ixx

        
        # Start by computing basic shear stresses

        cut_lst = [] #define list to 

        # Loop over all cells and specify required conditions in order to cut
        # We cut the upper panel left of each spar. Except for the last cell, here we cut to the
        # right of the last spar. Hence the elif statement with one less condition
        for idx, spar_loc in enumerate(self.wingbox_struct.spar_loc_nondim):
            spar_loc *= self.chord
            # find panel left of the spar and cut it it
            if idx != len(self.wingbox_struct.spar_loc_nondim) - 1:
                for pnl in self.panel_dict.values():
                    cond1 = pnl.b1.x != pnl.b2.x # Remove the spars from selectioj
                    cond2 = pnl.b2.y >= self.y_centroid and pnl.b1.y >= self.y_centroid # Only upper skin 
                    cond3 = pnl.b1.x <= spar_loc  and pnl.b2.x <= spar_loc # Get the panel left of the spar
                    cond4 = pnl.b1.x == spar_loc or pnl.b2.x == spar_loc # Only select panel attached to the spar
                    if cond1 and cond2 and cond3 and cond4: # Combine all statements
                        pnl.q_basic = 0 # Cut this panel
                        cut_lst.append(pnl) 
            # If are at the last cell cut both the left and right panel connected to the spar
            elif idx ==  len(self.wingbox_struct.spar_loc_nondim) - 1:
                for pnl in self.panel_dict.values():
                    cond1 = pnl.b1.x != pnl.b2.x
                    cond2 = (pnl.b2.y + pnl.b1.y)/2 >= self.y_centroid 
                    cond3 = pnl.b1.x == spar_loc or pnl.b2.x == spar_loc
                    if cond1 and cond2 and cond3:
                        pnl.q_basic = 0
                        cut_lst.append(pnl)
            # In case something goes wrong with the previous statements
            else:
                raise Exception(f"Something went wrong, more iterations were made than there are cells")
            
        cut_lst = sorted(cut_lst, key= lambda x1: np.min([x1.b1.x, x1.b2.x]))# Sort the cut panels based on their x location.
            
        # Get panels per cell excluding the right spar of that cell 
        pnl_per_cell_lst = []
        spar_loc_arr = np.insert(self.wingbox_struct.spar_loc_nondim, 0,0)*self.chord # dimensionalize and insert a zero
        for idx, spar_loc in enumerate(spar_loc_arr):
            # conditions for any cell except the last
            if idx != len(spar_loc_arr) - 1:
                pnl_per_cell_lst.append([i for i in self.panel_dict.values() if (spar_loc <= (i.b1.x + i.b2.x)/2 < spar_loc_arr[idx + 1])])
            # Conditions for the last cell
            else:
                pnl_per_cell_lst.append([i for i in self.panel_dict.values() if  (i.b1.x + i.b2.x)/2 >= spar_loc])
            
        shear_const = -intern_shear/self.Ixx # define -Sy/Ixx which is used repeatdly

        # Chain from the cut panel per cell until all q_basic have been defined
        for idx, cell in enumerate(pnl_per_cell_lst):
            # Shows the selection of panels made, verify that the right spar is not included
            if validate:
                for panel in cell:
                    x = [panel.b1.x, panel.b2.x]
                    y = [panel.b1.y, panel.b2.y]
                    plt.plot(x,y, ">-")
                plt.show()

            curr_pnl = cut_lst[idx] # set beginning of the q_basic chain
            q_basic = 0
            for pnl_num in range(len(cell) - 1):
                # Find connected panels to boom 1 of the current panel except itself of course
                b1_lst = [i for i in cell if (curr_pnl.b1 == i.b1 or curr_pnl.b1 == i.b2) and i != curr_pnl]
                b2_lst = [i for i in cell if (curr_pnl.b2 == i.b1 or curr_pnl.b2 == i.b2)  and i != curr_pnl] # idem but for boom 2
                # If b1 is attached to another panel and q_basic is not attached yet continue from this panel
                if len(b1_lst) == 1 and b1_lst[0].q_basic == None:
                    # Check if it was to boom 1
                    if curr_pnl.b1 == b1_lst[0].b1:
                        curr_pnl = b1_lst[0]
                        q_basic += shear_const*curr_pnl.b1.A*(curr_pnl.b1.y - self.y_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b1_to_b2_vector()
                    # if not boom 1 then it was boom 2
                    else:
                        curr_pnl = b1_lst[0]
                        q_basic += shear_const*curr_pnl.b2.A*(curr_pnl.b2.y - self.y_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b2_to_b1_vector() # Set the direction in which the shear flow was defined
                # If b2 is attached to another panel and q_basic is not attached yet continue from this panel
                elif len(b2_lst) == 1 and b2_lst[0].q_basic == None:
                    # Check if it was to boom 1
                    if curr_pnl.b2 == b2_lst[0].b2:
                        curr_pnl = b2_lst[0]
                        q_basic += shear_const*curr_pnl.b2.A*(curr_pnl.b2.y - self.y_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b2_to_b1_vector() # Set the direction in which the shear flow was defined
                    # if not boom 1 then it was boom 2
                    else:
                        curr_pnl = b2_lst[0]
                        q_basic += shear_const*curr_pnl.b1.A*(curr_pnl.b1.y - self.y_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b1_to_b2_vector()
                else: 
                    raise Exception("No connecting panel was found")


        #=========================================================================
        # Now Compute the complementary shear flows and the twist per unit lengt   
        #========================================================================
        area_lst = self.get_cell_areas() # Get the area per cell
        centroid_lst = [np.array(poly.centroid.xy).flatten() for poly in self.get_polygon_cells()] # get the centroid of each cell

        # Get the panel per cell, that is the fully defined cell. 
        pnl_per_cell_lst2 = []
        spar_loc_arr = np.insert(self.wingbox_struct.spar_loc_nondim, 0,0)*self.chord # dimensionalize and insert a zero
        for idx, spar_loc in enumerate(spar_loc_arr):
            # conditions for any cell except the last
            if idx != len(spar_loc_arr) - 1:
                pnl_per_cell_lst2.append([i for i in self.panel_dict.values() if (spar_loc <= (i.b1.x + i.b2.x)/2 <= spar_loc_arr[idx + 1])])
            # Conditions for the last cell
            else:
                pnl_per_cell_lst2.append([i for i in self.panel_dict.values() if  (i.b1.x + i.b2.x)/2 >= spar_loc])
        
        # Define A and b matrix to compute qs,1 qs,2 \cdots qs,n and dtheta/dz 
        n_cell =  self.wingbox_struct.n_cell  # shortcut to amount of cells
        b_arr = np.zeros((n_cell + 1,1))
        A_arr = np.zeros((n_cell + 1, n_cell + 1,))

        # Set up the equations for the twist per unit length in array A
        # Everything counterclockwise (ccw) is set up as positive. We can check whether something is ccw 
        # as we have the direction the q_basic was set up and the centroid of each cell thus a simple cross 
        # product will tell us so
        for idx, cell in enumerate(pnl_per_cell_lst2):
            # Flag to easily verify whether the cell geometry selection makes sense
            if validate:
                for panel in cell:
                    x = [panel.b1.x, panel.b2.x]
                    y = [panel.b1.y, panel.b2.y]
                    plt.plot(x,y, ">-")
                plt.show()

            if idx == 0:
                x_max = np.max([i.b1.x for i in cell])
                # first assign dtheta/dz
                A_arr[idx, n_cell] = 2*area_lst[idx]*shear_mod
                A_arr[idx, idx] = -1*np.sum([pnl.length()/pnl.t_pnl for pnl in cell])
                A_arr[idx, idx + 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_max)])
            elif  0 < idx < len(pnl_per_cell_lst2) - 1:
                x_min = np.min([i.b1.x for i in cell])
                x_max = np.max([i.b1.x for i in cell])
                A_arr[idx, n_cell] = 2*area_lst[idx]*shear_mod
                A_arr[idx, idx - 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_min) ])
                A_arr[idx, idx] = -1*np.sum([pnl.length()/pnl.t_pnl for pnl in cell])
                A_arr[idx, idx + 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_max)])
                pass
            elif idx == len(pnl_per_cell_lst2) - 1:
                x_min = np.min([i.b1.x for i in cell])
                A_arr[idx, n_cell] = 2*area_lst[idx]*shear_mod
                A_arr[idx, idx - 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_min)])
                A_arr[idx, idx] = -1*np.sum([pnl.length()/pnl.t_pnl for pnl in cell])
            else: 
                raise Exception(f"Something went wrong, more iterations were made then there were cells")

        # Set up eqautions for twist per unit legnth but in b array
        for idx, cell in enumerate(pnl_per_cell_lst2):
            b_ele = 0
            for pnl in cell:
                r_abs_vec = np.array([(pnl.b1.x + pnl.b2.x)/2 , (pnl.b1.y + pnl.b2.y)/2])
                r_rel_vec = r_abs_vec - centroid_lst[idx]
                if pnl.q_basic != 0: 
                    sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                    b_ele += sign*pnl.q_basic*pnl.length()/pnl.t_pnl
                # When we have a panel that was not cut, they do not have a defined direction yet and it does not matter since magnitude is 0
                elif pnl.q_basic == 0:
                    b_ele += pnl.q_basic*pnl.length()/pnl.t_pnl
                else:
                    raise Exception(f"Line should not have been reached")
            b_arr[idx,0] = b_ele


        #------------------------------- Fill in the final equatin, moment equivalence ------------------
        # Contribution from the complementary shear flows
        for idx, cell in enumerate(pnl_per_cell_lst2):
            A_arr[n_cell, idx] = 2*area_lst[idx]
        
        # Contribution to b_arr from basic shear flows and shear force itself
        sum = 0
        for pnl in self.panel_dict.values():
            if pnl.q_basic == 0:
                continue
            else:
                r_abs_vec = np.array([(pnl.b1.x + pnl.b2.x)/2 , (pnl.b1.y + pnl.b2.y)/2])
                moment = pnl.q_basic*np.cross(r_abs_vec, pnl.dir_vec)*pnl.length()
                sum += moment

        b_arr[n_cell, 0] = -1*sum + intern_shear*shear_centre_rel*self.chord

        # Get the actual solution
        X = np.linalg.solve(A_arr, b_arr)
        qs_lst = X[:-1,0]
        dtheta_dz = X[-1]

        #---------------------- Apply the solution to all of the panels (so sorry about the flow of logic and indentation) -------------------------
        # general logic is as follows (felt this was necessary else only God knows what happens here in a month)
        # 1. check whether we are at first cell, last cell or somwhere in between
        # 2. Depending on what cell check if we are on the left, right spar or not on a spar at all
        # 3. Act accordingly to prevous step, if we are somewhere in between some logic is required to define the direction of that panel

        for idx, cell in enumerate(pnl_per_cell_lst2):
            x_max = np.max([i.b1.x for i in cell]) # get maximum x value in cell (will help us find spars)
            x_min = np.min([i.b1.x for i in cell]) # idem but for  minimum
            # Now we will loop over all panel 
            for pnl in cell:
                r_abs_vec = np.array([(pnl.b1.x + pnl.b2.x)/2 , (pnl.b1.y + pnl.b2.y)/2])
                r_rel_vec = r_abs_vec - centroid_lst[idx]
                if idx == 0:
                    # If it is on the right spar qs,0+1 will also have an influence
                    if (pnl.b1.x == pnl.b2.x == x_max):
                        sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                        pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx] - sign*qs_lst[idx + 1]
                        pnl.tau = pnl.q_tot/pnl.t_pnl
                    # Else if it not on the right spar just add qs0
                    else:
                        # Check if it was not the cut panel
                        if pnl.q_basic != 0: 
                            sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                            pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx]
                            pnl.tau = pnl.q_tot/pnl.t_pnl
                        # if it is a cut panel
                        elif pnl.q_basic == 0:
                            pnl.q_tot = qs_lst[idx]
                            pnl.tau = pnl.q_tot/pnl.t_pnl
                            # Define ccw direction as these panel did not have a direction yet
                            pnl.set_b1_to_b2_vector()
                            if np.cross(r_rel_vec, pnl.dir_vec) > 0:
                                pass
                            else:
                                pnl.set_b2_to_b1_vector()
                        else:
                            raise Exception(f"Line should not have been reached")
                elif idx != 0 and idx < len(pnl_per_cell_lst2) - 1:
                    # If it is on the right spar qs,0+1 will also have an influence
                    if (pnl.b1.x == pnl.b2.x == x_max):
                        sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                        pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx] - sign*qs_lst[idx + 1]
                        pnl.tau = pnl.q_tot/pnl.t_pnl
                    # If it is on the left spar
                    elif (pnl.b1.x == pnl.b2.x == x_min) :
                        sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                        pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx] - sign*qs_lst[idx - 1]
                        pnl.tau = pnl.q_tot/pnl.t_pnl
                    # Else if it not on the right spar just add qs,n
                    else:
                        # Check if it was not the cut panel
                        if pnl.q_basic != 0: 
                            sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                            pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx]
                            pnl.tau = pnl.q_tot/pnl.t_pnl
                        # if it is a cut panel
                        elif pnl.q_basic == 0:
                            pnl.q_tot = qs_lst[idx]
                            pnl.tau = pnl.q_tot/pnl.t_pnl
                            # Define ccw direction as these panel did not have a direction yet
                            pnl.set_b1_to_b2_vector()
                            if np.cross(r_rel_vec, pnl.dir_vec) > 0:
                                pass
                            else:
                                pnl.set_b2_to_b1_vector()
                        else:
                            raise Exception(f"Line should not have been reached")
                elif idx == len(pnl_per_cell_lst2) - 1:
                    # If it is on the left spar
                    if (pnl.b1.x == pnl.b2.x == x_min) :
                        sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                        pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx] - sign*qs_lst[idx - 1]
                        pnl.tau = pnl.q_tot/pnl.t_pnl
                    # Else if  just add qs,n. As no other influece needs to be taken into account
                    else:
                        # Check if it was not the cut panel
                        if pnl.q_basic != 0: 
                            sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                            pnl.q_tot  = pnl.q_basic + sign*qs_lst[idx]
                            pnl.tau = pnl.q_tot/pnl.t_pnl
                        # if it is a cut panel
                        elif pnl.q_basic == 0:
                            pnl.q_tot = qs_lst[idx]
                            pnl.tau = pnl.q_tot/pnl.t_pnl
                            # Define ccw direction as these panel did not have a direction yet
                            pnl.set_b1_to_b2_vector()
                            if np.cross(r_rel_vec, pnl.dir_vec) > 0:
                                pass
                            else:
                                pnl.set_b2_to_b1_vector()
                        else:
                            raise Exception(f"Line should not have been reached")

        return  qs_lst, dtheta_dz

    def plot_direct_stresses(self) -> None:
            plt.figure(figsize=(10,1))
            x_lst = np.array([i.x for i in self.boom_dict.values()])
            y_lst = np.array([i.y for i in self.boom_dict.values()])
            stress_arr = np.array([i.sigma/1e6 for i in self.boom_dict.values()])
            hover_data = [f"stress = {i.sigma/1e6} Mpa" for i in self.boom_dict.values()]
            fig = px.scatter(x= x_lst, y= y_lst, color= stress_arr, title= "Direct stress")
            fig.update_traces(marker=dict(size=12,
                            line=dict(width=2,
                            color='DarkSlateGrey')),
                            selector=dict(mode='markers'))
            fig.show()



    def plot_shear_stress(self) -> None:
        y_arr  = [i.y for i in self.boom_dict.values()]
        y_max = np.max(y_arr)
        y_min = np.min(y_arr)

        # Sample data and color mapping - replace with your actual data
        stress_values = np.abs([panel.tau for panel in self.panel_dict.values()])/1e6
        norm = (stress_values - stress_values.min()) / (stress_values.max() - stress_values.min())

        # Modified traces and colorbar dummy trace
        traces = []
        colorbar_trace_x = []
        colorbar_trace_y = []
        colorbar_trace_stress = []

        stress_values = np.abs([panel.tau for panel in self.panel_dict.values()])/1e6
        norm = plt.Normalize(stress_values.min(), stress_values.max())
        cmap = plt.cm.plasma

        for key, panel in self.panel_dict.items():
            x = [panel.b1.x, panel.b2.x]
            y = [panel.b1.y, panel.b2.y]
            stress = abs(panel.tau/1e6)
            hover_text = f"Panel: {panel.pid}, Stress: {stress} MPa"  # Modify as needed
            col = cmap(norm(stress))
            col = "rgb" + str(col[:-1])

            trace = go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                marker=dict(color=col, symbol='circle'),
                line=dict(color=col, width= 4),
                showlegend=False,
                hovertext= hover_text
            )
            traces.append(trace)

            # For colorbar
            colorbar_trace_x.extend(x)
            colorbar_trace_y.extend(y)
            colorbar_trace_stress.extend([stress, stress])  # Repeated for each point

        # Layout configuration
        layout = go.Layout(
            title='Panel Stress Visualization',
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis', range=[y_min - 0.1, y_max + 0.1]),
            coloraxis=dict(colorscale='Viridis', colorbar=dict(title='Stress Value')),
        )

        # Create figure and add traces
        fig = go.Figure(data=traces, layout=layout)
        fig.show()


        # Rest of your layout and figure code
    def plot_quiver_shear_stress(self, scale=.020, arrow_scale=0.4) -> None:
        pnl_lst = [i for i in self.panel_dict.values()]

        x = [(i.b1.x + i.b2.x)/2 for i in pnl_lst]
        y = [(i.b1.y + i.b2.y)/2 for i in pnl_lst]
        u = list()
        v = list()

        for pnl in pnl_lst:
            pass 
            if pnl.tau > 0:
                u.append(pnl.dir_vec[0])
                v.append(pnl.dir_vec[1])
            elif pnl.tau <= 0:
                u.append(-pnl.dir_vec[0])
                v.append(-pnl.dir_vec[1])

        # Create quiver figure
        fig = ff.create_quiver(x, y, u, v,
                            scale= scale,
                            arrow_scale= arrow_scale,
                            line= dict(color="red", width = 3),
                            name='Direction of shear flows',
                            line_width=1)
        fig.show()




    def plot_geometry(self) -> None:

        # Modified traces and colorbar dummy trace
        traces = []
        col_lst = px.colors.qualitative.Dark24
        colorbar_trace_x = []
        colorbar_trace_y = []
        colorbar_trace_stress = []

        for key, panel in self.panel_dict.items():
            x = [panel.b1.x, panel.b2.x]
            y = [panel.b1.y, panel.b2.y]
            hover_text = [f"Boom : {panel.bid1}, Area: {np.round(panel.b1.A*1e6, 2)} mm^2",
                          f"Boom : {panel.bid2}, Area: {np.round(panel.b2.A*1e6,2 )} mm^2"]  # Modify as needed

            trace = go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                marker=dict(color= "red", symbol='circle', size=12),
                line=dict(color= random.choice(col_lst), width= 4),
                showlegend=False,
                opacity=0.7,
                hovertext= hover_text
            )
            traces.append(trace)

            # For colorbar
            colorbar_trace_x.extend(x)
            colorbar_trace_y.extend(y)

        # Layout configuration
        layout = go.Layout(
            title='Discretization of airfoil',
            xaxis=dict(title='X-axis'),
            yaxis=dict(title='Y-axis'),
        )

        # Create figure and add traces
        fig = go.Figure(data=traces, layout=layout)
        fig.show()


def read_coord(path_coord:str) -> np.ndarray:
    """
    Returns an  m x 2 array of the airfoil coordinates based on a Selig formatted dat file.

    :param path_coord: m x 2 array with the airfoil coordinates where x goes top trailing edge to top leading edge and then back to  lower trailing edge. I.e it keeps the Selig format
    :type path_coord: str
    :return: _description_
    :rtype: np.ndarray
    """    
    # List to save formatted coordinates
    airfoil_coord = []

    with open(path_coord) as f:
        for line in f.readlines():
            if any([i.isalpha() for i in line]):
                continue
            airfoil_coord.append([float(i) for i in line.split()])
        airfoil_coord = np.array(airfoil_coord)
    return airfoil_coord

def spline_airfoil_coord(path_coord:str, chord:float) -> Tuple[CubicSpline, CubicSpline]:
    """ Return two function which interpolate the coordinates of the airfoil given. Two functions are returned
    the first interpolates the top skin and the second function interpolates the bottom skin. The result interpolation functions 
    take into account an airfoil scaled by the given chord.

    :param path_coord: Path to the airfoil coordinates using the Selig format.
    :type path_coord: str
    :return: A cubic spline of the top skin and lower skin, respectively.
    :rtype: Tuple[CubicSpline, CubicSpline]
    """        
    
    coord_scaled = read_coord(path_coord)*chord # coordinates of airfoil scaled by the chord
    top_coord =  coord_scaled[:np.argmin(coord_scaled[:,0]), :] #coordinates of top skin
    top_coord = np.flip(top_coord, axis=0)
    bot_coord =  coord_scaled[np.argmin(coord_scaled[:,0]):, :] # coordinates of bottom skin

    top_interp = CubicSpline(top_coord[:,0], top_coord[:,1])
    bot_interp = CubicSpline(bot_coord[:,0], bot_coord[:,1])
    return top_interp, bot_interp

def get_centroids(path_coord:str) -> Tuple[float, float]:
    r""" Compute the nondimensional x and y centroid based on the coordinate file of an airfoil.
    The centroids are computing assuming the following:
     
     **Assumptions**

     1. It is only based on the skin, i.e  the spar webs and stringers are ignored. Additionally the different thickness of the skin are not taken into account 
     The implication being that the x centroid should be at x/c = 0.5. Unless there was a bias in the sampling points

    **Future improvement** 

     1. Take into account the spar webs for a better x centroid. However irrelevant for now as we only
     take into account forces in the vertical directoin
    

    :param path_coord: Path to the geometry file using the Selig format
    :type path_coord: str
    :return: The nondimensional x and y centroid of the airfoil
    :rtype: Tuple[float, float]
    """    
    coord = read_coord(path_coord)
    y_centroid_dimless = np.sum(coord[:,1])/coord.shape[0]
    x_centroid_dimless = np.sum(coord[:,0])/coord.shape[0]
    return x_centroid_dimless, y_centroid_dimless


def discretize_airfoil(path_coord:str, chord:float, wingbox_struct:Wingbox) -> IdealWingbox:
    r""" Create a discretized airfoil according to the principles of Megson based on a path to a txt file containing
    the non-dimensional coordinates of the airfoil, the corresonding chord and the wingbox data structure fully filled in.

    **Assumptions**

    #. Airfoil is idealized according Megson ch. 20 
    #. Each stringer will form one boom in the discretized airfoil 
    #. Only an equal amount of stringers can be specified per cell, if that is not the case a warning is issued however. (due to the method of discretization)
    #. The ratio of :math:`\frac{\sigma_1}{\sigma_2}` required for the skin contribution to the  boom size based on the skin is determined by the ratio of their y positin thus :math:`\frac{y_1}{y_2}`. \n
    

    **General Procedure**

    #. Create a spline of the top and bottom airfoil
    #. Create array along which to sample this spline to create the booms, creating specific samples for the spar positions
    #. Move over top surface creating booms and panel as we go
    #. Do the same for the bottom surface moving in a circle like motion
    #. Move over all the spars and create booms and panels as we go.
    #. Iterate over all booms and add skin contribution and stringer contribution to all their areas

    **Future Improvements**

    #. Add contribution of spar caps (For now as been left out as I did not see the value of it at the time)


    :param path_coord: _description_
    :type path_coord: str
    :param chord: _description_
    :type chord: float
    :param wingbox_struct: _description_
    :type wingbox_struct: Wingbox
    :return: _description_
    :rtype: IdealWingbox
    """    
    
    # Check  wingbox data structure
    assert len(wingbox_struct.t_sk_cell) == wingbox_struct.n_cell, "Length of t_sk_cell should be equal to the amount of cells"
    assert len(wingbox_struct.str_cell) == wingbox_struct.n_cell, "Length of str_cell should be equal to the amount of cells"
    assert len(wingbox_struct.spar_loc_nondim) == wingbox_struct.n_cell - 1, "Length of spar_loc should be equal to the amount of cells - 1"
    assert all(np.round(wingbox_struct.str_cell,0) > 4), "Each cell must have atleast five stringers. The following configuration would results in an error during runtime"

    top_interp, bot_interp = spline_airfoil_coord(path_coord, chord)
    x_centr, y_centr = get_centroids(path_coord)

    wingbox = IdealWingbox(wingbox_struct, chord)
    wingbox.x_centroid =  x_centr*chord
    wingbox.y_centroid =  y_centr*chord


    spar_loc_lst: list = np.insert(wingbox_struct.spar_loc_nondim, [0, wingbox_struct.n_cell - 1], [0, 1])*chord
    str_lst: list = wingbox_struct.str_cell # list with the amount of stringers per cell

    x_boom_loc = np.array([])

    #TODO create a new x_boom_loc 
    for idx, loc in enumerate(spar_loc_lst): 
        if idx != len(spar_loc_lst) - 1:
            len_cell = spar_loc_lst[idx + 1] - loc
            # A 0.1 starting point is chosen to avoid 
            str_tot = str_lst[idx] 
            
            # See assumptions, the following make sure any float get rounded since
            # the optimizer usually does not return nice integers. It then also gives a warning
            # if the value was not an integer
            if isinstance(str_tot, int):
                if str_tot % 2 != 0:
                    warn(f"{str_lst[idx]} was not an even number and will be floored for conservative reasons")
                    n_str = np.floor(str_lst[idx]/2)
                else: 
                    n_str = str_lst[idx]/2
            else: 
                str_tot = np.round(str_tot,0)
                if str_tot % 2 != 0:
                    warn(f"{str_lst[idx]} was not an even number and will be floored for conservative reasons")
                    n_str = np.floor(str_lst[idx]/2)
                else: 
                    n_str = str_lst[idx]/2

            if idx != 0:
                loc_cell =  np.linspace(loc , spar_loc_lst[idx + 1], int(n_str + 1))[1:]
            else:
                loc_cell =  np.linspace(loc , spar_loc_lst[idx + 1], int(n_str + 1))
            x_boom_loc = np.append(x_boom_loc, loc_cell)
        else: 
            pass


    boom_dict = {}
    panel_dict = {}
    bid = 0 # set bid to zero
    pid = 0  # set panel counter to zero
    # create the upper booms and panels from leading edge to trailing edge
    for idx, x_boom in enumerate(x_boom_loc, start=0):
        boom = Boom()
        boom.bid = bid
        boom.x =  x_boom
        boom.y = float(top_interp(x_boom))

        if boom.bid not in boom_dict.keys():
            boom_dict[boom.bid] = boom 
        else:
            raise RuntimeError(f"Boom id {boom.bid} already exists in variable boom dictionary")

        if not idx == 0: # Create panels, can not create a panel on the first pass through the for loop
            pnl = IdealPanel()
            pnl.pid = pid
            pnl.bid1 = bid - 1
            pnl.bid2 = bid 
            pnl.b1 =  boom_dict[pnl.bid1]
            pnl.b2 =  boom_dict[pnl.bid2]
            cell_idx = pnl.get_cell_idx(wingbox_struct, chord)
            pnl.t_pnl = wingbox_struct.t_sk_cell[cell_idx]

            if pnl.pid not in panel_dict.keys():
                panel_dict[pid] = pnl
            else:
                raise RuntimeError(f"Boom id {boom.bid} already exists in variable boom dictionary")
            pid += 1

        bid += 1

    # Create the lower booms and panels from trailing edge to leading edge
    for x_boom in np.flip(x_boom_loc[1:-1]):
        boom = Boom()
        boom.bid = bid
        boom.x =  x_boom
        boom.y = float(bot_interp(x_boom))
        boom_dict[boom.bid] = boom

        pnl = IdealPanel()
        pnl.pid = pid
        pnl.bid1 = bid - 1
        pnl.bid2 = bid 
        pnl.b1 =  boom_dict[pnl.bid1]
        pnl.b2 =  boom_dict[pnl.bid2]
        cell_idx = pnl.get_cell_idx(wingbox_struct, chord)
        pnl.t_pnl = wingbox_struct.t_sk_cell[cell_idx]
        #TODO create the areas  based on sigma ratio
        panel_dict[pid] = pnl

        bid += 1
        pid += 1

    # Connect the last panel (the panel between the first and the last boom)
    pnl = IdealPanel()
    pnl.pid = pid
    pnl.bid1 = bid - 1
    pnl.bid2 = 0
    pnl.b1 =  boom_dict[pnl.bid1]
    pnl.b2 =  boom_dict[pnl.bid2]
    cell_idx = pnl.get_cell_idx(wingbox_struct, chord)
    pnl.t_pnl = wingbox_struct.t_sk_cell[cell_idx]
    #TODO create the areas  based on sigma ratio
    panel_dict[pid] = pnl
    pid +=1 

    # Create panels on the spar booms
    for spar_loc in wingbox_struct.spar_loc_nondim:
        spar_loc *= chord
        spar_booms = [i for i in boom_dict.values() if i.x == spar_loc] # find booms defined on this spar

        # assert that only two booms should be found
        if len(spar_booms) != 2:
            raise RuntimeError(f"Object {spar_booms} should have length two but had length {len(spar_booms)}")

        # define upper and lower boom
        if spar_booms[0].y > spar_booms[1].y:
            upper_b = spar_booms[0]
            lower_b = spar_booms[1]
        else:
            upper_b = spar_booms[1]
            lower_b = spar_booms[0]


        # Create panel connecting upper boom and lower boom
        pnl = IdealPanel()
        pnl.pid = pid
        pnl.bid1 = upper_b.bid 
        pnl.bid2 = lower_b.bid
        pnl.t_pnl = wingbox_struct.t_sp
        pnl.b1 =  boom_dict[pnl.bid1]
        pnl.b2 =  boom_dict[pnl.bid2]



        if pnl.pid not in panel_dict.keys():
            panel_dict[pid] = pnl
        else:
            raise RuntimeError(f"Boom id {boom.bid} already exists in variable boom dictionary")
        pid += 1


    wingbox.boom_dict.update(boom_dict)
    wingbox.panel_dict.update(panel_dict)

    wingbox._compute_boom_areas(chord)

    return wingbox


class ProblemFreePanel(ElementwiseProblem):
    """ The following problem sets up Discrete variable problem which optimises the amount of stringers per cell. This was done 
    in a separate problem as changing the amount of stringers changes the amount of panels and hence the amount of constraints. Most optimizers
    do not allow this, hence the following problem uses the output of the :class:`WingboxFixedPanel` to abide by the constraints.
    """    

    def __init__(self, shear: float, moment: float, box_struct: Wingbox ):
        super().__init__(n_var= box_struct.n_cell, n_obj=1, n_ieq_constr=0, xl=5, xu=40, vtype= int)

    def _evaluate(self, x, out, *args, **kwargs):
        """ This function get evaluated for each 'element', where an element in this case is a single sequence of n_var long. For more info
        please see the documentation of the pymoo library.

        :param x: A list of n_var long containing the amount of stringers per cell
        :type x: np.ndarray
        :param out: _description_
        :type out: _type_
        """        
        print(x)
        print("\n")
        out["F"] = np.sum((x - 0.5) ** 2)
        out["G"] = 0.1 - out["F"]

class ProblemFixedPanel(ElementwiseProblem):
    """
        The followinng problem sets up an exploration of a wingbox for a fixed amount of stringers per cell. Since it is difficult for an optimizer
        like COBYLA to find a global minimum a genetica algorith is set up first to explore the design space. This problem sets up this search.
    """    

    def __init__(self, shear: float, moment: float, applied_loc: float, chord: float, len_sec: float,
                 box_struct: Wingbox, mat_struct: Material, path_coord: str, **kwargs):

        self.shear = shear
        self.moment = moment
        self.applied_loc = applied_loc
        self.chord = chord
        self.len_sec = len_sec
        self.box_struct = box_struct
        self.mat_struct = mat_struct
        self.path_coord = path_coord

        super().__init__(n_var= box_struct.n_cell + 2, n_obj=1, n_ieq_constr=2*np.sum(box_struct.str_cell), xl=np.ones(self.box_struct.n_cell + 2)*1e-8, xu=np.ones(self.box_struct.n_cell + 2), **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        """
        This function is not to be intended by the user hence it is hinted to be a private method. The function passed to the scip.optimize.minmize function. The arguments that are passed should be in the following 
        order t_sk_cell, t_sp, area_str. Together given a total length of N + 2 where N is the amount of cells.
        This interally checked if it is not the case a runtime error will be raised

        :param x: The design variable vector which is in the following format [t_sk_cell, t_sp, area_str, str_cell]
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

        if len(x) != n_cell + 2:
            raise RuntimeError("The flattened  design vector received does not match the wingbox properties")
        
        
        # Assing new properties to the wingbox struct assuming the specified length
        t_sk_cell =  x[:n_cell]
        t_sp = x[n_cell]
        area_str = x[n_cell + 1]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.area_str =  area_str

        # Discretize airfoil from new given parameters
        wingbox_obj = discretize_airfoil(self.path_coord, self.chord, self.box_struct)

        # Perform stress analysis
        wingbox_obj.stress_analysis(self.shear, self.moment, self.applied_loc, self.mat_struct.shear_modulus)

        #================ Get constraints ======================================
        constr_cls = IsotropicWingboxConstraints(wingbox_obj, self.mat_struct, self.len_sec)
        # return  np.concatenate((constr_cls.von_Mises(), constr_cls.interaction_curve_constr()))
        # return constr_cls.von_Mises()
        out["F"] = wingbox_obj.get_total_area() 
        # out["G"] = -1*np.concatenate((constr_cls.interaction_curve_constr(), constr_cls.von_Mises()))
        out["G"] = np.negative(np.concatenate((constr_cls.interaction_curve_constr(), constr_cls.von_Mises()))) # negative is necessary because pymoo handles inequality constraints differently

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
        :param wingbox_struct: _description_
        :type wingbox_struct: Wingbox
        :param material_struct: _description_
        :type material_struct: Material
        """        

        self.path_coord = path_coord
        self.chord = chord
        self.len_sec =  len_sec
        self.box_struct: Wingbox = wingbox_struct
        self.mat_struct: Material  = material_struct

        # Required Overhead
        self.wingbox_obj: None | IdealWingbox = None

    def GA_optimize(self, shear: float, moment: float, applied_loc: float,
                     n_gen: int = 50, # Possible keywords
                     pop: int = 100,
                     verbose: bool = True,
                     seed: int = 1,
                     cores: int = multiprocessing.cpu_count(),
                     save_hist: bool = True):
        
        """ The following function executes the Genetic Algorithm (`GA <https://pymoo.org/algorithms/soo/ga.html>`_) to optimize the wingbox given to the overarching class
        and with the loads fed to the function. Additionally there are some keywords which are explained below.

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
        n_proccess =  cores
        pool = multiprocessing.Pool(n_proccess)
        runner = StarmapParallelization(pool.starmap)

        problem = ProblemFixedPanel(shear, moment, applied_loc, self.chord,  self.len_sec, 
                                    self.box_struct, self.mat_struct, self.path_coord, elementwise_runner=runner)
        method = GA(pop_size=pop, eliminate_duplicates=True)
        resGA = minimize(problem, method, termination=('n_gen', n_gen   ), seed= seed,
                        save_history=save_hist, verbose=verbose)
        return resGA

    def _obj_func_cobyla(self, x: list, shear: float, moment: float, applied_loc: float, str_lst: list):
        """
        This function is not to be intended by the user hence it is hinted to be a private method. The function passed to the scip.optimize.minmize function. The arguments that are passed should be in the following 
        order t_sk_cell, t_sp, area_str. Together given a total length of N + 2 where N is the amount of cells.
        This interally checked if it is not the case a runtime error will be raised

        :param x: The design variable vector which is in the following format [t_sk_cell, t_sp, area_str, str_cell]
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

        if len(x) != n_cell + 2:
            raise RuntimeError("The flattened  design vector received does not match the wingbox properties")
        
        
        # Assing new properties to the wingbox struct assuming the specified length
        t_sk_cell =  x[:n_cell]
        t_sp = x[n_cell]
        area_str = x[n_cell + 1]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.area_str =  area_str
        self.box_struct.str_cell = str_lst

        # Discretize airfoil from new given parameters
        self.wingbox_obj = discretize_airfoil(self.path_coord, self.chord, self.box_struct)

        # Perform stress analysis
        self.wingbox_obj.stress_analysis(shear, moment, applied_loc, self.mat_struct.shear_modulus)

        return self.wingbox_obj.get_total_area() 


    def optimize_cobyla(self, shear: float, moment: float, applied_loc: float, str_lst: list,
                        bnd_mult: int = 1e3) -> sop._optimize.OptimizeResult:
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
        :return: _description_
        :rtype: sop._optimize.OptimizeResult
        """        

        n = self.box_struct.n_cell # quick reference to number of cells
        # Whatever parameters were given in the datastrucrtre are used as inital estimate
        x0 = self.box_struct.t_sk_cell + [self.box_struct.t_sp] + [self.box_struct.area_str] 

        constr_lst: List[dict] = [
            {'type': 'ineq', 'fun': self._get_constraint_vector, "args": [shear, moment, applied_loc]},
                ]

        #FIXME Cobyla does not take bounds class
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

        res = sop.minimize(self._obj_func_cobyla, x0, args=(shear, moment, applied_loc, str_lst), method="COBYLA" , constraints= constr_lst)
        return res



    def _get_constraint_vector(self, x: list, shear: float, moment: float, applied_loc: float) -> list:
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
        area_str = x[n_cell + 1]

        self.box_struct.t_sk_cell = t_sk_cell
        self.box_struct.t_sp = t_sp
        self.box_struct.area_str =  area_str 

        ideal_wingbox: IdealWingbox = discretize_airfoil(self.path_coord, self.chord, self.box_struct)
        ideal_wingbox.stress_analysis(shear, moment, applied_loc, self.mat_struct.shear_modulus)
        constr_cls = IsotropicWingboxConstraints(ideal_wingbox, self.mat_struct, self.len_sec)
        return np.concatenate(constr_cls.interaction_curv_constr(), constr_cls.von_Mises())


class IsotropicWingboxConstraints:
    def __init__(self, wingbox:IdealWingbox, material_struct: Material, len_to_rib: float) -> None:
        self.wingbox = wingbox
        self.material_struct = material_struct
        self.len_to_rib = len_to_rib
        self.pnl_lst =  [i for i in self.wingbox.panel_dict.values()] 
        self.tens_pnl_idx =  [idx for idx, i in enumerate(self.wingbox.panel_dict.values()) if (i.b1.sigma > 0) and (i.b2.sigma > 0)] # Panel which are in tension since for some of the constraints it is not relevant here

        # Value used for the interpolation to get Kb
        x = [1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ]
        y = [9.5, 7.2, 6.4, 6, 5.8, 5.9, 5.8, 5.6, 5.4]

        # Interpolator for critical shear function
        self.kb_spline = CubicSpline(x,y, extrapolate=False) 
    
    def _kb_interp(self, ar_lst: list):
        res = self.kb_spline(ar_lst)
        res = np.nan_to_num(res, nan= 5.) # Set values that were outside of the interpolation range to 5.
        return res


    def _crit_instability_compr(self) -> list:#TODO
        r""" 
        Compute the elastic instability of a flat sheet in compression for each panel in the idealized wingbox using 
        the equation shown below.


        .. math::
            \sigma_{cr} = k_c  \frac{pi^2 E}{12(1 - \nu)} \left(\frac{t_{sk}}{b}\right)^2

        Where b is the short dimension of plate or loaded edge. For :math:`K_c` a value of 4 was chosen. Please see the figure below for the reasoning.
        Since all edges are considerded simpy supported from either the stringer or the ribs it is conservative to go for a value of 4.
        For any other information please see source 1.

        .. image:: ../_static/buckle_coef.png
            :width: 300
        
        **Future improvements**
        1.  compute the proper buckling coefficient in real time using the sheet aspect ratio (or just check for aspect ratio's smaller than 1. These seem to be the most relevant to catch)
        2. A plasticity factor could be implemented (see source 1, equation C5.2)

        **Bibliography**

        1. Chapter C5.2, Bruhn, Analysis & Design of Flight Vehicle Structures

        :param wingbox: Ideal wingbox class which is utilized to create constraints per panel 
        :type wingbox: IdealWingbox
        :param material_struct: Data structure containing all material propertie
        :type material_struct: Material
        :param len_to_rib: The  distance to the next rib
        :type len_to_rib: float
        :return: The critical buckling stress due to compression
        :rtype: float
        """    
        kc = 4 # buckling coefficient (currently very conservative but should be computed in real time using)
        t_arr = np.array([i.t_pnl for i in self.pnl_lst])  # thickness array of all the panels
        b = np.array([min(i.length(), self.len_to_rib) for i in self.pnl_lst])
        res = kc*np.pi**2*self.material_struct.young_modulus/(12*(1 - self.material_struct.poisson ** 2)) * (t_arr/b) ** 2
        # res[self.tens_pnl_idx] = 1
        return res


    def _crit_instability_shear(self) -> list:#TODO
        r""" 
        Compute the elastic instability of a flat sheet in shear for each panel in the idealized wingbox using  
        the equation shown below. Very similar for the case in compression (see :func:`crit_instability_compr`) except for the shear buckling coefficient.


        .. math::
            \sigma_{cr} = k_b  \frac{pi^2 E}{12(1 - \nu)} \left(\frac{t_{sk}}{b}\right)^2

        Where b is the short dimension of plate or loaded edge. For :math:`K_b`, the shear buckling coefficient, 
        the figure below was used to make a polynomial fit of the 3rd degree. The dataset used was as follows:

        x = [1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ] 
        y = [9.5, 7.2, 6.4, 6, 5.8, 5.9, 5.8, 5.6, 5.4]


        The interpolator was created in the initilisation of the class. For any other information please see source 1.

        .. image:: ../_static/shear_buck_coef.png
            :width: 300
        
        **Future improvements**
        1. A plasticity factor could be implemented (see source 1, equation C5.2)

        **Bibliography**

        1. Chapter C5.7, Bruhn, Analysis & Design of Flight Vehicle Structures

        :param wingbox: Ideal wingbox class which is utilized to create constraints per panel 
        :type wingbox: IdealWingbox
        :param material_struct: Data structure containing all material propertie
        :type material_struct: Material
        :param len_to_rib: The length to the next rib, thus the length to the next simply supported edge in the spanwise direction.
        :type len_to_rib: float
        :return: The critical buckling stress due to compression
        :rtype: float
        """    
        t_arr = np.array([i.t_pnl for i in self.pnl_lst])  # thickness array of all the panels

        # For the length of each panel we must look at the stringers 

        asp_lst = [] # Aspect ratio list
        b_lst = []
        for pnl in self.pnl_lst:
            len_pnl: float = pnl.length()
            # Make sure to divide by the shortest side
            if self.len_to_rib < len_pnl:
                asp_lst.append(len_pnl/self.len_to_rib)
                b_lst.append(self.len_to_rib)
            else: 
                asp_lst.append(self.len_to_rib/len_pnl)
                b_lst.append(len_pnl)

        asp_lst = np.array(asp_lst)
        b_lst = np.array(b_lst)

        # Compute the shear buckling coefficient using poly fit described in the docstring
        kb_vec: np.ndarray =  self._kb_interp(asp_lst)

        res = kb_vec* np.pi ** 2 * self.material_struct.young_modulus/(12*(1 - self.material_struct.poisson**2)) * (t_arr/ b_lst)**2
        return res

    # def flange_buckling(t_st, w_st):#TODO
    #     buck = 2 * np.pi ** 2 * material_struct.young_modulus / (12 * (1 - material_struct.poisson ** 2)) * (t_st / w_st) ** 2
    #     return buck


    # def web_buckling(t_st, h_st):#TODO
    #     buck = 4 * np.pi ** 2 *material.young_modulus / (12 * (1 -material.poisson ** 2)) * (t_st / h_st) ** 2
    #     return buck0


    # def global_buckling( h_st, t_st, t):#TODO
    #     # n = n_st(c_r, b_st)
    #     tsmr = (t *pitch_str + t_st *wing.n_str * (h_st - t)) /pitch_str
    #     return 4 * np.pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson ** 2)) * (tsmr / self.pitch_str) ** 2


    # def shear_buckling(self,t_sk):#TODO
    #     buck = 5.35 * pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson)) * (t_sk / self.pitch_str) ** 2
    #     return buck



    def interaction_curve_constr(self):
        r""" The following function ensures the panel remains below the interaction curve of a composite panel
        under combined compression and shear forces. This function is designed to be used with the :func:`SectionOptimization._get_constraint_vector` however it
        can also be used to check this specific constraints for any given design. The following equation is used for the 
        interaction curve which has been rewritten from equation 6.38, page 144 in source [1]:

        .. math::
            -\frac{N_x}{N_{x,crit}} - \left(\frac{N_{xy}}{N_{xy,crit}}\right) + 1 > 0



        **Bibliography**
        [1] Kassapoglou. Design and Analysis of Composite Structures. 2nd Edition. John Wiley & Sons Ltd, 2013.

        :param wingbox: _description_
        :type wingbox: IdealWingbox
        :param material_struct: _description_
        :type material_struct: Material
        :param len_to_rib: _description_
        :type len_to_rib: float
        :return: _description_
        :rtype: _type_
        """    

        
        area_pnl: list = [pnl.t_pnl*pnl.length() for pnl in self.pnl_lst]
        Nx_crit = self._crit_instability_compr()*area_pnl
        Nxy_crit = self._crit_instability_shear()*area_pnl

        Nx: list = np.abs([min(pnl.b1.sigma*pnl.b1.A, pnl.b2.sigma*pnl.b2.A) for pnl in self.pnl_lst]) # Take the maximum of the two booms
        Nxy: list = np.abs([pnl.q_tot*pnl.length() for pnl in self.pnl_lst])

        interaction_constr = -Nx/ Nx_crit - (Nxy/Nxy_crit) ** 2 + 1
        interaction_constr[self.tens_pnl_idx] = 1


        return interaction_constr


    # def column_st(self, h_st, w_st, t_st, t_sk):#
    #     #Lnew=new_L(b,L)
    #     Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
    #     i= pi ** 2 * self.material.young_modulus * Ist / (2*w_st* self.rib_pitch ** 2)#TODO IF HE FAILS REPLACE SPAN WITH RIB PITCH
    #     return i


    # def f_ult(self, h_st,w_st,t_st,t_sk,y):#TODO
    #     A_st = self.get_area_str(h_st,w_st,t_st)
    #     # n=n_st(c_r,b_st)
    #     A=self.wing.n_str*A_st+self.width_wingbox*self.chord(y)*t_sk
    #     f_uts=self.sigma_uts*A
    #     return f_uts


    # def buckling_constr(self, x):
    #     buck = self.buckling(x)
    #     return -1*(buck - 1)


    # def global_local(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     glob = self.global_buckling(h_st, t_st, t_sk)
    #     loc = self.local_buckling(t_sk)
    #     diff = glob - loc
    #     return diff


    # def local_column(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     col = self.column_st(h_st,w_st,t_st, t_sk)
    #     loc = self.local_buckling(t_sk)
    #     # print("col=",col/1e6)
    #     # print("loc=",loc/1e6)
    #     diff = col - loc
    #     return diff


    # def flange_loc_loc(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     flange = self.flange_buckling(t_st,w_st)
    #     loc = self.local_buckling(t_sk)
    #     diff = flange - loc
    #     return diff


    # def web_flange(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     web = self.web_buckling(t_st, h_st)
    #     loc = self.local_buckling(t_sk)
    #     diff =web-loc
    #     return diff


    def von_Mises(self):
        r""" THe following constraint implemetns the von mises failure criterion which is defined as follows for the case where only a direct stress
        in the y axis occurs and one shear stress is present.

        .. math::
             \sigma_y & \geq  \sigma_v \\
              \sigma_y  - \sqrt{\sigma_{11}^2 + 3\tau^2} & \geq  0 \\


        :return: _description_
        :rtype: _type_
        """        

        shear_arr = np.array([i.tau for i in  self.pnl_lst])
        direct_stress_arr = np.array([(i.b1.sigma + i.b2.sigma)/2 for i in  self.pnl_lst])
        return self.material_struct.sigma_yield - np.sqrt(direct_stress_arr**2 + 3*shear_arr**2)


    # def crippling(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     A = self.get_area_str(h_st,w_st,t_st)
    #     col = self.column_st( h_st,w_st,t_st,t_sk)
    #     crip = t_st * self.material.beta_crippling * self.material.sigma_yield* ((self.material.g_crippling * t_st ** 2 / A) * np.sqrt(self.material.young_modulus / self.material.sigma_yield)) ** self.m_crip
    #     return crip

    # #----OWN CONSTRAINTS-----
    # def str_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
    #     i= pi ** 2 * self.material.young_modulus * Ist / (self.rib_pitch ** 2)#TODO IF HE FAILS REPLACE SPAN WITH RIB PITCH
    #     i_sigma = (i/self.get_area_str(h_st,w_st,t_st))#convert to stress
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/(i_sigma) - 1)

    # def f_ult_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.material.sigma_ultimate - 1)
    # def flange_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.flange_buckling(t_st,w_st) - 1)

    # def web_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.web_buckling(t_st,h_st) - 1)

    # def global_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.global_buckling(h_st,t_st,t_sk) - 1)

 



def class2_wing_mass(vtol, flight_perf, wing ):
        """ Returns the structural weight of both wings 

        :param vtol: VTOL data structure
        :type vtol: VTOL
        :param flight_perf: FlightPerformance data structure
        :type flight_perf: FlightPerformance
        :param wing: SingleWing datastructure
        :type wing: SingleWing
        :return: Mass of both wings
        :rtype: float
        """        
        S_ft = wing.surface*1/const.foot**2
        mtow_lbs = 1/const.pound * vtol.mtom
        wing.mass= 0.04674*(mtow_lbs**0.397)*(S_ft**0.36)*(flight_perf.n_ult**0.397)*(wing.aspect_ratio**1.712)*const.pound
        return wing.mass


def class2_fuselage_mass(vtol, flight_perf, fuselage):
        """ Returns the mass of the fuselage

        :param vtol: VTOL data structure, requires: mtom
        :type vtol: VTOL
        :param flight_perf: FlightPerformance data structure
        :type flight_perf: FlightPerformance
        :param fuselage: Fuselage data structure
        :type fuselage: Fuselage
        :return: Fuselage mass
        :rtype: float
        """        
        mtow_lbs = 1/const.pound * vtol.mtom
        lf_ft, lf = fuselage.length_fuselage*1/const.foot, fuselage.length_fuselage

        nult = flight_perf.n_ult # ultimate load factor
        wf_ft = fuselage.width_fuselage*1/const.foot # width fuselage [ft]
        hf_ft = fuselage.height_fuselage*1/const.foot # height fuselage [ft]
        Vc_kts = flight_perf.v_cruise*1/const.foot # design cruise speed [kts]

        fweigh_USAF = 200*((mtow_lbs*nult/10**5)**0.286*(lf_ft/10)**0.857*((wf_ft + hf_ft)/10)*(Vc_kts/100)**0.338)**1.1
        fuselage.mass= fweigh_USAF*const.pound
        return fuselage.mass
