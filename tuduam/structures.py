import numpy  as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import scipy.constants as const
from .data_structures import Wingbox
from math import ceil
from shapely.geometry import Polygon, Point
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
    """_summary_
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
    
    @property
    def length(self) -> float:
        """ Length of the panel based on the coordinates of the boom. Boom center is used as the 
        assumption is that the booms are infitestimally small.

        :return: The length of the panel
        :rtype: float
        """        
        return np.sqrt((self.b2.x - self.b1.x)**2 + (self.b2.y - self.b1.y)**2)

    def set_b1_to_b2_vector(self) -> tuple:
        try:
            x_comp = (self.b2.x - self.b1.x)/self.length
            y_comp = (self.b2.y - self.b1.y)/self.length
        except AttributeError as err:
            raise err("The boom instance has not been assigned yet or is missing the attribute x and y")
        self.dir_vec = (x_comp, y_comp)

    def set_b2_to_b1_vector(self) -> tuple:
        try:
            x_comp = (self.b1.x - self.b2.x)/self.length
            y_comp = (self.b1.y - self.b2.y)/self.length
        except AttributeError as err:
            raise err("The boom instance has not been assigned yet or is missing the attribute x and y")
        self.dir_vec =  (x_comp, y_comp)

class IdealWingbox():
    """ A class representing an idealized wingbox, containing methods to perform computations
    on that instants and some accessed methods. It is not recommended to use as a standalone tool but should 
    be using in conjunction with the discretize_airfoil function.

    Assumptions
    --------------------------------------------------------------------------------
    - The x datum of the coordinate system should be attached to the leading edge of the 
    wingbox
    - Some methods such as the read_cell_area expect the first and last to begin and end on a vertex.


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


    def _get_polygon_of_cells(self, validate=False) -> List[float]:
        """ Compute the area of each cell with the help of the shapely.geometry.Polygon class. The function expects a fully loaded airfoil to be in the 
        class using the idealized_airfoil function. Errenous results or an error will be given in case this is not the case! When using this function for the first time
        with a new airfoil it is advised to run it once with validate= True to see if the resulting areas are trustworthy. This will
        show you n plots of the cell polygon where n is the amount of cells.

        :param validate: When True will show the 3 plots described above, defaults to False
        :type validate: bool, optional
        :param validate:
        :return: A list whic is m long where m is the amount of cells. Each element is the area
        of the respective cell.
        :rtype: List[float]
        """        
        bm_per_cell_lst =  []
        polygon_lst =  []

        # Get all the booms per cell
        spar_loc_arr = np.insert(self.wingbox_struct.spar_loc_nondim, 0,0)*self.chord
        for idx, spar_loc in enumerate(spar_loc_arr):
            if idx != len(spar_loc_arr) - 1:
                bm_per_cell_lst.append([i for i in self.boom_dict.values() if (spar_loc <= i.x <= spar_loc_arr[idx + 1])])
            else:
                bm_per_cell_lst.append([i for i in self.boom_dict.values() if  i.x >= spar_loc])
        
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

    def read_cell_areas(self):
        polygon_lst = self._get_polygon_of_cells()
        return [i.area for i in polygon_lst]

    def _compute_boom_areas(self, chord) -> None:
        """ Function that creates all boom areas, program assumes a fully functional panel and boom
        dictionary where all values have the full classes assigned. Function can be used by user manually
        but it is generall advised to use the discretize_airfoil function to create a wingbox.

        Assumptions
        --------------------------
        - The idealizatin only takes into account a force in the vertical direction (that is tip-deflection path)
        - The area of the stringers is smeared acrossed all skin booms in the respective cell
        """        

        # Find stringer area to add per cell
        str_contrib = []
        for idx, n_str in enumerate(self.wingbox_struct.str_cell):
            str_contrib.append(n_str*self.wingbox_struct.area_str/len(self._read_skin_panels_per_cell[idx]))

        # Define absolute spar location for use within the loop
        spar_loc_abs = np.array(self.wingbox_struct.spar_loc_nondim)*chord
        # Per boom find all the panel in which the boom is found and add skin contribution
        for boom in self.boom_dict.values():
            boom_area = boom.A =  0
            # Retrieve all panel where boom is a part of
            pnl_lst = [pnl for pnl in self.panel_dict.values() if pnl.bid1 == boom.bid or pnl.bid2 == boom.bid]
            for pnl in pnl_lst:
                if boom.bid == pnl.bid1:
                    boom_area += pnl.t_pnl*pnl.length/6*(2 + (pnl.b2.y - self.y_centroid )/(boom.y - self.y_centroid))
                else:
                    boom_area += pnl.t_pnl*pnl.length/6*(2 + (pnl.b1.y - self.y_centroid)/(boom.y - self.y_centroid))

            boom.A = boom_area

            if len(str_contrib) != 0 and not any(np.isclose(boom.x, spar_loc_abs )):
                boom.A += str_contrib[boom.get_cell_idx(self.wingbox_struct, chord )]

    def stress_analysis(self, intern_shear:float, internal_mz:float, shear_mod:float, validate=False) ->  Tuple[float, dict]:
        """ 
        Perform stress analysis on  a wingbox section 

        List of assumptions (Most made in Megson, some for simplificatoin of code)
        ---------------------------------------
        - The effect of taper are not included see 21.2 (See megson) TODO: future implementation
        - Lift acts through the shear centre (no torque is created) TODO: future implementation
        - Stresses due to drag are not considered. 



        :param intern_shear: _description_
        :type intern_shear: float
        :param internal_mz: _description_
        :type internal_mz: float
        :param shear_mod: _description_
        :type shear_mod: float
        :return: _description_
        :rtype: Tuple[float, dict]
        """    

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
                    cond2 = pnl.b2.y >= self.y_centroid and pnl.b1.y >= self.y_centroid
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
                        curr_pnl.set_b2_to_b1_vector()
                # If b2 is attached to another panel and q_basic is not attached yet continue from this panel
                elif len(b2_lst) == 1 and b2_lst[0].q_basic == None:
                    # Check if it was to boom 1
                    if curr_pnl.b2 == b2_lst[0].b2:
                        curr_pnl = b2_lst[0]
                        q_basic += shear_const*curr_pnl.b2.A*(curr_pnl.b2.y - self.y_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b2_to_b1_vector()
                    # if not boom 1 then it was boom 2
                    else:
                        curr_pnl = b2_lst[0]
                        q_basic += shear_const*curr_pnl.b1.A*(curr_pnl.b1.y - self.y_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b1_to_b2_vector()
                else: 
                    raise Exception("No connecting panel was found")


        area_lst = self.read_cell_areas()





    def plot_direct_stresses(self) -> None:
            plt.figure(figsize=(10,1))
            x_lst = np.array([i.x for i in self.boom_dict.values()])
            y_lst = np.array([i.y for i in self.boom_dict.values()])
            stress_arr = np.array([i.sigma for i in self.boom_dict.values()])
            norm = plt.Normalize(stress_arr.min(), stress_arr.max())
            plt.scatter(x_lst, y_lst, c=stress_arr, cmap='viridis', norm=norm)
            plt.colorbar(label='Direct Stress ')

            y_arr  = [i.y for i in self.boom_dict.values()]
            y_max = np.max(y_arr)
            y_min = np.min(y_arr)
            plt.ylim([ y_min - 0.1,y_max + 0.1])
            plt.show()


    def plot_geometry(self) -> None:
        plt.figure(figsize=(10,1))
        for key, panel in self.panel_dict.items():
            x = [panel.b1.x, panel.b2.x]
            y = [panel.b1.y, panel.b2.y]
            plt.plot(x,y, ">-")
        y_arr  = [i.y for i in self.boom_dict.values()]
        y_max = np.max(y_arr)
        y_min = np.min(y_arr)
        plt.ylim([ y_min - 0.1,y_max + 0.1])
        plt.show()


def read_coord(path_coord:str) -> np.ndarray:
    """ Returns an  m x 2 array of the airfoil coordinates based on a Selig formatted dat file.

    :return: An m x 2 array with the airfoil coordinates where x goes top trailing edge to top leading edge and then back to 
    lower trailing edge. I.e it keeps the Selig format
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
    """ Compute the nondimensional x and y centroid based on the coordinate file of an airfoil.
    The centroids are computing assuming the following:
     
     Assumptions
     -----------------------------
     - It is only based on the skin, i.e  the spar webs and stringers are ignored. Additionally the different thickness of the skin are not taken into account 
     The implication being that the x centroid should be at x/c = 0.5. Unless there was a bias in the sampling points

    Future improvement
     -----------------------------
     - Take into account the spar webs for a better x centroid. However irrelevant for now as we only
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
    """ Create a discretized airfoil according to the principles of Megson based on a path to a txt file containing
    the non-dimensional coordinates of the airfoil, the corresonding chord and the wingbox data structure fully filled in.

    Assumptions
    -------------------------------------
    - Airfoil is idealized according Megson ch. 20
    - The stringers are modeled by equally smearing the total area of the combined 
    stringers in a certain cell  to all booms attached to the skin in that cell 
    - The ratio of $\frac{\sigma_1}{\sigma_2}$  required for the boom size based on the skin is 
    determined by the ratio of their y positin thus $\frac{y_1}{y_2}$.

    General Procedure
    -----------------------------------------------
    1. Create a spline of the top and bottom airfoil
    2. Create array along which to sample this spline to create the booms, creating specific samples for the spar positions
    3. Move over top surface creating booms and panel as we go
    4. Do the same for the bottom surface moving in a circle like motion
    5. Move over all the spars and create booms and panels as we go.
    6. Iterate over all booms and add skin contribution and stringer contribution to all their areas


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
    assert wingbox_struct.booms_spar > 4, "Length of spar_loc should be equal to the amount of cells - 1"

    top_interp, bot_interp = spline_airfoil_coord(path_coord, chord)
    x_centr, y_centr = get_centroids(path_coord)

    wingbox = IdealWingbox(wingbox_struct, chord)
    wingbox.x_centroid =  x_centr*chord
    wingbox.y_centroid =  y_centr*chord


    x_boom_loc = np.linspace(0, chord, ceil(wingbox_struct.booms_sk/2 + 2))

    # Put booms at the spar locations
    for spar_loc in wingbox_struct.spar_loc_nondim:
        spar_loc *= chord
        idx = np.argmin(np.abs(x_boom_loc - spar_loc))
        x_boom_loc[idx] = spar_loc

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

        y_locations = np.delete(np.linspace(upper_b.y, lower_b.y, wingbox_struct.booms_spar), [0,-1]) # delete padding as they already have booms

        # Loop over the spar to create the booms
        for idx, y_loc in enumerate(y_locations, start=0):
            # Create boom
            boom = Boom()
            boom.bid = bid
            boom.x =  spar_loc
            boom.y = y_loc
            if boom.bid not in boom_dict.keys():
                boom_dict[boom.bid] = boom
            else:
                raise RuntimeError(f"Boom id {boom.bid} already exists in variable boom dictionary")

            if idx == 0:
                pnl = IdealPanel()
                pnl.pid = pid
                pnl.bid1 = upper_b.bid 
                pnl.bid2 = bid
                pnl.t_pnl = wingbox_struct.t_sp
                pnl.b1 =  boom_dict[pnl.bid1]
                pnl.b2 =  boom_dict[pnl.bid2]
                if pnl.pid not in panel_dict.keys():
                    panel_dict[pid] = pnl
                else:
                    raise RuntimeError(f"Boom id {boom.bid} already exists in variable boom dictionary")
            else: 
                pnl = IdealPanel()
                pnl.pid = pid
                pnl.bid1 = bid - 1
                pnl.bid2 = bid
                pnl.t_pnl = wingbox_struct.t_sp
                pnl.b1 =  boom_dict[pnl.bid1]
                pnl.b2 =  boom_dict[pnl.bid2]

                if pnl.pid not in panel_dict.keys():
                    panel_dict[pid] = pnl
                else:
                    raise RuntimeError(f"Boom id {boom.bid} already exists in variable boom dictionary")

            bid += 1
            pid += 1

        # Connect the last panel to the lower boom
        pnl = IdealPanel()
        pnl.pid = pid
        pnl.bid1 = bid - 1
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
