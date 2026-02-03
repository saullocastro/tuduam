""" This module contains the tools to describe a wingbox and to perform the necessary analyses on them.
"""


import random
from typing import Tuple, List
from warnings import warn

import numpy  as np
from scipy.interpolate import CubicSpline
from shapely.geometry import Polygon
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
from ..data_structures import *




class Boom:
    """
    A class to represent an idealized boom.

    .. note::

        All of the parameters are optional as in this library the attributes
        have been allocated dynamically. But if this class is used you could also simply load them
        when instantiating them.



    Parameters
    ----------
    bid : int, optional
        Boom ID.
    A : int, optional
        Boom area.
    x : float, optional
        Boom x position.
    y : float, optional
        Boom y position.
    sigma : float, optional
        The direct stress the boom experiences.

    Attributes
    ----------
    bid : int or None
        Boom ID.
    A : float or None
        Boom area.
    x : float or None
        Boom x position.
    y : float or None
        Boom y position.
    sigma : float or None
        The direct stress the boom experiences.
    """

    def __init__(self) -> None:
        self.bid: int | None = None  # Boom ID
        self.A: float | None = None  # Boom area
        self.x: float | None = None  # Boom x position
        self.y: float | None = None  # Boom y position
        self.sigma: float | None = None  # Direct stress the boom experiences

    def get_cell_idx(self, wingbox_struct: Wingbox, chord: float) -> int:
        """
        Returns the cell index of where the panel is located for the specified wingbox structure.

        Parameters
        ----------
        wingbox_struct : Wingbox
            The wingbox data structure containing the locations of the spars.
        chord : float
            The local chord of the wing section.

        Returns
        -------
        int
            The cell index of where the panel is located.
        """

        # Get index of the cell
        cell_idx = np.asarray(self.x >= np.insert(wingbox_struct.spar_loc_nondim, 0, 0)*chord)  # type: ignore # type: ignore
        if  not any(cell_idx):
            cell_idx = 0
        else:
            cell_idx = cell_idx.nonzero()[0][-1]
        return cell_idx


class IdealPanel:
    """
    A class representing a panel in an idealized wingbox.

    Parameters
    ----------
    pid : int
        Panel ID.
    bid1 : int
        The boom ID of its corresponding first boom.
    bid2 : int
        The boom ID of its corresponding second boom.
    b1 : Boom
        Instance of one of the two :class:`Boom` connecting the panels.
    b2 : Boom
        Instance of the other :class:`Boom` connecting the panels.
    t_pnl : float
        Panel thickness.
    bid : int
        Boom ID.

    Attributes
    ----------
    pid : int or None
        Panel ID.
    bid1 : int or None
        The boom ID of its corresponding first boom.
    bid2 : int or None
        The boom ID of its corresponding second boom.
    b1 : Boom or None
        Instance of one of the two :class:`Boom` connecting the panels.
    b2 : Boom or None
        Instance of the other :class:`Boom` connecting the panels.
    t_pnl : float or None
        Panel thickness.
    q_basic : float or None
        Basic shear flow.
    q_tot : float or None
        Total shear flow.
    tau : float or None
        Shear stress.
    dir_vec : float or None
        Direction vector.
    """

    def __init__(self):
        self.pid = None # Panel ID
        self.bid1 = None  # The boom id of its corresponding first boom id
        self.bid2 = None # The boom id of its corresponding second boom id
        self.b1: Boom | None = None
        self.b2: Boom | None  = None
        self.t_pnl: float | None = None
        self.q_basic: float | None = None
        self.q_tot: float | None = None
        self.tau: float | None = None
        self.dir_vec: float | None = None

    def get_cell_idx(self, wingbox_struct: Wingbox, chord: float) -> int:
        """
        Returns the cell index of where the panel is located.

        Parameters
        ----------
        wingbox_struct : Wingbox
            The wingbox data structure containing the locations of the spars.
        chord : float
            The local chord of the wing section.

        Returns
        -------
        int
            The cell index of where the panel is located.
        """


        cell_idx = np.asarray((self.b1.x + self.b2.x)/2 >= np.insert(wingbox_struct.spar_loc_nondim, 0, 0)*chord) # type: ignore # Get index of the cell
        if  not any(cell_idx):
            cell_idx = 0
        else:
            cell_idx = cell_idx.nonzero()[0][-1]
        return cell_idx

    def length(self) -> float:
        """
        Length of the panel based on the coordinates of the boom. Boom center is used as the
        assumption is that the booms are infinitesimally small.

        Returns
        -------
        float
            The length of the panel.
        """

        return np.sqrt((self.b2.x - self.b1.x)**2 + (self.b2.y - self.b1.y)**2) # type: ignore

    def set_b1_to_b2_vector(self) -> tuple:
        """
        The following functions sets the attribute :attr:`dir_vec` from boom 1 to boom 2.

        Returns
        -------
        tuple
            The unit vector from b1 to b2.

        Raises
        ------
        AttributeError
            If boom 1 or 2 has not been assigned yet.
        """
        try:
            x_comp = (self.b2.x - self.b1.x)/self.length() # type: ignore
            y_comp = (self.b2.y - self.b1.y)/self.length() # type: ignore
        except AttributeError as err:
            raise err("The boom instance has not been assigned yet or is missing the attribute x and y")
        self.dir_vec = [x_comp, y_comp] # type: ignore

    def set_b2_to_b1_vector(self) -> tuple:
        """
        The following functions sets the attribute :attr:`dir_vec` from boom 2 to boom 1.

        Returns
        -------
        tuple
            The unit vector from b2 to b1.

        Raises
        ------
        AttributeError
            If boom 1 or 2 has not been assigned yet.
        """
        try:
            x_comp = (self.b1.x - self.b2.x)/self.length()
            y_comp = (self.b1.y - self.b2.y)/self.length()
        except AttributeError as err:
            raise err("The boom instance has not been assigned yet or is missing the attribute x and y")
        self.dir_vec =  [x_comp, y_comp]

class IdealWingbox():
    """
    A class representing an idealized wingbox, containing methods to perform computations
    on that instance and some accessed methods. It is strongly advised not to use this class without the :func:`discretize_airfoil` function,
    as all functions rely on the :class:`Wingbox` to be properly loaded with geometry.

    .. admonition:: Assumptions

        1. The x datum of the coordinate system should be attached to the leading edge of the wingbox.
        2. Some methods, such as read_cell_area, expect the first and last cells to begin and end on a vertex.

    Parameters
    ----------
    wingbox : Wingbox
        The wingbox structure.
    chord : float
        The chord length of the wingbox.

    Attributes
    ----------
    wingbox_struct : Wingbox
        The wingbox structure.
    area_str : float
        The area of the stringer.
    chord : float
        The chord length of the wingbox.
    x_centroid : float or None
        The x-coordinate of the centroid, datum attached to leading edge.
    y_centroid : float or None
        The y-coordinate of the centroid.
    panel_dict : dict
        Dictionary containing panel information.
    boom_dict : dict
        Dictionary containing boom information.
    area_lst : list or None
        List of cell areas.
    centroid_lst : list or None
        List of cell centroids.
    Ixx : float or None
        Moment of inertia about the x-axis.
    Ixy : float or None
        Product of inertia.
    Iyy : float or None
        Moment of inertia about the y-axis.
    """

    def __init__(self, wingbox: Wingbox, chord:float) -> None:
        self.wingbox_struct = wingbox
        t_st = self.wingbox_struct.t_st
        w_st = self.wingbox_struct.w_st
        h_st = self.wingbox_struct.h_st
        # Boolean expressoin below check if any of the stringer geometry were specified
        bool_expr_str = (t_st is not None) or (w_st is not None) or (h_st is not None)
        if self.wingbox_struct.area_str is not None and (bool_expr_str):
            raise UserWarning("Both stringer area and stringer geometry were specified")
        elif bool_expr_str:
            try:
                self.area_str = self._compute_area_z_str(t_st, w_st, h_st)
            except AttributeError as err:
                raise UserWarning(f"Not all stringer geometry was specified, resultingin the following error: {err} please refer to API docs")

        elif self.wingbox_struct.area_str is not None:
            self.area_str = self.wingbox_struct.area_str

        else:
            raise UserWarning("No stringer area nor stringer geometry were specified")

        self.chord = chord
        self.x_centroid = None # datum attached to leading edge
        self.y_centroid = None
        self.panel_dict = {}
        self.boom_dict = {}
        self.area_lst = None # List of cell areas
        self.centroid_lst = None # List of cell centroid
        self.Ixx = None
        self.Ixy = None
        self.Iyy = None
        pass

    def _compute_area_z_str(self, t_st, w_st, h_st) -> float:
        """
        Computes the area of a Z-stringer.

        Parameters
        ----------
        t_st : float
            The thickness of the stringer.
        w_st : float
            The width of the stringer.
        h_st : float
            The height of the stringer.

        Returns
        -------
        float
            The computed area of the Z-stringer.
        """

        if t_st > h_st:
            warn("The thickness of stringer is larger than than the height of the stringer perhaps resulting in a negative area.")

        if t_st  > w_st:
            warn("The thickness of stringer is larger than than the width of the stringer resulting in nonsensical geometries.")

        A_str = 2*w_st*t_st + (h_st - 2*t_st)*t_st

        if A_str < 0:
            warn("The stringer area is negative, please see previous error")

        return A_str

    def _set_Ixx(self):

        """
        Computes the moment of inertia about the x-axis.

        Returns
        -------
        float
            The moment of inertia about the x-axis.
        """

        Ixx = 0
        for boom in self.boom_dict.values():
            Ixx += boom.A*(boom.y - self.y_centroid)**2
        return Ixx

    def _set_Ixy(self):
        """
        Computes the product of inertia.

        Returns
        -------
        float
            The product of inertia.
        """

        Ixy = 0
        for boom in self.boom_dict.values():
            Ixy += boom.A*(boom.x - self.x_centroid)*(boom.y - self.y_centroid)
        return Ixy

    def _set_Iyy(self):
        """
        Computes the moment of inertia about the y-axis.

        Returns
        -------
        float
            The moment of inertia about the y-axis.
        """

        Iyy = 0
        for boom in self.boom_dict.values():
            Iyy += boom.A*(boom.x - self.x_centroid)**2
        return Iyy

    @property
    def _read_skin_panels_per_cell(self) -> List[int]:
        """
        Returns a list with the number of panels on the skin per cell, ignoring the panels
        which are part of one of the spars. This function requires a fully filled out boom and panel dictionary.

        Returns
        -------
        list
            An n x m 2D list where n is the number of cells and m is the number of panels (might not be identical for each cell).
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
        """
        Returns the total area of all the booms, including the addition of the skin thicknesses. This function is used
        for the optimization of a wingbox.

        Returns
        -------
        float
            The total area of all the booms combined.
        """


        tot_area = 0
        for boom in self.boom_dict.values():
            tot_area += boom.A
        return tot_area


    def get_polygon_cells(self, validate=False) -> List[Polygon]:
        """
        Compute the area of each cell with the help of the shapely.geometry.Polygon class. The function expects a fully loaded airfoil to be in the
        class using the idealized_airfoil function. Erroneous results or an error will be given in case this is not the case! When using this function for the first time
        with a new airfoil, it is advised to run it once with validate=True to see if the resulting areas are trustworthy. This will
        show you n plots of the cell polygon where n is the number of cells.

        .. admonition:: Assumptions

            1. Function is built for an object built with the discretize airfoil, that is cell 0 has a singular point as a leading edge, that is one point is the furthest ahead. The same goes for cell n but with the trailing edge.

        Parameters
        ----------
        validate : bool, optional
            When True will show the 3 plots described above, defaults to False.

        Returns
        -------
        List[Polygon]
            A list of n cells long where each element contains a (shapely) Polygon instance representing the corresponding cell.
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

        """
        Compute the area of each cell using the shapely.geometry.Polygon class.

        Parameters
        ----------
        validate : bool, optional
            When True, shows validation plots, defaults to False.

        Returns
        -------
        List[float]
            A list of areas for each cell.
        """

        polygon_lst = self.get_polygon_cells(validate)
        return [i.area for i in polygon_lst]

    def compute_direct_stress(self, boom: Boom, moment_x: float, moment_y: float):

        """
        Compute the direct stress at a given boom due to bending moments.

        Parameters
        ----------
        boom : Boom
            The boom at which to compute the direct stress.
        moment_x : float
            The bending moment around the x-axis.
        moment_y : float
            The bending moment around the y-axis.

        Returns
        -------
        float
            The direct stress at the given boom.
        """

        Ixx = self.Ixx
        Ixy = self.Ixy
        Iyy = self.Iyy
        return ((moment_y*Ixx - moment_x*Ixy)/(Ixx*Iyy -Ixy**2)*boom.x +
                (moment_x*Iyy  - moment_y*Ixy)/(Ixx*Iyy -Ixy**2)*boom.y)



    def _compute_boom_areas(self, chord) -> None:
        """
        Function that creates all boom areas. The program assumes a fully functional panel and boom
        dictionary where all values have the full classes assigned. This function can be used manually
        by the user, but it is generally advised to use the discretize_airfoil function to create a wingbox.

        .. admonition:: Assumptions

            1. The idealization only takes into account a force in the vertical direction (that is, tip-deflection path).

        Parameters
        ----------
        chord : float
            The chord length of the wingbox.

        Returns
        -------
        None
            Returns none as the boom areas are directly assigned to the :class:`Boom` instances.
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
                boom.A += self.area_str
            if boom.A < 0:
                warn("Negative boom areas encountered this is currently a bug, temporary fix takes the absolute value")
                boom.A = np.abs(boom.A)


    def load_new_gauge(self, t_sk_cell: list, t_sp: float, t_st: float, w_st: float, h_st: float) -> None:
        """
        This function allows you to change the thickness and hence your boom areas of the wingbox whilst
        maintaining the shape. This is useful for any optimization as you do not have to call the entire discretize function.

        Parameters
        ----------
        t_sk_cell : list
            List of skin thicknesses for each cell.
        t_sp : float
            Thickness of the spar.
        t_st : float
            Thickness of the stringer.
        w_st : float
            Width of the stringer.
        h_st : float
            Height of the stringer.
        """


        # Load new data in data structure
        self.wingbox_struct.t_sk_cell = t_sk_cell
        self.wingbox_struct.t_sp = t_sp
        self.wingbox_struct.t_st = t_st
        self.wingbox_struct.w_st = w_st
        self.wingbox_struct.h_st = h_st

        # Required for backwards compatibility
        self.t_st = t_st
        self.w_st = w_st
        self.h_st = h_st

        # First compute new stringer area
        self.area_str = self._compute_area_z_str(t_st, w_st, h_st)

        for pnl in self.panel_dict.values():
            pnl: IdealPanel = pnl
            # Check if it is a spar
            if pnl.b1.x == pnl.b2.x:
                # If spar change thickness
                pnl.t_pnl = t_sp
            # Else if it was a panel
            else:
                idx = pnl.get_cell_idx(self.wingbox_struct, self.chord) # Find which cell  panel is in
                pnl.t_pnl = t_sk_cell[idx] # Assign new thickness

        self._compute_boom_areas(self.chord)
        self.Ixx = self._set_Ixx()
        self.Iyy = self._set_Iyy()
        self.Ixy = self._set_Ixy()

    def stress_analysis(self,  shear_y: float, shear_x: float,   moment_y: float, moment_x: float, shear_y_appl : float, shear_mod: float, validate=False) ->  Tuple[list,list]:
        """
        Perform stress analysis on a wingbox section based on the internal shear loads and moments. All data is stored within
        the :class:`IdealPanel` and :class:`Boom` classes.


        **Sign convention forces and coordinate system**

        .. image:: ../_static/sign_convention_forces.png
            :width: 300

        The sign convention above is applied for the forces. Please consider that Figure 16.9 shows positive directions and senses for the above loads and moments applied externally to
        a beam and also the positive directions of the components of displacement u, v, and w
        of any point in the beam cross-section parallel to the x, y, and z axes, respectively. If we refer internal forces and moments to that face of a section which is seen when
        viewed in the direction zO then, as shown in Fig. 16.10, positive internal forces and
        moments are in the same direction and sense as the externally applied loads whereas
        on the opposite face they form an opposing system. A further condition defining the signs of the bending moments Mx and My is that they are
        positive when they induce tension in the positive xy quadrant of the beam cross-section. Finally, the beam seen in Figure 16.10 is also
        immediately the coordinate system used. Where the aircraft flies
        in the x direction, thus analyzing the right wing structure.




        .. admonition:: Assumptoins

            1. The effect of taper is not included, see 21.2 (See Megson). TODO: future implementation
            2. Lift acts through the shear centre (no torque is created). TODO: future implementation
            3. Stresses due to drag are not considered.


        **Sources**


        [1] Section 16.2.2, T.H.G Megson, Aircraft Structures For Engineering Students, 4th Edition

        Parameters
        ----------
        shear_y : float
            The internal shear force in the y-direction.
        shear_x : float
            The internal shear force in the x-direction.
        moment_y : float
            The internal moment around the y-axis.
        moment_x : float
            The internal moment around the x-axis.
        shear_y_appl: float
            The location where they y force is applied
        shear_mod : float
            The shear modulus.
        validate : bool, optional
            If True, validate the results, defaults to False.

        Returns
        -------
        Tuple[list, list]
            A tuple containing the results of the stress analysis.
        """

        # Ensure all shear flows are set to none because the function relies on it
        for pnl in self.panel_dict.values():
            pnl.q_basic = None
            pnl.q_total = None

        # First compute all the direct stresses
        for boom in self.boom_dict.values():
            boom.sigma = self.compute_direct_stress(boom, moment_x, moment_y)

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

        shear_const_y = -(shear_y*self.Iyy - shear_x*self.Ixy)/(self.Ixx*self.Iyy - self.Ixy) # define -Sy/Ixx which is used repeatdly
        shear_const_x = -(shear_x*self.Ixx - shear_y*self.Ixy)/(self.Ixx*self.Iyy - self.Ixy) # define -Sy/Ixx which is used repeatdly

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
                        q_basic += shear_const_y*curr_pnl.b1.A*(curr_pnl.b1.y - self.y_centroid) + shear_const_x*curr_pnl.b1.A*(curr_pnl.b1.x - self.x_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b1_to_b2_vector()
                    # if not boom 1 then it was boom 2
                    else:
                        curr_pnl = b1_lst[0]
                        q_basic += shear_const_y*curr_pnl.b2.A*(curr_pnl.b2.y - self.y_centroid) + shear_const_x*curr_pnl.b2.A*(curr_pnl.b2.x - self.x_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b2_to_b1_vector() # Set the direction in which the shear flow was defined
                # If b2 is attached to another panel and q_basic is not attached yet continue from this panel
                elif len(b2_lst) == 1 and b2_lst[0].q_basic == None:
                    # Check if it was to boom 1
                    if curr_pnl.b2 == b2_lst[0].b2:
                        curr_pnl = b2_lst[0]
                        q_basic += shear_const_y*curr_pnl.b2.A*(curr_pnl.b2.y - self.y_centroid) + shear_const_x*curr_pnl.b2.A*(curr_pnl.b2.x - self.x_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b2_to_b1_vector() # Set the direction in which the shear flow was defined
                    # if not boom 1 then it was boom 2
                    else:
                        curr_pnl = b2_lst[0]
                        q_basic += shear_const_y*curr_pnl.b1.A*(curr_pnl.b1.y - self.y_centroid) + shear_const_x*curr_pnl.b1.A*(curr_pnl.b1.x - self.x_centroid)
                        curr_pnl.q_basic =  q_basic
                        curr_pnl.set_b1_to_b2_vector()
                else:
                    raise Exception("No connecting panel was found")


        #=========================================================================
        # Now Compute the complementary shear flows and the twist per unit lengt
        #========================================================================

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
                A_arr[idx, n_cell] = 2*self.area_lst[idx]*shear_mod
                A_arr[idx, idx] = -1*np.sum([pnl.length()/pnl.t_pnl for pnl in cell])
                A_arr[idx, idx + 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_max)])
            elif  0 < idx < len(pnl_per_cell_lst2) - 1:
                x_min = np.min([i.b1.x for i in cell])
                x_max = np.max([i.b1.x for i in cell])
                A_arr[idx, n_cell] = 2*self.area_lst[idx]*shear_mod
                A_arr[idx, idx - 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_min) ])
                A_arr[idx, idx] = -1*np.sum([pnl.length()/pnl.t_pnl for pnl in cell])
                A_arr[idx, idx + 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_max)])
                pass
            elif idx == len(pnl_per_cell_lst2) - 1:
                x_min = np.min([i.b1.x for i in cell])
                A_arr[idx, n_cell] = 2*self.area_lst[idx]*shear_mod
                A_arr[idx, idx - 1] = np.sum([pnl.length()/pnl.t_pnl for pnl in cell if (pnl.b1.x == pnl.b2.x == x_min)])
                A_arr[idx, idx] = -1*np.sum([pnl.length()/pnl.t_pnl for pnl in cell])
            else:
                raise Exception(f"Something went wrong, more iterations were made then there were cells")

        # Set up eqautions for twist per unit legnth but in b array
        for idx, cell in enumerate(pnl_per_cell_lst2):
            b_ele = 0
            for pnl in cell:
                r_abs_vec = np.array([(pnl.b1.x + pnl.b2.x)/2 , (pnl.b1.y + pnl.b2.y)/2])
                r_rel_vec = r_abs_vec - self.centroid_lst[idx]
                if pnl.q_basic != 0:
                    sign = np.sign(np.cross(r_rel_vec, pnl.dir_vec))
                    b_ele += sign*pnl.q_basic*pnl.length()/pnl.t_pnl
                # When we have a panel that was not cut, they do not have a defined direction yet and it does not matter since magnitude is 0
                elif pnl.q_basic == 0:
                    b_ele += pnl.q_basic*pnl.length()/pnl.t_pnl
                else:
                    raise Exception(f"Line should not have been reached")
            b_arr[idx,0] = b_ele


        #------------------------------- Fill in the final equation, moment equivalence ------------------
        # Contribution from the complementary shear flows
        for idx, cell in enumerate(pnl_per_cell_lst2):
            A_arr[n_cell, idx] = 2*self.area_lst[idx]

        # Contribution to b_arr from basic shear flows and shear force itself
        sum = 0
        for pnl in self.panel_dict.values():
            if pnl.q_basic == 0:
                continue
            else:
                r_abs_vec = np.array([(pnl.b1.x + pnl.b2.x)/2 , (pnl.b1.y + pnl.b2.y)/2])
                moment = pnl.q_basic*np.cross(r_abs_vec, pnl.dir_vec)*pnl.length()
                sum += moment

        b_arr[n_cell, 0] = -1*sum + shear_y*shear_y_appl*self.chord + shear_x*self.y_centroid

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
                r_rel_vec = r_abs_vec - self.centroid_lst[idx]
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
        """
        Plots the direct stresses in the booms of the wingbox.

        This function creates a scatter plot of the direct stresses in the booms, with the color indicating the magnitude
        of the stress.

        Returns
        -------
        None
        """
        plt.figure(figsize=(10,1))
        x_lst = np.array([i.x for i in self.boom_dict.values()])
        y_lst = np.array([i.y for i in self.boom_dict.values()])
        stress_arr = np.array([i.sigma/1e6 for i in self.boom_dict.values()])
        hover_data = [f"stress = {i.sigma/1e6} Mpa" for i in self.boom_dict.values()]
        fig = px.scatter(x= x_lst, y= y_lst, color=stress_arr, title= "Direct stress")
        fig.update_traces(marker=dict(size=12,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                        selector=dict(mode='markers'))
        fig.show()



    def plot_shear_stress(self) -> None:
        """
        Plots the shear stress in the panels of the wingbox.

        This function creates a scatter plot of the shear stress in the panels, with the color indicating the magnitude
        of the stress.

        Returns
        -------
        None
        """
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
            col = [str(float(coli)) for coli in cmap(norm(stress))]
            print('DEBUG col', col)
            col = "rgb(" + ",".join(col[:-1]) + ")"

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
        """
        Plots the shear stress directions in the panels of the wingbox using a quiver plot.

        Parameters
        ----------
        scale : float, optional
            Scaling factor for the arrow length, defaults to 0.020.
        arrow_scale : float, optional
            Scaling factor for the arrow size, defaults to 0.4.

        Returns
        -------
        None
        """

        pnl_lst = [i for i in self.panel_dict.values()]

        x = [(i.b1.x + i.b2.x)/2 for i in pnl_lst]
        y = [(i.b1.y + i.b2.y)/2 for i in pnl_lst]
        u = list()
        v = list()

        for pnl in pnl_lst:
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
        """
        Plots the geometry of the wingbox, showing the discretized panels and booms.

        This function creates a scatter plot of the geometry of the wingbox, with hover text showing the boom ID and area.

        Returns
        -------
        None
        """

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
    Returns an m x 2 array of the airfoil coordinates based on a Selig formatted .dat file.

    Parameters
    ----------
    path_coord : str
        Path to the Selig formatted .dat file containing the airfoil coordinates.

    Returns
    -------
    np.ndarray
        An m x 2 array with the airfoil coordinates where x goes from the top trailing edge to the top leading edge and then back to the lower trailing edge, maintaining the Selig format.
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
    """
    Returns two functions which interpolate the coordinates of the given airfoil. The first function interpolates the top skin,
    and the second function interpolates the bottom skin. The resulting interpolation functions take into account an airfoil
    scaled by the given chord.

    Parameters
    ----------
    path_coord : str
        Path to the airfoil coordinates using the Selig format.

    Returns
    -------
    Tuple[CubicSpline, CubicSpline]
        A cubic spline of the top skin and lower skin, respectively.
    """


    coord_scaled = read_coord(path_coord)*chord # coordinates of airfoil scaled by the chord
    top_coord =  coord_scaled[:np.argmin(coord_scaled[:,0]), :] #coordinates of top skin
    top_coord = np.flip(top_coord, axis=0)
    bot_coord =  coord_scaled[np.argmin(coord_scaled[:,0]):, :] # coordinates of bottom skin

    top_interp = CubicSpline(top_coord[:,0], top_coord[:,1])
    bot_interp = CubicSpline(bot_coord[:,0], bot_coord[:,1])
    return top_interp, bot_interp

def get_centroids(path_coord:str) -> Tuple[float, float]:
    r"""
    Compute the nondimensional x and y centroid based on the coordinate file of an airfoil.
    The centroids are computed assuming the following:

    .. admonition:: Assumptions

        1. It is only based on the skin, i.e., the spar webs and stringers are ignored. Additionally, the different thickness of the skin are not taken into account.
        The implication being that the x centroid should be at x/c = 0.5, unless there was a bias in the sampling points.

    .. admonition:: Future improvment

        1. Take into account the spar webs for a better x centroid. However, this is irrelevant for now as we only
        take into account forces in the vertical direction.

    Parameters
    ----------
    path_coord : str
        Path to the geometry file using the Selig format.

    Returns
    -------
    Tuple[float, float]
        The nondimensional x and y centroid of the airfoil.
    """

    coord = read_coord(path_coord)
    y_centroid_dimless = np.sum(coord[:,1])/coord.shape[0]
    x_centroid_dimless = np.sum(coord[:,0])/coord.shape[0]
    return x_centroid_dimless, y_centroid_dimless


def discretize_airfoil(path_coord:str, chord:float, wingbox_struct:Wingbox) -> IdealWingbox:
    r"""
    Create a discretized airfoil according to the principles of Megson based on a path to a txt file containing
    the non-dimensional coordinates of the airfoil, the corresponding chord, and the wingbox data structure fully filled in.

    .. admonition:: Assumptions

        1. Airfoil is idealized according to Megson ch. 20.
        2. Each stringer will form one boom in the discretized airfoil.
        3. Only an equal amount of stringers can be specified per cell; if that is not the case, a warning is issued (due to the method of discretization).
        4. The ratio of :math:`\frac{\sigma_1}{\sigma_2}` required for the skin contribution to the boom size based on the skin is determined by the ratio of their y position, thus :math:`\frac{y_1}{y_2}`.

    **General Procedure**

    1. Create a spline of the top and bottom airfoil.
    2. Create an array along which to sample this spline to create the booms, creating specific samples for the spar positions.
    3. Move over the top surface, creating booms and panels as we go.
    4. Do the same for the bottom surface, moving in a circular motion.
    5. Move over all the spars and create booms and panels as we go.
    6. Iterate over all booms and add skin contribution and stringer contribution to all their areas.

    .. admonition:: Future improvement

        1. Add contribution of spar caps (for now, this has been left out as I did not see the value of it at the time).

    Parameters
    ----------
    path_coord : str
        Path to the file containing the non-dimensional coordinates of the airfoil.
    chord : float
        Chord length of the airfoil.
    wingbox_struct : Wingbox
        The wingbox data structure fully filled in.

    Returns
    -------
    IdealWingbox
        The discretized idealized wingbox.
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


            # See assumptions, the amount of stringers is supposed to be an even numbers hence
            # the code below
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
    wingbox.area_lst = wingbox.get_cell_areas()
    wingbox.centroid_lst = [np.array(poly.centroid.xy).flatten() for poly in wingbox.get_polygon_cells()]
    wingbox.Ixx = wingbox._set_Ixx()
    wingbox.Ixy = wingbox._set_Ixy()
    wingbox.Iyy = wingbox._set_Iyy()

    return wingbox



