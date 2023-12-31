import numpy  as np
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from .data_structures import Wingbox
from math import ceil
import pdb

class Boom:
    """ A class to represent an idealized boom
    """
    def __init__(self):
        self.bid = None #  Boom ID
        self.A = None  # Area boom
        self.x = None # X location
        self.y = None # Y location

class IdealPanel:
    """_summary_
    """
    def __init__(self):
        self.pid = None # Panel ID
        self.bid1 = None  # The boom id of its corresponding first boom id
        self.bid2 = None # The boom id of its corresponding second boom id
        self.b1 = None
        self.b2  = None
        self.q_basic = None
        self.q_tot = None
        self.tau = None

class IdealWingbox():
    def __init__(self, wingbox: Wingbox) -> None:
        self.wingbox_struct = wingbox
        self.x_centroid = None # datum attached to leading edge
        self.y_centroid = None
        self.panel_dict = {}
        self.boom_dict = {}
        pass

    def plot(self) -> None:
        plt.figure(figsize=(10,1))
        for key, panel in self.panel_dict.items():
            x = [panel.b1.x, panel.b2.x]
            y = [panel.b1.y, panel.b2.y]
            plt.plot(x,y, ">-")
        y_arr  = [i.y for i in self.boom_dict.values()]
        y_max = np.max(y_arr)
        y_min = np.min(y_arr)
        # aspect_ratio = np.max([i.y for i in self.boom_dict.values()])*2/np.max([i.x for i in self.boom_dict.values()])
        aspect_ratio = 0.3
        # plt.gca().set_aspect(aspect_ratio, adjustable='box')
        plt.ylim([ y_min - 0.1,y_max + 0.1])
        plt.show()
        pass


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

def interp_airfoil(path_coord:str, chord:float) -> Tuple[CubicSpline, CubicSpline]:
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
      1. Discretize the skin first and create the panels there first.



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
    assert len(wingbox_struct.spar_loc_dimless) == wingbox_struct.n_cell - 1, "Length of spar_loc should be equal to the amount of cells - 1"

    top_interp, bot_interp = interp_airfoil(path_coord, chord)
    x_centr, y_centr = get_centroids(path_coord)

    wingbox = IdealWingbox(wingbox_struct)
    wingbox.x_centroid =  x_centr*chord
    wingbox.y_centroid =  y_centr*chord


    x_boom_loc = np.linspace(0, chord, ceil(wingbox_struct.booms_sk/2 + 2))
    #TODO correct for spar locations 

    boom_dict = {}
    panel_dict = {}
    bid = 0 # set bid to zero
    pid = 0  # set panel counter to zero
    # create the upper booms and panels from leading edge to trailing edge
    for idx, x_boom in enumerate(x_boom_loc):
        boom = Boom()
        boom.bid = bid
        boom.x =  x_boom
        boom.y = top_interp(x_boom)

        boom_dict[boom.bid] = boom

        if not idx == 0: # Create panels, can not create a panel on the first pass through the for loop
            pnl = IdealPanel()
            pnl.pid = pid
            pnl.bid1 = bid - 1
            pnl.bid2 = bid 
            pnl.b1 =  boom_dict[pnl.bid1]
            pnl.b2 =  boom_dict[pnl.bid2]
            #TODO create the areas  based on sigma ratio

            panel_dict[pid] = pnl
            # boom.A = 0 # Boom on  x axis should not have any area as it 

        bid += 1
        pid += 1

    # Create the lower booms and panels from trailing edge to leading edge
    for x_boom in np.flip(x_boom_loc[1:-1]):
        boom = Boom()
        boom.bid = bid
        boom.x =  x_boom
        boom.y = bot_interp(x_boom)
        boom_dict[boom.bid] = boom

        pnl = IdealPanel()
        pnl.pid = pid
        pnl.bid1 = bid - 1
        pnl.bid2 = bid 
        pnl.b1 =  boom_dict[pnl.bid1]
        pnl.b2 =  boom_dict[pnl.bid2]
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
    #TODO create the areas  based on sigma ratio
    panel_dict[pid] = pnl


    wingbox.boom_dict.update(boom_dict)
    wingbox.panel_dict.update(panel_dict)

    return wingbox


def stress_analysis(airfoil: IdealWingbox, intern_shear:float, internal_mz:float) ->  Tuple[float, dict]:
    """  TODO: Consider implementing as a method of IdealWingbox class
    
    Perform stress analysis on  a wingbox section 

    List of assumptions (Most made in Megson, some for simplificatoin of code)
    ---------------------------------------
    - The effect of taper are not included see 21.2 (See megson) TODO: future implementation
    - Lift acts through the shear centre (no torque is created) TODO: future implementation
    - Forces in the x-y plane are not considered 



    :param airfoil: _description_
    :type airfoil: IdealWingbox
    :param intern_shear: _description_
    :type intern_shear: float
    :return: _description_
    :rtype: Tuple[float, dict]
    """    
    pass

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
        S_ft = wing.surface*10.7639104
        mtow_lbs = 2.20462 * vtol.mtom
        wing.mass= 0.04674*(mtow_lbs**0.397)*(S_ft**0.36)*(flight_perf.n_ult**0.397)*(wing.aspect_ratio**1.712)*0.453592
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
        mtow_lbs = 2.20462 * vtol.mtom
        lf_ft, lf = fuselage.length_fuselage*3.28084, fuselage.length_fuselage

        nult = flight_perf.n_ult # ultimate load factor
        wf_ft = fuselage.width_fuselage*3.28084 # width fuselage [ft]
        hf_ft = fuselage.height_fuselage*3.28084 # height fuselage [ft]
        Vc_kts = flight_perf.v_cruise*1.94384449 # design cruise speed [kts]

        fweigh_USAF = 200*((mtow_lbs*nult/10**5)**0.286*(lf_ft/10)**0.857*((wf_ft + hf_ft)/10)*(Vc_kts/100)**0.338)**1.1
        fuselage.mass= fweigh_USAF*0.453592
        return fuselage.mass
