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
        self.bid = 0 #  Boom ID
        self.A = 0  # Area boom
        self.x = None # X location
        self.y = None # Y location

class IdealPanel:
    """_summary_
    """
    def __init__(self):
        self.pid = 0 # Panel ID
        self.bid1 = 0  # The boom id of its corresponding first boom id
        self.bid2 = 0 # The boom id of its corresponding second boom id

class Cell:
    """_summary_
    """
    def __init__(self):
        self.cid = 0 # Panel ID
        self.panel_dict = {} 
        self.boom_dict = {} 

class IdealWingbox():
    def __init__(self, wingbox: Wingbox) -> None:
        self.wingbox_struct = wingbox
        self.cell_dict = {}
        self.panel_dict = {}
        self.boom_dict = {}
        pass

    def plot(self) -> None:
        pass


def read_coord(path_coord:str) -> np.ndarray:
    """ Returns an array Using a path to coordinate file of the airfoil

    :return: Returns an array with the airfoil coordinates
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

def create_panels(boom_dict:dict, pid:int) -> Tuple[dict,int]:
    """ Function assumes the dictonary keys are integers where each consecutive key-value pair should be connected.
    Thus it is important that the key 0 and key 1 should be adjacent and connected. If this is wrongfully done the panel
    will be falsely placed. The value should be of the class Boom. e.g
    
    boom_dict ={
        "0": Boom class 1
        "1": Boom class 1
        .................
        "n: Boom class n
    }

    

    :param boom_dict: _description_
    :type boom_dict: dict
    :param pid: The counter used to asign panel id's. The pid increments in the function and is returned
    at the end in order to reassign to the global variable used
    :type pid: int
    :return: 
    :rtype: Tuple[dict,int]
    """    
    pnl_dict = {}

    for key, boom in boom_dict:
        pnl = IdealPanel()
        pnl.pid = pid
        pid += 1
        pass

    return pnl_dict, pid

def discretize_airfoil(path_coord:str, chord:float, wingbox_struct:Wingbox) -> IdealWingbox:
    """ Create a discretized airfoil according to the principles of Megson based on a path to a txt file containing
      the non-dimensional coordinates of the airfoil, the corresonding chord and the wingbox data structure fully filled in.

    :param path_coord: _description_
    :type path_coord: str
    :param chord: _description_
    :type chord: float
    :param wingbox_struct: _description_
    :type wingbox_struct: Wingbox
    :return: _description_
    :rtype: IdealWingbox
    """    
    
    # Check data structure
    assert len(wingbox_struct.t_sk_cell) == wingbox_struct.n_cell, "Length of t_sk_cell should be equal to the amount of cells"
    assert len(wingbox_struct.str_cell) == wingbox_struct.n_cell, "Length of str_cell should be equal to the amount of cells"
    assert len(wingbox_struct.spar_loc_dimless) == wingbox_struct.n_cell - 1, "Length of spar_loc should be equal to the amount of cells - 1"

    coord_scaled = read_coord(path_coord)*chord # coordinates of airfoil scaled by the chord
    top_coord =  coord_scaled[:np.argmin(coord_scaled[:,0]), :] #coordinates of top skin
    top_coord = np.flip(top_coord, axis=0)
    bot_coord =  coord_scaled[np.argmin(coord_scaled[:,0]):, :] # coordinates of bottom skin

    top_interp = CubicSpline(top_coord[:,0], top_coord[:,1])
    bot_interp = CubicSpline(bot_coord[:,0], bot_coord[:,1])


    id_wingbox = IdealWingbox(wingbox_struct)
    bid = 0 # set bid to zero
    pid = 0  # set panel counter to zero
    for cell in range(wingbox_struct.n_cell):
        cell_cls = Cell()
        cell_cls.cid = cell

        if cell == 0: #left most cell
            spar_loc = wingbox_struct.spar_loc_dimless[cell]*chord # location of the right spar for the cell
            # Divide first cell into three section, upper skin, lower skin and spar. Distribute booms evenly
            x_boom_loc = np.linspace(0, spar_loc, ceil(wingbox_struct.booms_cell/3))

            # Create booms on the skin
            boom_dict = {}
            for idx, x_boom in enumerate(x_boom_loc):
                if  x_boom == 0: # If at x/c = 0  we only create one boom as they coincide
                    boom = Boom()
                    boom.bid = bid
                    # boom.A = wingbox_struct.area_str
                    boom.x =  x_boom
                    boom.y = bot_interp(x_boom) # It is better to select the boottom one due to the indexing of the coordinates
                    bid += 1
                    boom_dict[str(boom.bid)] = boom
                else:
                    boom_up = Boom()
                    boom_up.bid = bid
                    # boom_up.A = wingbox_struct.area_str
                    boom_up.x =  x_boom
                    boom_up.y = top_interp(x_boom)
                    bid += 1
                    boom_dict[str(boom_up.bid)] = boom

                    boom_low = Boom()
                    boom_low.bid = bid
                    # boom_low.A = wingbox_struct.area_str
                    boom_low.x =  x_boom
                    boom_low.y = bot_interp(x_boom)
                    bid += 1
                    boom_dict[str(boom_low.bid)] = boom
            # Now we create the booms on the first sparweb
            for y_boom in np.linspace(bot_interp(spar_loc), top_interp(spar_loc), ceil(wingbox_struct.booms_cell/3)):
                boom_low = Boom()
                boom_low.bid = bid
                # boom_low.A = wingbox_struct.area_str
                boom_low.x = spar_loc 
                boom_low.y =  y_boom
                bid += 1
                boom_dict[str(boom_low.bid)] = boom
            
            
            # Create panels from the boom locations defined

            
        elif cell == wingbox_struct.n_cell - 1: # right most cell
            pass
        else: # cells in between
            pass

        cell_cls.boom_dict.update(boom_dict) #Write boom dict to cell class 
        id_wingbox.cell_dict[str(cell)] = cell_cls

    return id_wingbox

