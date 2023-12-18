import numpy 
import matplotlib.pyplot as plt


class Boom:
    """ A class to represent an idealized boom 
    """    
    def __init__(self):
        self.bid = 0 #  Boom ID
        self.A = 0  # Area boom
        self.x # X location 
        self.y # Y location

class IdealPanel:
    """_summary_
    """    
    def __init__(self):
        self.pid = 0 # Panel ID
        self.bid1 = 0  # The boom id of its corresponding first boom id
        self.bid2 = 0 # The boom id of its corresponding second boom id

class IdealWingbox:
    
    def __init__(self) -> None:

        self.cells = 0 # The amount of cells in the airfoil
        self.panels = {} # Dictionary with all panels
        self.booms = {} # Dictionary with all booms


def idealize_wingbox(path_airfoil, n_panel_top, n_panel_bot ):

    # Lists to store x and y coordinates
    x_coordinates = []
    y_coordinates = []

    # Read the file and extract coordinates
    with open(path_airfoil, 'r') as file:
        for line in file:

            if any(c.isalpha() for c in line):
                continue
            values = line.split()
            x, y = map(float, values[:2])  # Assuming floats; use int() if dealing with integers
            
            x_coordinates.append(x)
            y_coordinates.append(y)
