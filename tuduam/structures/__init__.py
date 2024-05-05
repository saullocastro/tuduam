
from tuduam.structures.wingbox import discretize_airfoil, IdealWingbox
from .wingbox import discretize_airfoil, class2_wing_mass, class2_fuselage_mass 
from .constraints import IsotropicWingboxConstraints
from .optimization import  SectionOptimization

# Following import were to keep backwards compatiblity, may be removed but then the tests have to be fixed
from .wingbox import read_coord, get_centroids, Boom, IdealPanel


