import sys
import pathlib as pl
import numpy as np

sys.path.append(str(list(pl.Path(__file__).parents)[1]))

import tuduam.aerodynamics as aero
import matplotlib.pyplot as plt
import pdb


def test_lift_distribution(FixtSingleWing):
    if True:
        lift_func = aero.lift_distribution(FixtSingleWing, 3*3.14/180,20, 1.225, 80)
        x = np.linspace(0, FixtSingleWing.span/2, 100)
        plt.plot(x, lift_func(x))
        plt.show()








