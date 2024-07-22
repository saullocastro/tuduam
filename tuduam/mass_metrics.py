"""The following module contains methods to size for the weight of an eVTOL. Most of these methods are class 2 weigt estimations
"""

import scipy.constants as const
from .data_structures import *

def class2_wing_mass(vtol: VTOL, flight_perf: FlightPerformance, wing: SingleWing):
    """
        Returns the structural weight of both wings.

        Parameters
        ----------
        vtol : VTOL
            VTOL data structure.
        flight_perf : FlightPerformance
            FlightPerformance data structure.
        wing : SingleWing
            SingleWing data structure.

        Returns
        -------
        float
            Mass of both wings in kg.
        """
    S_ft = wing.surface * 1 / const.foot ** 2
    mtow_lbs = 1 / const.pound * vtol.mtom
    wing.mass = (
        0.04674
        * (mtow_lbs ** 0.397)
        * (S_ft ** 0.36)
        * (flight_perf.n_ult ** 0.397)
        * (wing.aspect_ratio ** 1.712)
        * const.pound
    )
    return wing.mass


def class2_fuselage_mass(vtol: VTOL, flight_perf: FlightPerformance, fuselage: Fuselage):
    """
    Returns the mass of the fuselage.

    Parameters
    ----------
    vtol : VTOL
        VTOL data structure, requires: mtom.
    flight_perf : FlightPerformance
        FlightPerformance data structure.
    fuselage : Fuselage
        Fuselage data structure.

    Returns
    -------
    float
        Fuselage mass.
    """
    mtow_lbs = 1 / const.pound * vtol.mtom
    lf_ft, lf = fuselage.length_fuselage * 1 / const.foot, fuselage.length_fuselage

    nult = flight_perf.n_ult  # ultimate load factor
    wf_ft = fuselage.width_fuselage * 1 / const.foot  # width fuselage [ft]
    hf_ft = fuselage.height_fuselage * 1 / const.foot  # height fuselage [ft]
    Vc_kts = flight_perf.v_cruise * 1 / const.foot  # design cruise speed [kts]

    fweigh_USAF = (
        200
        * (
            (mtow_lbs * nult / 10 ** 5) ** 0.286
            * (lf_ft / 10) ** 0.857
            * ((wf_ft + hf_ft) / 10)
            * (Vc_kts / 100) ** 0.338
        )
        ** 1.1
    )
    fuselage.mass = fweigh_USAF * const.pound
    return fuselage.mass
