from scipy.constants import g
import numpy as np

class ISA:
    """
    Calculates the atmospheric parameters at a specified altitude h (in meters).
    An offset in sea level temperature can be specified to allow for variations.

    Note: Since our aircraft probably doesn't fly too high, this is only valid in the troposphere

    Verified by comparison to: https://www.digitaldutch.com/atmoscalc/
    """

    def __init__(self, h, T_offset=0):

        # Constants
        self.a = -0.0065    # [K/m]     Temperature lapse rate
        self.g0 = g   # [m/s^2]   Gravitational acceleration
        self.R = 287        # [J/kg K]  Specific gas constant
        self.gamma = 1.4    # [-]       Heat capacity ratio

        # Sea level values
        # [kg/m^3]  Sea level density
        self.rho_SL = 1.225
        # [Pa]      Sea level pressure
        self.p_SL = 101325
        # [K]       Sea level temperature
        self.T_SL = 288.15 + T_offset
        # [kg/m/s] Sea Level Dynamic Viscosity 1.81206E-5
        self.mu_SL = 1.7894E-5
        # [m/s] Sea level speed of sound
        self.a_SL = np.sqrt(self.gamma*self.R*self.T_SL)

        self.h = h  # [m]       Altitude

        # Throw an error if the specified altitude is outside of the troposphere
        if np.any(h) > 11000:
            raise ValueError(
                'The specified altitude is outside the range of this class')

        # [K] Temperature at altitude h, done here because it is used everywhere
        self.T = self.T_SL + self.a * self.h

    def temperature(self):
        return self.T

    def pressure(self):
        p = self.p_SL * (self.T / self.T_SL) ** (-self.g0 / (self.a * self.R))
        return p

    def density(self):
        rho = self.rho_SL * \
            (self.T / self.T_SL) ** (-self.g0 / (self.a * self.R) - 1)
        return rho

    def soundspeed(self):
        a = self.a_SL * np.sqrt(self.T/self.T_SL)
        return a

    def viscosity_dyn(self):
        mu = self.mu_SL * (self.T / self.T_SL) ** (1.5) * \
            (self.T_SL + 110.4) / (self.T + 110.4)
        # 1.458E-6 * self.T ** 1.5 / (self.T + 110.4) # Sutherland Law, using Sutherland's constant S_mu = 110.4 for air
        return mu
