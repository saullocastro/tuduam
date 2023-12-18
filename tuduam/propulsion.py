import numpy as np
import re
from tuduam.data_structures import Propeller
import os
import matplotlib.pyplot as plt
import scipy.integrate as spint
from scipy.interpolate import NearestNDInterpolator
import scipy.optimize as sp_opt
import pdb



def alpha_xfoil_interp(dir_path):
    """ 
    This function interpolates the angle of attack based on the reynolds numbers and lift coefficient
    It uses the xfoil polar from various reynolds number created by the user. Its functionality
    depends on the format of these files thus it is important xfoil is used. The spacing of the reynolds number
    can be arbitrary but it is up to the user to verify this accuracy. This function utilizest a radial basis
    interpolator from the scipy library

    :param dir_path: Directory of the files containting the xfoil polar
    :type dir_path: str
    :return: A interpolator with the input cl and the Reynolds number. In that order.
    :rtype: _type_
    """    

    data_points = list()
    res_lst = list()

    for file in os.listdir(dir_path):
        reyn = float(re.findall(r'\d+', file)[0])
        with open(os.path.join(dir_path,file), "r") as f:
            write = False
            for line in f.readlines():
                if  line.count("-") > 3:
                    write = True
                    continue
                if write:
                    value_lst = [float(value) for value in line.split()]
                    data_points.append([value_lst[1], reyn])
                    res_lst.append(value_lst[0])

    data_points =  np.array(data_points)
    res_arr =  np.array(res_lst)

    return NearestNDInterpolator(data_points, res_arr)

def cl_xfoil_interp(dir_path):
    """ 
    This function interpolates the angle of attack based on the reynolds numbers and lift coefficient
    It uses the xfoil polar from various reynolds number created by the user. Its functionality
    depends on the format of these files thus it is important xfoil is used. The spacing of the reynolds number
    can be arbitrary but it is up to the user to verify this accuracy. This function utilizest a radial basis
    interpolator from the scipy library

    :param dir_path: Directory of the files containting the xfoil polar
    :type dir_path: str
    :return: A interpolator with the input angle of attack and the Reynolds number. In that order.
    :rtype: _type_
    """    

    data_points = list()
    res_lst = list()

    for file in os.listdir(dir_path):
        reyn = float(re.findall(r'\d+', file)[0])
        with open(os.path.join(dir_path,file), "r") as f:
            write = False
            for line in f.readlines():
                if  line.count("-") > 3:
                    write = True
                    continue
                if write:
                    value_lst = [float(value) for value in line.split()]
                    data_points.append([value_lst[0], reyn])
                    res_lst.append(value_lst[1])

    data_points =  np.array(data_points)
    res_arr =  np.array(res_lst)

    return NearestNDInterpolator(data_points, res_arr)


def cd_xfoil_interp(dir_path):
    """ 
    This function interpolates the angle of attack based on the lift coefficient and lift coefficient
    It uses the xfoil polar from various reynolds number created by the user. Its functionality
    depends on the format of these files thus it is important xfoil is used. The spacing of the reynolds number
    can be arbitrary but it is up to the user to verify this accuracy. This function utilizest a radial basis
    interpolator from the scipy library

    :param dir_path: Directory of the files containting the xfoil polar
    :type dir_path: str
    :return: A interpolator with the input cl and the Reynolds number. In that order.
    :rtype: _type_
    """    

    data_points = list()
    res_lst = list()

    for file in os.listdir(dir_path):
        reyn = float(re.findall(r'\d+', file)[0])
        with open(os.path.join(dir_path,file), "r") as f:
            write = False
            for line in f.readlines():
                if  line.count("-") > 3:
                    write = True
                    continue
                if write:
                    value_lst = [float(value) for value in line.split()]
                    data_points.append([value_lst[1], reyn])
                    res_lst.append(value_lst[2])

    data_points =  np.array(data_points)
    res_arr =  np.array(res_lst)

    return NearestNDInterpolator(data_points, res_arr)

class PlotBlade:
    def __init__(self, chords, pitchs, radial_coords, R, xi_0, airfoil_name='naca4412', tc_ratio=0.12):
        """
        :param chords: Array with chords, from root to tip [m]
        :param pitchs: Array with pitch angles, from root to tip [rad]
        :param radial_coords: Radial coordinates per station [m]
        :param R: Radius of propeller [m]
        :param xi_0: Hub ratio [-]
        :param airfoil_name: String with ile name of the airfoil to use, by default NACA4412 [-]
        :param tc_ratio: Thickness to chord ratio of the airfoil, by default 12% for NACA4412 [-]
        """
        self.chords = chords
        self.pitchs = pitchs
        self.radial_coords = radial_coords
        self.R = R
        self.xi_0 = xi_0
        self.airfoil_name = airfoil_name
        self.tc_ratio = tc_ratio

    def load_airfoil(self):
        file = open('../PropandPower/'+self.airfoil_name)

        airfoil = file.readlines()

        # Close file
        file.close()

        # List to save formatted coordinates
        airfoil_coord = []

        for line in airfoil:
            # Separate variables inside file
            a = line.split()

            new_line = []
            for value in a:
                new_line.append(float(value))

            # Set c/4 to be the origin
            new_line[0] -= 0.25
            airfoil_coord.append(new_line)

        airfoil_coord = np.array(airfoil_coord)
        airfoil_coord = airfoil_coord.T

        return airfoil_coord

    def plot_blade(self):
        # Create figures
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('equal')

        # Plot side view of the airfoil cross-sections
        for i in range(len(self.chords)):
            # Scale the chord length and thickness
            x_coords = self.load_airfoil()[0] * self.chords[i]
            y_coords = self.load_airfoil()[1] * self.chords[i]

            # New coordinates after pitch
            x_coords_n = []
            y_coords_n = []

            # Apply pitch
            for j in range(len(x_coords)):
                # Transform coordinates with angle
                x_coord_n = np.cos(self.pitchs[i]) * x_coords[j] + np.sin(self.pitchs[i]) * y_coords[j]
                y_coord_n = -np.sin(self.pitchs[i]) * x_coords[j] + np.cos(self.pitchs[i]) * y_coords[j]

                # Save new coordinates
                x_coords_n.append(x_coord_n)
                y_coords_n.append(y_coord_n)

            # Plot the cross section

            axs[0].plot(x_coords_n, y_coords_n)
        axs[0].hlines(0, -0.2, 0.3, label='Disk plane', colors='k', linewidths=0.75)

        y_mins = []
        y_maxs = []
        for i in range(len(self.chords)):
            chord_len = self.chords[i]
            # Plot chord at its location, align half chords
            y_maxs.append(chord_len/4)
            y_mins.append(-3*chord_len/4)


        # Polinomial regression for smooth distribution
        coef_y_max_fun = np.polynomial.polynomial.polyfit(self.radial_coords, y_maxs, 5)
        coef_y_min_fun = np.polynomial.polynomial.polyfit(self.radial_coords, y_mins, 5)

        y_max_fun = np.polynomial.polynomial.Polynomial(coef_y_max_fun)
        y_min_fun = np.polynomial.polynomial.Polynomial(coef_y_min_fun)

        # Plot
        axs[1].axis('equal')

        # Plot actual points
        axs[1].scatter(self.radial_coords, y_maxs)
        axs[1].scatter(self.radial_coords, y_mins)

        # Plot smooth distribution  TODO: revise
        radius = np.linspace(self.xi_0*self.R, self.R, 200)
        axs[1].plot(radius, y_min_fun(radius))
        axs[1].plot(radius, y_max_fun(radius))

        axs[0].legend()
        plt.show()

    def plot_3D_blade(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.set_aspect('equal')

        # Plot airfoil blade in 3D
        for i in range(len(self.chords)):
            # Scale the chord length and thickness
            x_coords = self.load_airfoil()[0] * self.chords[i]
            y_coords = self.load_airfoil()[1] * self.chords[i]

            # New coordinates after pitch
            x_coords_n = []
            y_coords_n = []

            blade_plot = np.empty(3)

            # Apply pitch
            for j in range(len(x_coords)):
                # Transform coordinates with angle
                x_coord_n = np.cos(self.pitchs[i]) * x_coords[j] + np.sin(self.pitchs[i]) * y_coords[j]
                y_coord_n = -np.sin(self.pitchs[i]) * x_coords[j] + np.cos(self.pitchs[i]) * y_coords[j]

                # Save new coordinates
                x_coords_n.append(x_coord_n)
                y_coords_n.append(y_coord_n)

                # Save coordinates of each point
                point = [x_coord_n, y_coord_n, self.radial_coords[i]]
                blade_plot = np.vstack((blade_plot, point))

            ax.plot3D(x_coords_n, y_coords_n, self.radial_coords[i], color='k')

        # ax.plot3D(blade_plot[:][0], blade_plot[:][1], blade_plot[:][2], color='k')


        # Trick to set 3D axes to equal scale, obtained from:
        # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

        # Just to get max X, Y, and Z
        X = np.array([self.chords[0], self.chords[-1]])
        Y = np.array([self.chords[0]*self.tc_ratio, self.chords[-1]*self.tc_ratio])
        Z = np.array([0, self.radial_coords[-1]])

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

class BEM:
    def __init__(self, data_path, propclass, rho, dyn_vis, V_fr, N_stations, a, T=None, P=None):
        """
        :param propclass: Propeller class from [kg/m^3]
        :param rho: Density [kg/m^3]
        :param dyn_vis: Dynamic viscosity [N s/m^2)
        :param V_fr: Freestream velocity
        :param T: Thrust delivered BY the propeller [N]
        :param P: Power delivered TO the propeller [W]
        :param N_stations: Number of stations to calculate [-] (preferably > 20)
        :param a: Speed of sound [m/s]
        :param RN_spacing: Spacing of the Reynold's numbers of the airfoil data files [-] (added for flexibility,
                           but probably should be 100,000)

        :out: [0] -> speed ratio, used for iterations, internal variable
              [1] -> Array with relevant parameters for blade design and propeller performance:
                     [chord per station, beta per station, alpha per station, E per station, eff, Tc, Pc]
        """
        self.B = propclass.n_blades
        self.R = propclass.r_prop
        self.D = 2*self.R
        self.Omega = propclass.rpm*2 * np.pi / 60  # rpm to rad/s
        self.xi_0 = propclass.xi_0
        self.rho = rho
        self.dyn_vis = dyn_vis
        self.V = V_fr
        # self.phi_T = 1
        self.lamb = V_fr/(self.Omega*self.R)  # Speed ratio
        self.N_s = N_stations
        self.a = a
        self.dir_path = data_path
        self.alpha_interp = alpha_xfoil_interp(data_path)
        self.cd_interp = cd_xfoil_interp(data_path)

        # Define thrust or power coefficients, depending on input
        if T is not None:
            self.Tc = 2 * T / (rho * V_fr**2 * np.pi * self.R**2)
            self.Pc = None

        elif P is not None:
            self.Pc = 2 * P / (rho * V_fr**3 * np.pi * self.R**2)
            self.Tc = None
        else:
            raise Exception("Please specify either T or P (not both)")

    # Prandtl relation for tip loss
    def F(self, r, zeta):
        return (2/np.pi) * np.arccos(np.exp(-self.f(r, zeta)))

    # Exponent used for function above
    def f(self, r, zeta):
        return (self.B/2)*(1-self.Xi(r))/(np.sin(self.phi_t(zeta)))

    # Pitch of blade tip
    def phi_t(self, zeta):
        return np.arctan(self.lamb * (1 + zeta/2))

    # Non-dimensional radius, r/R
    def Xi(self, r):
        return r/self.R

    # Angle of local velocity of the blade wrt to disk plane
    def phi(self, r, zeta):
        return np.arctan(np.tan(self.phi_t(zeta)) * self.R / r)

    # Mach as a function of radius
    def M(self, r):
        speed = np.sqrt(self.V**2 + (self.Omega*r)**2)
        return speed/self.a

    # Reynolds number
    def RN(self, Wc):
        # Reynolds number. Wc is speed times chord
        return Wc * self.rho / self.dyn_vis

    # Product of local speed at the blade and chord
    def Wc(self, F, phi, zeta, Cl):
        return 4*np.pi*self.lamb * F * np.sin(phi) * np.cos(phi) * self.V * self.R * zeta / (Cl * self.B)
        # return 4 * np.pi * r * zeta * self.V * F * np.sin(phi) * np.cos(phi) / (Cl * self.B)

    # Non-dimensional speed
    def x(self, r):
        return self.Omega*r/self.V

    # Advance ratio
    def J(self):
        return self.V / ((self.Omega/(2*np.pi)) * self.D)

    def phi_int(self, xi, zeta):
        return np.arctan((1 + zeta/2)*self.lamb/xi)

    # F function used for integration part only
    def F_int(self, xi, zeta):
        return 2*np.arccos(np.exp(-self.f_int(xi, zeta)))/np.pi

    # f function used for integration part only
    def f_int(self, xi, zeta):
        return (self.B/2)*(1 - xi)/np.sin(self.phi_t(zeta))

    # G function used for integration part only
    def G_int(self, xi, zeta):
        return self.F_int(xi, zeta) * np.cos(self.phi_int(xi, zeta)) * np.sin(self.phi_int(xi, zeta))


    # Integrals used to calculate internal variables, refer to paper for more explanation if needed
    # Assuming average eps
    def I_prim_1(self, xi, zeta, eps):
        return 4 * xi * self.G_int(xi, zeta) * (1 - eps * np.tan(self.phi_int(xi, zeta)))

    def I_prim_2(self, xi, zeta, eps):
        return self.lamb * (self.I_prim_1(xi, zeta, eps) / (2 * xi)) * (1 + eps / np.tan(self.phi_int(xi, zeta))) * \
               np.sin(self.phi_int(xi, zeta)) * np.cos(self.phi_int(xi, zeta))

    def J_prim_1(self, xi, zeta, eps):
        return 4 * xi * self.G_int(xi, zeta) * (1 + eps / np.tan(self.phi_int(xi, zeta)))

    def J_prim_2(self, xi, zeta, eps):
        return (self.J_prim_1(xi, zeta, eps) / 2) * (1 - eps * np.tan(self.phi_int(xi, zeta))) * \
               (np.cos(self.phi_int(xi, zeta))) ** 2



    # Propeller efficiency Tc/Pc
    def efficiency(self, Tc, Pc):
        return Tc/Pc

    # Prandtl-Glauert correction factor: sqrt(1 - M^2)
    def PG(self, M):
        return np.sqrt(1 - M**2)

    # This function runs the design procedure from an arbitrary start zeta (which can be 0)
    def run_BEM(self, zeta):
        # Array with station numbers
        stations = np.arange(1, self.N_s + 1)

        # Length of each station
        st_len = (self.R - self.R*self.xi_0)/len(stations)

        # Radius of the middle point of each station. Station 1 has st length/2, each station has that plus N*st length, Station 1 starts after hub
        stations_r = self.xi_0*self.R + st_len/2 + (stations-1)*st_len
        # stations_r = self.xi_0*self.R + (stations)*st_len
        # F and phi for each station
        F = self.F(stations_r, zeta)
        phis = self.phi(stations_r, zeta)

        # trial with a different range of Cls
        Cls_trial = np.arange(0.1, 1.2, 0.05)

        # Create arrays for lift and drag coefficients, angle of attack and D/L ratio for each station
        Cl = np.ones(self.N_s)
        Cd = np.ones(self.N_s)
        alpha = np.ones(self.N_s)
        E = np.ones(self.N_s)
        cs = np.ones(self.N_s)
        betas = np.ones(self.N_s)
        # Ves = np.ones(self.N_s)
        Ves = zeta*self.V + self.V

        # Optimise each station for max L/D
        for station in stations:
            station -= 1
            eps_min = 1
            optim_vals = [1, 1, 1, 1]

            # Optimise each station
            for lift_coef in Cls_trial:
                # lift_coef = lift_coef * self.PG(self.M(stations_r[station]))

                # Calculate product of local speed with chord
                Wc = self.Wc(F[station], phis[station], zeta, lift_coef)
                # Wc = self.Wc(F[station], phis[station], zeta, lift_coef, stations_r[station])

                # Calculate Reynolds number at the station to look for the correct airfoil datafile
                Reyn = self.RN(Wc)

                reyn_lst = []
                for file in os.listdir(self.dir_path):
                    reyn_lst.append(int(re.findall(r'\d+', file)[0]))

                reyn_min = np.min(reyn_lst)
                reyn_max = np.min(reyn_lst)

                # Maximum and minimum RN in database
                if Reyn<reyn_min:
                    Reyn = reyn_min 
                if Reyn>reyn_max:
                    Reyn = reyn_max

                cl_corr = (lift_coef * self.PG(self.M(stations_r[station]))) # Corrected cl for compressibility
                Cd_ret = self.cd_interp([[cl_corr, Reyn]])
                alpha_ret = np.deg2rad(self.alpha_interp([[cl_corr, Reyn]]))       # Retrieved AoA (from deg to rad)

                # Compute D/L ration
                eps = Cd_ret / lift_coef

                # See if D/L is minimum. If so, save the values
                if eps < eps_min and cl_corr > 0:
                    optim_vals = [lift_coef, Cd_ret, alpha_ret, eps, Wc]
                    eps_min = eps

            # Save the optimum config of the blade station
            Cl[station] = optim_vals[0]
            Cd[station] = optim_vals[1]
            alpha[station] = optim_vals[2]
            E[station] = optim_vals[3]

            local_Cl = optim_vals[0]
            local_Cd = optim_vals[1]
            local_AoA = optim_vals[2]
            local_eps = optim_vals[3]
            Wc = optim_vals[4]

        # Smooth the Cl distribution and recalculate the lift coefficient: Polinomial regression for smooth distribution
        coef_cl = np.polynomial.polynomial.polyfit(stations_r, Cl, 1)
        cl_fun = np.polynomial.polynomial.Polynomial(coef_cl)

        Cl = cl_fun(stations_r)

        # Calculate product of local speed with chord
        Wc = self.Wc(F, phis, zeta, Cl)
        # Wc = self.Wc(F, phis, zeta, Cl, stations_r)

        # After smoothing the Cl, get new AoA and E corresponding to such Cls
        for station in range(len(Cl)):
            # lift_coef = lift_coef * self.PG(self.M(stations_r[station]))

            lift_coef = Cl[station]

            # # Calculate product of local speed with chord
            # Wc = self.Wc(F[station], phis[station], zeta, lift_coef)

            # Calculate Reynolds number at the station to look for the correct airfoil datafile
            Reyn = self.RN(Wc[station])

            reyn_lst = []
            for file in os.listdir(self.dir_path):
                reyn_lst.append(int(re.findall(r'\d+', file)[0]))

            reyn_min = np.min(reyn_lst)
            reyn_max = np.min(reyn_lst)

            # Maximum and minimum RN in database
            if Reyn<reyn_min:
                Reyn = reyn_min 
            if Reyn>reyn_max:
                Reyn = reyn_max

            cl_corr = (lift_coef * self.PG(self.M(stations_r[station]))) # Corrected cl for compressibility
            Cd_ret = self.cd_interp([[cl_corr, Reyn]])
            alpha_ret = np.deg2rad(self.alpha_interp([[cl_corr, Reyn]]))       # Retrieved AoA (from deg to rad)

            # Compute D/L ration
            eps = Cd_ret / lift_coef

            # Update arrays with values
            Cd[station] = Cd_ret
            alpha[station] = alpha_ret
            E[station] = eps

        # Calculate interference factors
        a = (zeta/2) * (np.cos(phis))**2 * (1 - E*np.tan(phis))
        a_prime = (zeta/(2*self.x(stations_r))) * np.cos(phis) * np.sin(phis) * \
                  (1 + E/np.tan(phis))

        # Calculate local speed at the blade station
        W = self.V * (1 + a) / np.sin(phis)

        # Calculate required chord of the station and save to array
        c = Wc/W
        cs = c

        # Calculate blade pitch angle as AoA+phi and save to array
        beta = alpha + phis
        betas = beta

        # Use average epsilon, independent of r/R (xi), to simplify calculations, as it is very similar in all stations
        eps_avg = np.average(E)

        # Integrate the derivatives from xi_0 to 1 (from hub to tip of the blade)
        I1 = spint.quad(self.I_prim_1, self.xi_0, 1, args=(zeta, eps_avg))[0]
        I2 = spint.quad(self.I_prim_2, self.xi_0, 1, args=(zeta, eps_avg))[0]
        J1 = spint.quad(self.J_prim_1, self.xi_0, 1, args=(zeta, eps_avg))[0]
        J2 = spint.quad(self.J_prim_2, self.xi_0, 1, args=(zeta, eps_avg))[0]
        # Calculate solidity per station
        solidity = cs * self.B / (2 * np.pi * stations_r)

        # Calculate new speed ratio and Tc or Pc as required
        if self.Tc is not None:
            zeta_new = (I1/(2*I2)) - ((I1/(2*I2))**2 - self.Tc/I2)**(1/2)
            Pc = J1*zeta_new + J2*zeta_new**2

            # Propeller efficiency
            eff = self.efficiency(self.Tc, Pc)

            return zeta_new, [cs, betas, alpha, stations_r, E, eff, self.Tc, Pc], Ves, [Cl, Cd], solidity

        elif self.Pc is not None:
            zeta_new = -(J1/(2*J2)) + ((J1/(2*J2))**2 + self.Pc/J2)**(1/2)
            Tc = I1*zeta_new - I2*zeta_new**2

            # Propeller efficiency
            eff = self.efficiency(Tc, self.Pc)

            return zeta_new, [cs, betas, alpha, stations_r, E, eff, Tc, self.Pc], Ves, [Cl, Cd], solidity

    def optimise_blade(self, zeta_init):
        convergence = 1
        zeta = zeta_init
        # Optimisation converges for difference in zeta below 0.1%
        while convergence > 0.001:
            # Run BEM design procedure and retrieve new zeta
            design = self.run_BEM(zeta)
            zeta_new = design[0]

            # Check convergence
            if zeta == 0:
                convergence = np.abs(zeta_new - zeta)
            else:
                convergence = np.abs(zeta_new - zeta)/zeta

            zeta = zeta_new
        #
        design = self.run_BEM(zeta)
        return design




# Analyse the propeller in off-design conditions
class OffDesignAnalysisBEM:
    def __init__(self, dir_path:str, propclass: Propeller, V: float, B: int, R: float, chords: np.array, betas: np.array, r_stations: np.array,
                 Cls: np.array, Cds: np.array, rpm: float, rho: float, dyn_vis: float, a: float, RN: np.array,
                 RN_spacing=100000):
        self.V = V
        self.B = B
        self.R = R
        self.D = 2*R

        self.chords = chords
        self.betas = betas
        self.Cls = Cls
        self.Cds = Cds
        self.r_stations = r_stations

        self.rpm = rpm
        self.Omega = rpm * 2 * np.pi / 60  # rpm to rad/s
        self.n = self.Omega / (2 * np.pi)
        self.lamb = V / (self.Omega * R)  # Speed ratio

        self.J = V / (self.n * self.D)

        self.rho = rho
        self.dyn_vis = dyn_vis
        self.a = a
        self.cl_interp = cl_xfoil_interp(dir_path)
        self.cd_interp = cd_xfoil_interp(dir_path)


    # Prandtl relation for tip loss
    def F(self, r, phi_t):
        return (2 / np.pi) * np.arccos(np.exp(-self.f(r, phi_t)))

    # Exponent used for function above
    def f(self, r, phi_t):
        return (self.B / 2) * (1 - self.Xi(r)) / (np.sin(phi_t))

    # def phi_t(self):
    #     return 1

    # Non-dimensional radius, r/R
    def Xi(self, r):
        return r/self.R


    # Mach as a function of radius
    def M(self, W):
        return W/self.a

    # Reynolds number
    def RN(self, W, c):
        # Reynolds number. Wc is speed times chord
        return W * c * self.rho / self.dyn_vis

    def W(self, a, a_prim, r):
        return np.sqrt((self.V * (1 + a))**2 + (self.Omega * r * (1 - a_prim))**2)

    # Cx and Cy coefficients from Cl and Cd
    def Cy(self, Cl, Cd, phi):
        return Cl * np.cos(phi) - Cd * np.sin(phi)

    def Cx(self, Cl, Cd, phi):
        return Cl * np.sin(phi) + Cd * np.cos(phi)

    # Local solidity of a blade element
    def solidity_local(self, c, r):
        return self.B * c / (2 * np.pi * r)

    # Variables used in interference factors
    def K(self, Cl, Cd, phi):
 
        return self.Cy(Cl, Cd, phi) / (4 * (np.sin(phi))**2)

    def K_prim(self, Cl, Cd, phi):
        return self.Cx(Cl, Cd, phi) / (4 * np.sin(phi) * np.cos(phi))

    # Interference factors
    def a_fac(self, Cl, Cd, phi, c, r, phi_t):
        sigma = self.solidity_local(c, r)  # Local solidity
        K = self.K(Cl, Cd, phi)

        # From Viterna and Janetzke
        sign = np.sign(sigma * K / (self.F(r, phi_t) - sigma*K))
        # print(sign)
        # if any(sign) < 0:
        #     print("a sign negative")

        magnitude = np.minimum(np.abs(sigma * K / (self.F(r, phi_t) - sigma*K)), 0.7)

        # TODO: check sign of a
        # print("a:", omega * K / (self.F(r, zeta) - omega*K))
        return magnitude  # *sign
        # return np.abs(sigma * K / (self.F(r, phi_t) - sigma * K))

    def a_prim_fac(self, Cl, Cd, phi, c, r, phi_t):
        sigma = self.solidity_local(c, r)  # Local solidity
        K_prim = self.K_prim(Cl, Cd, phi)

 

        sign = sigma * K_prim / (self.F(r, phi_t) + sigma * K_prim)

        # if any(sign) < 0:
        #     print("a' sign negative")

        magnitude = np.minimum(np.abs(sigma * K_prim / (self.F(r, phi_t) + sigma * K_prim)), 0.7)

        return magnitude  # *sign
        # return np.abs(sigma * K_prim / (self.F(r, phi_t) + sigma * K_prim))

    def phi(self, a, a_prim, r):
        return np.arctan(self.V * (1 + a) / (self.Omega * r * (1 - a_prim)))

    def C_T(self, T):
        return T / (self.rho * self.n**2 * self.D**4)

    def C_P(self, P):
        return P / (self.rho * self.n**3 * self.D**5)

    # Differential forms wrt xi
    def C_T_prim(self, r, c, Cl, Cd, F, K_prim, phi):
        return (np.pi**3 / 4) * self.solidity_local(c, r) * self.Cy(Cl, Cd, phi) * (r/self.R) * \
               self.F(r, phi[-1])**32 / ((F + self.solidity_local(c, r)*K_prim) * np.cos(phi))**2

    def C_P_prim(self, r, c, Cl, Cd, F, K_prim, phi):
        return self.C_T_prim(r, c, Cl, Cd, F, K_prim, phi) * np.pi * (r/self.R) * self.Cx(Cl, Cd, phi) / \
               self.Cy(Cl, Cd, phi)

 

    """
    End of the functions used for integration
    """

    def eff(self, C_T, C_P):
        return C_T * self.J / C_P

    # Prandtl-Glauert correction factor: sqrt(1 - M^2)
    def PG_correct(self, M):
        return np.sqrt(1 - M**2)

    def analyse_propeller(self):
        # Initial estimate for phi and zeta
        phi = np.arctan(self.lamb / self.Xi(self.r_stations))

        alphas = self.betas - phi

        # Get initial estimate of CL and Cd per station
        Cls = np.ones(len(self.r_stations))
        Cds = np.ones(len(self.r_stations))

        """
        Start
        """
        Reyn = self.RN_init

        for station in range(len(Reyn)):

            RN = self.RN_spacing * round(Reyn[station] / self.RN_spacing)

            # Maximum and minimum RN in database
            if RN < 100000:
                RN = 100000
            if RN > 5000000:
                RN = 5000000

            # Look for corresponding airfoil data file for that RN
            filename1 = "4412_Re%d_up.txt" % RN
            filename2 = "4412_Re%d_dwn.txt" % RN


            file_up = open(os.path.join(os.path.dirname(__file__),"data",filename1), "r")
            file_down = open(os.path.join(os.path.dirname(__file__),"data",filename2), "r")
            # Read lines
            lines_up = file_up.readlines()
            lines_down = file_up.readlines()

            # Close files
            file_up.close()
            file_down.close()

            # List and Boolean to save relevant lines
            format_lines = []
            save_lines = False

            for line in lines_up:
                # Separate variables inside file
                a = line.split()

                # If the save_lines boolean is True (when the code gets to numerical values), save to list
                if save_lines:
                    # Create a line with floats (instead of strings) to append to main list
                    new_line = []
                    for value in a:
                        new_line.append(float(value))
                    format_lines.append(new_line)

                # Protect against empty lines
                if len(a) > 0:
                    # There is a line with ---- before the numbers, so once we get to this line, start saving
                    if a[0].count('-') >= 1:
                        save_lines = True

            # Restart boolean for down file
            save_lines = False

            # Do the same process for down file and append to the same array as up
            for line in lines_down:
                a = line.split()

                if save_lines:
                    new_line = []
                    for value in a:
                        new_line.append(float(value))
                    format_lines.append(new_line)

                if len(a) > 0:
                    if a[0].count('-') >= 1:
                        save_lines = True

            # Convert to numpy array with airfoil data
            airfoil_data = np.array(format_lines)

            # ------------ Format of each line --------------
            # alpha, CL, CD, Re(CL), CM, S_xtr, P_xtr, CDp

            # Order airfoil data by angle of attack, this can be eliminated to save time if needed
            airfoil_data = airfoil_data[airfoil_data[:, 0].argsort()]

            # Save airfoil data array to have a copy to modify
            airfoil_data_check = airfoil_data

            # Subtract current AoA from list of AoAs
            # Note that the AoA in the files is in degrees
            airfoil_data_check[:, 0] -= np.rad2deg(alphas[station])

            # Check what line has min AoA difference, and retrieve index of that column
            index = np.argmin(np.abs(airfoil_data_check[:, 0]))

            # Obtain the Cl and Cd from the line where Cl difference is min
            # Correct the Cl/Cd obtained for Mach number
            Cl_ret = airfoil_data[index, 1] / self.PG_correct(self.M(self.Omega*self.r_stations[station]))  # Retrieved Cl
            Cd_ret = airfoil_data[index, 2] / self.PG_correct(self.M(self.Omega*self.r_stations[station]))  # Retrieved Cd

            # Update the Cl and Cd at each station
            Cls[station] = Cl_ret
            Cds[station] = Cd_ret

        """
        End
        """

        # Calculate initial estimates for the interference factors
        a_facs = self.a_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)
        a_prims = self.a_prim_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)

        # Iterate to get a convergent analysis
        count = 0
        iterate = True
        while iterate or (count < 10):
            # Calculate AoA of the blade stations
            alphas = self.betas - phi

            # Calculate the speed
            Ws = self.W(a_facs, a_prims, self.r_stations)

            # Calculate the Reynolds number
            Re = self.RN(Ws, self.chords)
            for station in range(len(self.r_stations)):
                # Get the Reynold's number per station
                RN = Re[station]
                RN = self.RN_spacing * round(RN / self.RN_spacing)

                # Maximum and minimum RN in database
                if RN<100000:
                    RN = 100000
                if RN>5000000:
                    RN = 5000000

                # Look for corresponding airfoil data file for that RN
                filename1 = "4412_Re%d_up.txt" % RN
                filename2 = "4412_Re%d_dwn.txt" % RN

                file_up = open(os.path.join(os.path.dirname(__file__),"data",filename1), "r")
                file_down = open(os.path.join(os.path.dirname(__file__),"data",filename2), "r")
                # Read lines
                lines_up = file_up.readlines()
                lines_down = file_up.readlines()

                # Close files
                file_up.close()
                file_down.close()

                # List and Boolean to save relevant lines
                format_lines = []
                save_lines = False

                for line in lines_up:
                    # Separate variables inside file
                    a = line.split()

                    # If the save_lines boolean is True (when the code gets to numerical values), save to list
                    if save_lines:
                        # Create a line with floats (instead of strings) to append to main list
                        new_line = []
                        for value in a:
                            new_line.append(float(value))
                        format_lines.append(new_line)

                    # Protect against empty lines
                    if len(a) > 0:
                        # There is a line with ---- before the numbers, so once we get to this line, start saving
                        if a[0].count('-') >= 1:
                            save_lines = True

                # Restart boolean for down file
                save_lines = False

                # Do the same process for down file and append to the same array as up
                for line in lines_down:
                    a = line.split()

                    if save_lines:
                        new_line = []
                        for value in a:
                            new_line.append(float(value))
                        format_lines.append(new_line)

                    if len(a) > 0:
                        if a[0].count('-') >= 1:
                            save_lines = True

                # Convert to numpy array with airfoil data
                airfoil_data = np.array(format_lines)

                # ------------ Format of each line --------------
                # alpha, CL, CD, Re(CL), CM, S_xtr, P_xtr, CDp

                # Order airfoil data by angle of attack, this can be eliminated to save time if needed
                airfoil_data = airfoil_data[airfoil_data[:, 0].argsort()]

                # Save airfoil data array to have a copy to modify
                airfoil_data_check = airfoil_data

                # Subtract current AoA from list of AoAs
                # Note that the AoA in the files is in degrees
                airfoil_data_check[:, 0] -= np.rad2deg(alphas[station])

                # Check what line has min AoA difference, and retrieve index of that column
                index = np.argmin(np.abs(airfoil_data_check[:, 0]))

                # Obtain the Cl and Cd from the line where Cl difference is min
                # Correct the Cl/Cd obtained for Mach number
                Cl_ret = airfoil_data[index, 1]/self.PG_correct(self.M(Ws[station]))  # Retrieved Cl
                Cd_ret = airfoil_data[index, 2]/self.PG_correct(self.M(Ws[station]))  # Retrieved Cd

                Cds[station] = Cd_ret

            # Update the interference factors
            a_facs = self.a_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)
            a_prims = self.a_prim_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)

            # Update phi
            phi_new = self.phi(a_facs, a_prims, self.r_stations)

            # Check convergence of the phi angles
            conv = np.abs((phi - phi_new) / phi)
          
            if np.average(conv) > 0.03:
                # print("### conv", conv, np.average(conv))
                pass
            else:
                iterate = False

            # Update the phi angles
            phi = phi_new

            count += 1

        # Force coefficients
        Cx = self.Cx(Cls, Cds, phi)
        Cy = self.Cy(Cls, Cds, phi)

        # Thrust and torque per unit radius
        T_prim = 0.5 * self.rho * Ws**2 * self.B * self.chords * Cy
        Q_prim_r = 0.5 * self.rho * Ws**2 * self.B * self.chords * Cx

        # Do simple integration to get total thrust and Q per unit r
        T = spint.trapz(T_prim, self.r_stations)
        Q = spint.trapz(Q_prim_r * self.r_stations, self.r_stations)


        C_T_prim = T_prim/(self.rho * self.n**2 * self.D**4)
        C_T = spint.trapz(C_T_prim, self.r_stations)

        C_P_prim = C_T_prim * np.pi * self.r_stations/self.R * Cx/Cy
        C_P = spint.trapz(C_P_prim, self.r_stations)

        eff = self.eff(C_T, C_P)
        return [T, Q, eff], [C_T, C_P], [alphas, Cls, Cds]


class Optiblade:
    def __init__(self, B, R, rpm_cr, xi_0, rho_cr, dyn_vis_cr, V_cr, N_stations, a_cr, RN_spacing, max_M_tip, rho_h,
                 dyn_vis_h, V_h, a_h, rpm_h, delta_pitch, T_cr, T_h):
        """
        This file is for the blade shape optimisation. It optimises the blade for cruise, and checks if the blade can
        meet hover thrust requirements. If not, it designs the blade for higher thrust levels and checks again, up until
        we have a design that works in hover but is optimised for a condition as close to cruise as possible

                                            Propeller
        :param B: Number of blades [-]
        :param R: Outer radius of propeller [m]
        :param xi_0: Non-dimensional hub radius (r_hub/R) [-]
        :param N_stations: Number of stations to calculate [-]
        :param RN_spacing: Spacing of the Reynold's numbers of the airfoil data files [-] (added for flexibility,
                           but probably should be 100,000)

                                        Cruise conditions
        :param rpm_cr: rpm of the propeller during cruise [rpm]
        :param rho_cr: Density [kg/m^3]
        :param dyn_vis_cr: Dynamic viscosity [N s/m^2)
        :param V_cr: Freestream velocity
        :param T_cr: Thrust delivered BY the propeller [N]
        :param P_cr: Power delivered TO the propeller [W]
        :param a_cr: Speed of sound [m/s]

                                        Hover conditions
        :param max_rpm_h: Max allowable rpm of the propeller during hover [rpm]
        :param rho_h: Density [kg/m^3]
        :param dyn_vis_h: Dynamic viscosity [N s/m^2)
        :param V_h: Freestream velocity
        :param T_h: Thrust delivered BY the propeller [N]
        :param P_h: Power delivered TO the propeller [W]
        :param a_h: Speed of sound [m/s]
        :param max_M_tip: Maximum allowable tip Mach [-]
        :param rpm_h: rpm of the blade for hover (optimisation variable) [rpm]
        :param delta_pitch: Change in pitch of the blade to obtain more thrust (optimisation variable) [rad]
        """
        self.B = B
        self.R = R
        # self.D = 2 * R
        self.RN_spacing = RN_spacing
        self.N_s = N_stations
        self.xi_0 = xi_0

        self.rho_cr = rho_cr
        self.dyn_vis_cr = dyn_vis_cr
        self.V_cr = V_cr
        self.rpm_cr = rpm_cr
        self.Omega_cr = rpm_cr * 2 * np.pi / 60  # rpm to rad/s
        self.lamb_cr = V_cr / (self.Omega_cr * R)  # Speed ratio
        self.a_cr = a_cr
        self.T_cr = T_cr

        self.rho_h = rho_h
        self.dyn_vis_h = dyn_vis_h
        self.V_h = V_h
        self.a_h = a_h
        self.T_h = T_h
        self.max_M_tip = max_M_tip
        self.rpm_h = rpm_h
        self.delta_pitch = delta_pitch

    def blade_design(self, design_thrust_factor):
        # Design the propeller for given conditions
        propeller = BEM(self.B, self.R, self.rpm_cr, self.xi_0, self.rho_cr, self.dyn_vis_cr, self.V_cr, self.N_s,
                        self.a_cr, self.RN_spacing, T=self.T_cr * design_thrust_factor)

        # Zeta to initialise the procedure
        zeta_init = 0
        zeta, design, V_e, coefs = propeller.optimise_blade(zeta_init)

        return zeta, design, V_e, coefs



    def optimised_blade(self):
        # Multiply design (cruise) drag times factor
        thrust_factors = np.arange(1, np.floor(self.T_h / self.T_cr))

        # Check if the design can meet thrust requirements, else design it or higher thrust
        index = 0
        max_thrust = 0
        while max_thrust < self.T_h:
            thrust_factor = thrust_factors[index]

            # Design optimum blade for the thrust condition
            blade = self.blade_design(thrust_factor)

            # Check whether the max thrust is enough, once it is the loop ends
            max_thrust = self.max_T_check(blade)

            index += 1

        # Now we have a blade design that can provide the necessary max thrust
        # Analyse the blade at optimum hover condition to optimise
        blade_hover = OffDesignAnalysisBEM(self.V_h, self.B, self.R, blade[1][0], blade[1][1]-self.delta_pitch,
                                           blade[1][3], blade[3][0], blade[3][1], self.rpm_h,
                                           blade[0], self.rho_h, self.dyn_vis_h, self.a_h).analyse_propeller()

        # Return [blade in cruise], [T, Q, eff] of blade in hover, and thrust factor at which the blade is designed
        # The cost function should maximise both cruise and hover efficiency, so [0][1][5] and [1][2] TODO: check
        return blade, blade_hover, thrust_factor


