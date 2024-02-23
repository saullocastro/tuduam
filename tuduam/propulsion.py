import re
from typing import List
import plotly.graph_objs as go
import numpy as np
import os
import pdb
from tuduam.data_structures import Propeller
import matplotlib.pyplot as plt
import scipy.integrate as spint
from scipy.interpolate import NearestNDInterpolator
from warnings import warn


def extract_data_dir(dir_path:str) -> np.ndarray:
    """ This function pulls data from multiple files within a directory as outputted by Xfoil and puts them in one array. Information on how to do this can be found in the notebooks. 

    :param dir_path: The path of the directory to read from can be absolute or relative
    :type dir_path: str
    :return: An  m x 8 array where the colums are the following: [alpha, CL, CD, CDp, CM, Top_xtr, bot_xtr, Reynolds number]
    :rtype: np.ndarray
    """    
    data_points: List[List[float]] = list()

    for file in os.listdir(dir_path):
        try:
            reyn = float(re.findall(r'Re(\d+)', file, re.IGNORECASE)[0])
        except IndexError:
            raise Exception("One of the files is likely in the incorrect format. Please check wheter the Reynolds number has Re infront of it")
        with open(os.path.join(dir_path,file), "r") as f:
            write = False
            for line in f.readlines():
                if  line.count("-") > 10:  # increase the threshold to a number larger than the number of columns in the file
                    write = True
                    continue
                if line == '\n': # skip any empty line
                    continue
                if write:
                    value_lst: List[float]  = [float(value) for value in line.split()]
                    value_lst.append(reyn)
                    data_points.append(value_lst)
    return np.array(data_points)


def alpha_xfoil_interp(dir_path:str) -> NearestNDInterpolator:
    """ summary

    :param dir_path: _description_
    :type dir_path: str
    :return: _description_
    :rtype: NearestNDInterpolator
    """    

    raw_data =  extract_data_dir(dir_path)
    return NearestNDInterpolator(raw_data[:,[1,-1]], raw_data[:,0])

def cl_xfoil_interp(dir_path:str) -> NearestNDInterpolator:
    """  summary

    :param dir_path: Directory of the files containting the xfoil polar
    :type dir_path: str
    :return: A interpolator with the input angle of attack and the Reynolds number. In that order.
    :rtype: _type_
    """    

    raw_data =  extract_data_dir(dir_path)
    return NearestNDInterpolator(raw_data[:,[0,-1]], raw_data[:,1])


def cd_xfoil_interp(dir_path:str) -> NearestNDInterpolator:
    """ 

    :param dir_path: Directory of the files containting the xfoil polar
    :type dir_path: str
    :return: A interpolator with the input cl and the Reynolds number. In that order.
    :rtype: _type_
    """    
    raw_data =  extract_data_dir(dir_path)
    return NearestNDInterpolator(raw_data[:,[1,-1]], raw_data[:,2])


class PlotBlade:
    def __init__(self, propclass:Propeller, path_coord:str) -> None:
        """ Initialization of the plot class. The propeller data structures is required to be fully filled,
        also the thickness over chord ratio.

        :param propclass: Propeller class of which all the attributes are defined.
        :type propclass: Propeller
        :param path_coord: Path to the coordinates of the airfoil in the format starting at the top trailing edge,
        moving to the top leading edge and then looping back to the bottom trailinng edge. 
        :type path_coord: str
        """        
        self.chords = propclass.chord_arr
        self.pitchs = propclass.pitch_arr
        self.radial_coords = propclass.rad_arr
        self.R = propclass.r_prop
        self.xi_0 = propclass.xi_0
        self.tc_ratio = propclass.tc_ratio
        self.path_coord = path_coord

    def _load_airfoil(self) -> np.ndarray:
        """ Returns an array Using a path to coordinate file of the airfoil

        :return: Returns an array with the airfoil coordinates
        :rtype: np.ndarray
        """        

        file = open(self.path_coord)
        airfoil = file.readlines()
        file.close()

        # List to save formatted coordinates
        airfoil_coord = []

        for line in airfoil:
            if any([i.isalpha() for i in line]):
                continue
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

    def plot_blade(self,tst=False):
        """ Returns two plots, one top down view of the propeller showing the amount of twist and the various chords
        and one plot showing a side view of the propeller

        :param tst: A boolean which is used for testing to surpess the output, defaults to False
        :type tst: bool, optional
        """        
        # Create figures
        fig, axs = plt.subplots(2, 1)
        axs[0].axis('equal')

        # Plot side view of the airfoil cross-sections
        for i in range(len(self.chords)):
            # Scale the chord length and thickness
            x_coords = self._load_airfoil()[0] * self.chords[i]
            y_coords = self._load_airfoil()[1] * self.chords[i]

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
        axs[0].set_xlabel("Disk Plane [m]")
        axs[0].set_ylabel("Longitudinal direction [m]")

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
        axs[1].set_xlabel("Radial direction [m]")
        axs[1].set_ylabel("Tip-path direction [m]")

        # Plot smooth distribution  TODO: revise
        radius = np.linspace(self.xi_0*self.R, self.R, 200)
        axs[1].plot(radius, y_min_fun(radius), label= "Lower Edge")
        axs[1].plot(radius, y_max_fun(radius), label= "Upper Edge")

        axs[0].legend()
        axs[1].legend()
        if not tst:
            plt.show()

    def plot_3D(self,tst=False):
        """ Plot a 3D plot of one propeller blade. The user can drag his mouse around to see
        the blade from various angles.

        :param tst: A boolean which is used for testing to surpess the output, defaults to False
        :type tst: bool, optional
        """        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # ax.set_aspect('equal')

        # Plot airfoil blade in 3D
        for i in range(len(self.chords)):
            # Scale the chord length and thickness
            x_coords = self._load_airfoil()[0] * self.chords[i]
            y_coords = self._load_airfoil()[1] * self.chords[i]

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
        if not tst:
            plt.show()


    def plot_3D_plotly(self, tst=False):
        fig = go.Figure()

        for i in range(len(self.chords)):
            x_coords = self._load_airfoil()[0] * self.chords[i]
            y_coords = self._load_airfoil()[1] * self.chords[i]
            x_coords_n = []
            y_coords_n = []

            # Apply pitch
            for j in range(len(x_coords)):
                x_coord_n = np.cos(self.pitchs[i]) * x_coords[j] + np.sin(self.pitchs[i]) * y_coords[j]
                y_coord_n = -np.sin(self.pitchs[i]) * x_coords[j] + np.cos(self.pitchs[i]) * y_coords[j]

                x_coords_n.append(x_coord_n)
                y_coords_n.append(y_coord_n)

            fig.add_trace(go.Scatter3d(
                x=x_coords_n,
                y=y_coords_n,
                z=np.full(len(x_coords_n), self.radial_coords[i]),
                mode='lines',
                line=dict(color='black'),
                showlegend=False
            ))

        # Trick to set 3D axes to equal scale, obtained from:
        # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to

        X = np.array([self.chords[0], self.chords[-1]])
        Y = np.array([self.chords[0]*self.tc_ratio, self.chords[-1]*self.tc_ratio])
        Z = np.array([0, self.radial_coords[-1]])
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
        for xb, yb, zb in zip(Xb, Yb, Zb):
            fig.add_trace(go.Scatter3d(
                x=[xb],
                y=[yb],
                z=[zb],
                mode='markers',
                marker=dict(color='white', size=0.1),
                showlegend=False
            ))

        # Set layout
        fig.update_layout(scene=dict(aspectmode='data'))
        
        if not tst:
            fig.show()

class BEM:
    def __init__(self, data_path:str, propclass:Propeller, rho:float, dyn_vis:float, 
                 V_fr:float, n_stations:int, a:float, T=None, P=None) -> None:
        """ Initalization of the BEM class, please keep the amount of station above 20. Also, if any error occurs with the propeller class
        carefully check whether all parameters have been properly loaded in. Please specify either a thrust level or
        power level.

        :param data_path: Path to te directory containing all xfoil data for the various Reynolds number. Please note the format of the 
        file names given to polar files. They should only contain the Reynolds number in the name, no other numbers such as the airfoil code.
        :type data_path: str
        :param propclass: The propeller data structure with the propeller radius, number of blades, rpm cruise and non-dimensional hub radius specified
        :type propclass: Propeller
        :param rho: Density at the cruise height [kg/m^3]
        :type rho: float
        :param dyn_vis: Dynamic viscosity [N s/m^2]
        :type dyn_vis: float
        :param V_fr: Freestream velocity [m/s]
        :type V_fr: float
        :param N_stations: Number of stations to calculate [-] (preferably > 20)
        :type N_stations: int
        :param a: Speed of sound [m/s]
        :type a: float
        :param T: Thrust delivered BY the propeller [N], defaults to None
        :type T: float , optional
        :param P: Power deliverd to the propeller [W], defaults to None
        :type P: float, optional
        """        

        self.propeller = propclass
        self.B = propclass.n_blades
        self.R = propclass.r_prop
        self.D = 2*self.R
        self.Omega = propclass.rpm_cruise*2 * np.pi / 60  # rpm to rad/s
        self.xi_0 = propclass.xi_0
        self.rho = rho
        self.dyn_vis = dyn_vis
        self.V = V_fr
        # self.phi_T = 1
        self.lamb = V_fr/(self.Omega*self.R)  # Speed ratio
        self.N_s = n_stations
        self.a = a
        self.dir_path = data_path
        self.alpha_interp = alpha_xfoil_interp(data_path)
        self.cd_interp = cd_xfoil_interp(data_path)

        # Define thrust or power coefficients, depending on input
        if T is not None and P is None:
            self.Tc = 2 * T / (rho * V_fr**2 * np.pi * self.R**2)
            self.Pc = None

        elif P is not None and T is None:
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
        stations_arr = self.xi_0*self.R + st_len/2 + (stations-1)*st_len
        # stations_r = self.xi_0*self.R + (stations)*st_len
        # F and phi for each station
        F = self.F(stations_arr, zeta)
        phis = self.phi(stations_arr, zeta)

        # trial with a different range of Cls
        Cls_trial = np.arange(0.1, 1.2, 0.05)

        # Create arrays for lift and drag coefficients, angle of attack and D/L ratio for each station
        cl_arr = np.ones(self.N_s)
        cd_arr = np.ones(self.N_s)
        alpha_arr = np.ones(self.N_s)
        eps_arr = np.ones(self.N_s)
        chord_arr = np.ones(self.N_s)
        beta_arr = np.ones(self.N_s)
        v_e = zeta*self.V + self.V

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

                # Maximum and minimum RN in database
                if Reyn<self.cd_interp.tree.mins[1] and Reyn > 5e4:
                    warn(f"A Reynolds number of {Reyn:.3e} was encountered, lower than the minimum {self.cd_interp.tree.mins[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closest neighbour.", category=RuntimeWarning)
                if Reyn>self.cd_interp.tree.maxes[1]:
                    warn(f"A Reynolds number of {Reyn:.3e} was encountered, higher than the maximum {self.cd_interp.tree.maxes[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closes neighbour.", category=RuntimeWarning)

                cl_corr = (lift_coef * self.PG(self.M(stations_arr[station]))) # Corrected cl for compressibility
                Cd_ret = (self.cd_interp([[cl_corr, Reyn]])/self.PG(self.M(stations_arr[station])))[0]
                alpha_ret = np.deg2rad(self.alpha_interp([[cl_corr, Reyn]]))[0]       # Retrieved AoA (from deg to rad)

                if cl_corr > self.alpha_interp.tree.maxes[0]:
                    warn(f"A Cl {cl_corr} was encountered, higher than the max {self.alpha_interp.tree.maxes[0]:.3e} in the data set was encounterd")

                # Compute D/L ration
                eps = Cd_ret / lift_coef

                # See if D/L is minimum. If so, save the values
                if eps < eps_min and cl_corr > 0:
                    optim_vals = [lift_coef, Cd_ret, alpha_ret, eps, Wc]
                    eps_min = eps

            # Save the optimum config of the blade station
            cl_arr[station] = optim_vals[0]
            cd_arr[station] = optim_vals[1]
            alpha_arr[station] = optim_vals[2]
            eps_arr[station] = optim_vals[3]

            local_Cl = optim_vals[0]
            local_Cd = optim_vals[1]
            local_AoA = optim_vals[2]
            local_eps = optim_vals[3]
            Wc = optim_vals[4]

        # Smooth the Cl distribution and recalculate the lift coefficient: Polinomial regression for smooth distribution
        coef_cl = np.polynomial.polynomial.polyfit(stations_arr, cl_arr, 1)

        cl_fun = np.polynomial.polynomial.Polynomial(coef_cl)

        cl_arr = cl_fun(stations_arr)

        # Calculate product of local speed with chord
        Wc = self.Wc(F, phis, zeta, cl_arr)
        # Wc = self.Wc(F, phis, zeta, Cl, stations_r)

        # After smoothing the Cl, get new AoA and E corresponding to such Cls
        for station in range(len(cl_arr)):
            # lift_coef = lift_coef * self.PG(self.M(stations_r[station]))

            lift_coef = cl_arr[station]

            # # Calculate product of local speed with chord
            # Wc = self.Wc(F[station], phis[station], zeta, lift_coef)

            # Calculate Reynolds number at the station to look for the correct airfoil datafile
            Reyn = self.RN(Wc[station])

            # Maximum and minimum RN in database
            if Reyn<self.cd_interp.tree.mins[1] and Reyn > 5e4:
                warn(f"A Reynolds number of {Reyn:.3e} was encountered, lower than the minimum {self.cd_interp.tree.mins[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closest neighbour.", category=RuntimeWarning)
            if Reyn>self.cd_interp.tree.maxes[1]:
                warn(f"A Reynolds number of {Reyn:.3e} was encountered, higher than the maximum {self.cd_interp.tree.maxes[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closes neighbour.", category=RuntimeWarning)
       

            cl_corr = (lift_coef * self.PG(self.M(stations_arr[station]))) # Corrected cl for compressibility
            Cd_ret = (self.cd_interp([[cl_corr, Reyn]])/self.PG(self.M(stations_arr[station])))[0]
            alpha_ret = np.deg2rad(self.alpha_interp([[cl_corr, Reyn]]))[0]       # Retrieved AoA (from deg to rad)

            if cl_corr > self.alpha_interp.tree.maxes[0]:
                warn(f"A Cl {cl_corr} was encountered, higher than the max {self.alpha_interp.tree.maxes[0]:.3e} in the data set was encounterd")

            # Compute D/L ration
            eps = Cd_ret / lift_coef

            # Update arrays with values
            cd_arr[station] = Cd_ret
            alpha_arr[station] = alpha_ret
            eps_arr[station] = eps

        # Calculate interference factors
        a = (zeta/2) * (np.cos(phis))**2 * (1 - eps_arr*np.tan(phis))
        a_prime = (zeta/(2*self.x(stations_arr))) * np.cos(phis) * np.sin(phis) * \
                  (1 + eps_arr/np.tan(phis))

        # Calculate local speed at the blade station
        W = self.V * (1 + a) / np.sin(phis)

        # Calculate required chord of the station and save to array
        chord_arr = Wc/W

        # Calculate blade pitch angle as AoA+phi and save to array
        beta_arr = alpha_arr + phis

        # Use average epsilon, independent of r/R (xi), to simplify calculations, as it is very similar in all stations
        eps_avg = np.average(eps_arr)

        # Integrate the derivatives from xi_0 to 1 (from hub to tip of the blade)
        I1 = spint.quad(self.I_prim_1, self.xi_0, 1, args=(zeta, eps_avg))[0]
        I2 = spint.quad(self.I_prim_2, self.xi_0, 1, args=(zeta, eps_avg))[0]
        J1 = spint.quad(self.J_prim_1, self.xi_0, 1, args=(zeta, eps_avg))[0]
        J2 = spint.quad(self.J_prim_2, self.xi_0, 1, args=(zeta, eps_avg))[0]
        # Calculate solidity per station
        solidity = chord_arr * self.B / (2 * np.pi * stations_arr)

        res_dict = {
            "chord_arr":chord_arr,
            "pitch_arr": beta_arr,
            "alpha_arr": alpha_arr,
            "station_arr": stations_arr,
            "drag_to_lift_arr": eps_arr, 
            "v_e": v_e,
            "solidity": solidity,
            "cl": cl_arr,
            "cd": cd_arr,
            }
        
        self.propeller.chord_arr = chord_arr
        self.propeller.pitch_arr = beta_arr 
        self.propeller.rad_arr =  stations_arr

        # Calculate new speed ratio and Tc or Pc as required
        if self.Tc is not None and self.Pc is None:
            zeta_new = (I1/(2*I2)) - ((I1/(2*I2))**2 - self.Tc/I2)**(1/2)
            Pc = J1*zeta_new + J2*zeta_new**2

            # Propeller efficiency
            eff = self.efficiency(self.Tc, Pc)

            res_dict["eff"] = eff
            res_dict["tc"] = self.Tc
            res_dict["pc"] =  Pc
            res_dict["zeta"] =  zeta_new

            return res_dict

        elif self.Pc is not None and self.Tc is None:
            zeta_new = -(J1/(2*J2)) + ((J1/(2*J2))**2 + self.Pc/J2)**(1/2)
            Tc = I1*zeta_new - I2*zeta_new**2

            # Propeller efficiency
            eff = self.efficiency(Tc, self.Pc)

            res_dict["eff"] = eff
            res_dict["tc"] = Tc
            res_dict["pc"] =  self.Pc
            res_dict["zeta"] =  zeta_new

            return res_dict

    def optimise_blade(self, zeta_init):
        convergence = 1
        zeta = zeta_init
        # Optimisation converges for difference in zeta below 0.1%
        while convergence > 0.001:
            # Run BEM design procedure and retrieve new zeta
            design = self.run_BEM(zeta)
            zeta_new =  design["zeta"]

            # Check convergence
            if zeta == 0:
                convergence = np.abs(zeta_new - zeta)
            else:
                convergence = np.abs(zeta_new - zeta)/zeta

            zeta = zeta_new
        #
        design = self.run_BEM(zeta)
        return design


class OffDesignAnalysisBEM:
    """
    A class encapsulating the arbitrary analysis of a propeller blade as described in Adkins and Liebeck (1994).

    General notes
    ----------------------
    -  Interferences factors were clipped to -0.7 and 0.7. Viterna and Janetzke11 give empirical arguments 
       for clipping the magnitude of a and a' at the value of 0.7 in order to better convergence.  See a_fac and a_prime_fac

    Future improvement
    -----------------------------
    -  TODO: Also compare to sample example in original paper of Adkins and Liebeck (1994)
    - TODO: Refactor such that rpm and V can be changed in the  self.analyse_propeller method. Reinstantiating the class would not be
            necessary in that case
    - TODO: Make the off design analysis robust enough so a scipy.optimize could perhaps be used in the future.

    """    
    def __init__(self, dir_path:str, propclass: Propeller, V: float,
                  rpm: float, rho: float, dyn_vis: float, a: float) -> None:
        self.V = V
        self.B = propclass.n_blades
        self.R = propclass.r_prop
        self.D = 2*propclass.r_prop
        self.dir_path = dir_path

        self.chords = np.array(propclass.chord_arr)
        self.betas = np.array(propclass.pitch_arr)
        self.r_stations = np.array(propclass.rad_arr)

        self.rpm = rpm
        self.Omega = self.rpm * 2 * np.pi / 60  # rpm to rad/s
        self.n = self.Omega / (2 * np.pi)
        self.lamb = V / (self.Omega * self.R)  # Speed ratio

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
    def a_fac(self, Cl:np.ndarray, Cd:np.ndarray, phi:np.ndarray, c:np.ndarray, r:np.ndarray, phi_t:np.ndarray) -> np.ndarray:
        """
        Returns the rotational interferene factor. Note that Viterna and Janetzke11 give empirical arguments 
        for clipping the magnitude of a and a' at the value of 0.7 in order to better convergence.

        Viterna, A., and Janetzke, D., "Theoretical and Experimental Power from Large Horizontal-Axis Wind Turbines," Proceedings from
        the Large Horizontal-Axis Wind Turbine Conference, DOE/NASA-LeRC, July 1981.

        :param Cl: _description_
        :type Cl: np.ndarray
        :param Cd: _description_
        :type Cd: np.ndarray
        :param phi: _description_
        :type phi: np.ndarray
        :param c: _description_
        :type c: np.ndarray
        :param r: _description_
        :type r: np.ndarray
        :param phi_t: _description_
        :type phi_t: np.ndarray
        :return: _description_
        :rtype: np.ndarray
        """        
        sigma = self.solidity_local(c, r)  # Local solidity
        K = self.K(Cl, Cd, phi)
        # From Viterna and Janetzke
        sign = np.sign(sigma * K / (self.F(r, phi_t) - sigma*K))
        magnitude = np.minimum(np.abs(sigma * K / (self.F(r, phi_t) - sigma*K)), 0.7)

        return magnitude  # *sign

    def a_prim_fac(self, Cl, Cd, phi, c, r, phi_t):
        """ 
        Returns the rotational interferene factor. Note that Viterna and Janetzke11 give empirical arguments 
        for clipping the magnitude of a and a' at the value of 0.7 in order to better convergence.

        Viterna, A., and Janetzke, D., "Theoretical and Experimental Power from Large Horizontal-Axis Wind Turbines," Proceedings from
        the Large Horizontal-Axis Wind Turbine Conference, DOE/NASA-LeRC, July 1981.


        :param Cl: _description_
        :type Cl: _type_
        :param Cd: _description_
        :type Cd: _type_
        :param phi: _description_
        :type phi: _type_
        :param c: _description_
        :type c: _type_
        :param r: _description_
        :type r: _type_
        :param phi_t: _description_
        :type phi_t: _type_
        :return: _description_
        :rtype: _type_
        """        
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

    """"
    The function below were in the original source code but were not used so for now have been
    commented out. They will remain here for now


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

    def eff(self, C_T, C_P):
        return C_T * self.J / C_P

    # Prandtl-Glauert correction factor: sqrt(1 - M^2)
    def PG_correct(self, M):
        return np.sqrt(1 - M**2)

    def analyse_propeller(self, delta_pitch:float, max_iter=100, abs_extrapolation=0) -> dict:
        """ Analyse the propeller according to the procedure specified in Adkins and Liebeck (1994), returns a dictionary with the 
        keys as specified below.

        :param delta_pitch: A change in pitch of the entire blade in radians. A positive value will further in crease the pitch
        and vice versa.
        :type delta_pitch: float
        :param abs_extrapolation: The maximum extrapolation out of the given dataset allowed before a near zero thrust is returned  in degrees
        :type abs_extrapolation: float
        :return: A dictionary with the following keys:
            "thrust": thrust created by the propeller,
            "torque": torque required for the propeller ,
            "eff": propulsive efficiency of the propeller,
            "thrust_coeff": thrust coefficient of the propeller,
            "power_coeff": power coefficien of the propeller,
            "AoA": Angle of attack at each station of the propeller,
            "lift_coeff": Lift coefficient at each station of the propeller,
            "drag_coeff": Drag coefficient at each station of the propller,
        :rtype: dict
        """        
        betas = self.betas + delta_pitch
        # Initial estimate for phi and zeta
        phi = np.arctan(self.lamb / self.Xi(self.r_stations))

        alphas = betas - phi

        # Get initial estimate of CL and Cd per station
        Cls = np.ones(len(self.r_stations))
        Cds = np.ones(len(self.r_stations))
        Reyn =  self.Omega*self.rho*self.chords/self.dyn_vis # Initial estiamte of the reynolds number


        for station in range(len(Reyn)):

            # Maximum and minimum RN in database
            if Reyn[station]<self.cd_interp.tree.mins[1] and Reyn[station] > 5e4:
                warn(f"A Reynolds number of {Reyn[station]:.3e} was encountered, lower than the minimum {self.cd_interp.tree.mins[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closest neighbour.", category=RuntimeWarning)
            if Reyn[station]>self.cd_interp.tree.maxes[1]:
                warn(f"A Reynolds number of {Reyn[station]:.3e} was encountered, higher than the maximum {self.cd_interp.tree.maxes[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closes neighbour.", category=RuntimeWarning)

         
            # Correct the Cl/Cd obtained for Mach number
            Cl_uncorr = self.cl_interp([[np.degrees(alphas[station]), Reyn[station]]])[0]
            Cl_ret =  Cl_uncorr / self.PG_correct(self.M(self.Omega*self.r_stations[station]))
            Cd_ret = self.cd_interp([[Cl_ret, Reyn[station]]])[0] / self.PG_correct(self.M(self.Omega*self.r_stations[station]))  # Retrieved Cd

            # Update the Cl and Cd at each station
            Cls[station] = Cl_ret
            Cds[station] = Cd_ret

        # Calculate initial estimates for the interference factors
        a_facs = self.a_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)
        a_prims = self.a_prim_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)

        # Iterate to get a convergent analysis
        count = 0
        iterate = True
        while iterate or (count < 10):
            # Calculate AoA of the blade stations
            alphas = betas - phi

            # Calculate the speed
            Ws = self.W(a_facs, a_prims, self.r_stations)

            # Calculate the Reynolds number
            Reyn = self.RN(Ws, self.chords)
            for station in range(len(self.r_stations)):

                # Maximum and minimum RN in database
                if Reyn[station]<self.cd_interp.tree.mins[1] and Reyn[station] > 5e4:
                    warn(f"A Reynolds number of {Reyn[station]:.3e} was encountered, lower than the minimum {self.cd_interp.tree.mins[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closest neighbour.", category=RuntimeWarning)
                if Reyn[station]>self.cd_interp.tree.maxes[1]:
                    warn(f"A Reynolds number of {Reyn[station]:.3e} was encountered, higher than the maximum {self.cd_interp.tree.maxes[1]:.3e} in the data set was reached. scipy.interpolate.NearestNDInterpolator will default to closes neighbour.", category=RuntimeWarning)

                if np.degrees(alphas[station]) > self.cl_interp.tree.maxes[0] + abs_extrapolation:
                    raise ValueError(f"An AoA of {np.degrees(alphas[station])} was encountered, higher than the maximum of {self.cl_interp.tree.maxes[0]} in the dataset. This results in extremely unreliable results. Try to lower the pitch angle")
            
                # Correct the Cl/Cd obtained for Mach number
                Cl_uncorr = self.cl_interp([[np.degrees(alphas[station]), Reyn[station]]])[0]
                Cl_ret =  Cl_uncorr / self.PG_correct(self.M(Ws[station]))
                Cd_ret = self.cd_interp([[Cl_uncorr, Reyn[station]]])[0] / self.PG_correct(self.M(Ws[station]))  # Retrieved Cd


                Cls[station] = Cl_ret
                Cds[station] = Cd_ret

            # Update the interference factors
            # TODO: Figure out why phi_t is defined like this. In the paper this done differently.
            a_facs = self.a_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)
            a_prims = self.a_prim_fac(Cls, Cds, phi, self.chords, self.r_stations, phi[-1]*self.r_stations[-1]/self.R)

            # Update phi
            phi_new = self.phi(a_facs, a_prims, self.r_stations)

            # Check convergence of the phi angles
            conv = np.abs((phi - phi_new) / phi)
          
            if np.average(conv) > 0.03:
                pass
            else:
                iterate = False

            # Update the phi angles
            phi = phi_new

            if count > max_iter:
                raise RuntimeError("Convergence failed maximum amount of iterations was reached")

            count += 1

        # Force coefficients
        Cx = self.Cx(Cls, Cds, phi)
        Cy = self.Cy(Cls, Cds, phi)

        # Thrust and torque per unit radius
        T_prim = 0.5 * self.rho * Ws**2 * self.B * self.chords * Cy
        Q_prim_r = 0.5 * self.rho * Ws**2 * self.B * self.chords * Cx

        # Do simple integration to get total thrust and Q per unit r
        T = spint.trapezoid(T_prim, self.r_stations)
        Q = spint.trapezoid(Q_prim_r * self.r_stations, self.r_stations)


        C_T_prim = T_prim/(self.rho * self.n**2 * self.D**4)
        C_T = spint.trapezoid(C_T_prim, self.r_stations)

        C_P_prim = C_T_prim * np.pi * self.r_stations/self.R * Cx/Cy
        C_P = spint.trapezoid(C_P_prim, self.r_stations)

        eff = self.eff(C_T, C_P)

        data_dict = {
            "thrust": T,
            "torque": Q,
            "eff": eff,
            "thrust_coeff":C_T,
            "power_coeff":C_P,
            "AoA":alphas,
            "lift_coeff":Cls,
            "drag_coeff":Cds,
        }

        return data_dict

