from math import pi
import pdb
import numpy as np
from warnings import warn
from scipy.integrate import trapz
from scipy import integrate
from scipy.optimize import minimize
from pymoo.problems.functional import FunctionalProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as minimizeGA
from tuduam.aerodynamics import lift_distribution
from tuduam.performance import ISA


class WingboxGeometry():
    """ This class provided a framework for  computing the geometry of the wing and wingbox which are given in 
    the following coordinate system which is attached to the nose.

    .. image:: vehicle_reference_frame.png
        :width: 500 
    """        
    def __init__(self, aero, airfoil, engine, flight_perf ,material, wing):
        """ The initialization of this class takes in several data structures as described below
        the data structures can be accessed throughout the class. 


        :param aero: _description_
        :type aero: _type_
        :param airfoil: _description_
        :type airfoil: _type_
        :param engine: _description_
        :type engine: _type_
        :param flight_perf: _description_
        :type flight_perf: _type_
        :param material: _description_
        :type material: _type_
        :param wing: _description_
        :type wing: _type_
        """        

        self.aerodynamics = aero
        self.airfoil = airfoil
        self.engine = engine
        self.flight_perf = flight_perf
        self.material = material
        self.wing = wing

        #Material
        self.rib_pitch = (self.wing.span/2)/(self.wing.n_ribs+1)
        self.t_rib = 3e-3

        # Aerodynamics
        self.lift_func = lift_distribution(aero, wing, flight_perf.cL_cruise/aero.cL_alpha + aero.alpha_zero_lift,ISA(flight_perf.h_cruise).density(),  flight_perf.v_cruise)

        #Engine
        self.engine_weight = engine.mass
        self.y_rotor_loc = engine.y_rotor_locations
        self.x_rotor_loc = engine.x_rotor_locations
        self.thrust_per_engine = engine.thrust
        #Torsion shaft

        #STRINGERS
        self.n_str = 15
        self.str_array_root = np.linspace(0.15*self.wing.chord_root, 0.75*self.wing.chord_root,  self.n_str+2)


        #GEOMETRY
        self.width_wingbox = (wing.wingbox_end - wing.wingbox_start)*wing.chord_root
        self.pitch_str = self.width_wingbox/(self.n_str+1) #THE PROGRAM ASSUMES THERE ARE TWO STRINGERS AT EACH END AS WELL

        #OPT related
        self.y = np.linspace(0, self.wing.span/2, 18)


    #---------------Geometry functions-----------------

    def x_to_global(self, coordinate):
        return coordinate + self.wing.x_le_root_chord

    def perimiter_ellipse(self,a,b):
        """
        Ramanujans first approximation formula

        :param a: minor  length of the ellipse
        :type a: float
        :param b: Major length of the ellipse
        :type b: float
        :return: The perimeter of the ellipse
        :rtype: float
        """        

        return np.pi*(3*(a+b) - np.sqrt( (3*a + b)*(a + 3*b))) 


    def chord(self,y):
        return self.wing.chord_root - self.wing.chord_root * (1 - self.wing.taper) * y * 2 / self.wing.span

    def height(self,y):
        return self.airfoil.thickness_to_chord * self.chord(y)
    
    def l_sk(self,y):
        return np.sqrt(self.height(y) ** 2 + (0.25 * self.chord(y)) ** 2)


    def get_area_str(self, h_st,w_st,t_st):
        return t_st * (2 * w_st + h_st)

    def I_st_x(self, h_st,w_st,t_st):
        Ast = self.get_area_str(h_st,w_st,t_st)
        i = t_st * h_st ** 3 / 12 + w_st * t_st ** 3 / 12 + 2 * Ast * (0.5 * h_st) ** 2
        return i

    def I_st_z(self, h_st,w_st,t_st):
        Ast = self.get_area_str(h_st,w_st,t_st)
        i = (h_st*t_st ** 3)/12 + (t_st* w_st**3)/12
        return i

    def w_sp(self,y):
        return 0.3 * self.height(y)


    def I_sp_x(self,t_sp,y):
        h = self.height(y)
        wsp = self.w_sp(y)
        return t_sp * (h - 2 * t_sp) ** 3 / 12 + 2 * wsp * t_sp ** 3 / 12 + 2 * t_sp * wsp * (
                0.5 * h) ** 2

    def I_sp_z(self,t_sp,y):
        h = self.height(y)
        wsp = self.w_sp(y)
        return ((h - 2*t_sp)*t_sp**3)/12 + (2*t_sp*wsp**3)/12

    def get_x_le(self,y):
        return self.wing.x_le_root_global + np.tan(self.wing.sweep_le)*y

    def get_x_te(self,y):
        warn("change this one")
        return self.wing.x_lemac_local + 0.25*self.wing.chord_root +0.75*self.chord(y)

    def get_x_start_wb(self,y):
        return self.get_x_le(y) + self.wing.wingbox_start*self.chord(y)

    def get_x_end_wb(self,y):
        return self.get_x_le(y) + self.wing.wingbox_end*self.chord(y)

    def get_y_start_wb(self,x):
        return (x-self.wing.x_lemac-0.15*self.wing.chord_root)/(0.1*self.wing.chord_root*(1-self.wing.taper)*2/self.wing.span)

    def get_y_end_wb(self,x):
        return (x-self.wing.x_lemac-0.75*self.wing.chord_root)/(-0.5*self.wing.chord_root*(1-self.wing.taper)*2/self.wing.span)

    
    def get_r_o(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return self.height(self.y_rotor_loc[0])/2 - t_sk*3
  

    def I_xx(self, x):#TODO implement dissappearing stringers
        """_summary_

        :param x: _description_
        :type x: _type_
        :return: _description_
        :rtype: _type_
        """        
        t_sp, h_st, w_st, t_st, t_sk = x
        h = self.height(self.y)
        # nst = n_st(c_r, b_st)
        Ist = self.I_st_x(h_st,w_st,t_st)
        Isp = self.I_sp_x(t_sp, self.y)
        A = self.get_area_str(h_st,w_st,t_st)
        return 2 * (Ist + A * (0.5 * h) ** 2) * self.n_str + 2 * Isp + 2 * (0.6 * self.chord(self.y) * t_sk ** 3 / 12 + t_sk * 0.6 * self.wing.chord_root * (0.5 * h) ** 2)

    def I_zz(self, x):#TODO implement dissappearing stringers
        t_sp, h_st, w_st, t_st, t_sk = x
        y = self.y
        h = self.height(y)
        Ist = self.I_st_z(h_st,w_st,t_st)
        Isp = self.I_sp_z(t_sp,y)
        Ast = self.get_area_str(h_st,w_st,t_st)
        Asp = t_sp*self.w_sp(y)*2 + (h-2*t_sp)*t_sp
        centre_line = self.wing.chord_root*0.25 + self.chord(y)*0.25
        position_stringers = np.ones((len(y),len(self.str_array_root)))*self.str_array_root
        distances_from_centre_line = position_stringers - np.transpose(np.ones((len(self.str_array_root),len(y))) * centre_line)
        moments_of_inertia = np.sum(distances_from_centre_line * Ast * 2,axis=1)
        moments_of_inertia += 2*(Isp + Asp * (self.chord(y)/2)*(self.chord(y)/2)) + 2* t_sk*self.chord(y)**3/12
        return moments_of_inertia

    #------------------ Mass computation --------------------------------------------
    def weight_from_tip(self, x):#TODO implement dissappearing stringers #TODO Include rib weights
        t_sp, h_st, w_st, t_st, t_sk = x
        y = self.y

        weight_str = self.material.density * self.get_area_str(h_st,w_st,t_st) * (self.wing.span/2- y) * self.n_str * 2
        weight_skin = (t_sk * ((0.6 * self.chord(self.wing.span/2) + 0.6 * self.chord(y))* (self.wing.span/2- y) / 2) * 2 + self.perimiter_ellipse(0.15*self.chord(y),self.height(y))*t_sk * 0.15 + np.sqrt((0.25*self.chord(y))**2 + (self.height(y))**2)*2*t_sk)*self.material.density
        weight_spar_flanges = (self.w_sp(self.wing.span/2) + self.w_sp(y))*(self.wing.span/2- y)/2 * t_sp * self.material.density * 4
        weight_spar_web = (self.height(self.wing.span/2) - 2*t_sp + self.height(y) - 2*t_sp) * (self.wing.span/2- y) /2 * t_sp *self.material.density * 2

        total_weight = (weight_str + weight_skin + weight_spar_flanges + weight_spar_web)

        difference_array = np.absolute(y-self.y_rotor_loc[0])
        index = difference_array.argmin()

        total_weight[0:index+1] += self.engine_weight
        warn("Total weight currently also takes the engine weight, should be discussed")

        total_weight += self.engine_weight
        weight_ribs = np.linspace(self.wing.n_ribs,1,len(y)) * self.chord(y) * self.height(y) * self.t_rib * self.material.density
        total_weight += weight_ribs

        return total_weight


    def total_weight(self, x):
        return self.weight_from_tip(x)[0]


class WingboxInternalForces(WingboxGeometry):
    """ This class provided a framework for computing the internal forces in the wingbox of which are given in :download:`Optimization documentation <eVTOL_Structural_Analysis.pdf>`.
    The coordinate system used for this analysis is as follows.

    .. image:: coordinate_system.png
        :width: 500 
        :alt:  Coordinate system used in computations

    Additionally, the geometry of the wingbox is simplified as follows.

    .. image:: wingbox_geometry.png
        :width: 500 
        :alt:  Coordinate system used in computations

    :param WingboxGeometry: _description_
    :type WingboxGeometry: _type_
    :return: _description_
    :rtype: _type_
    """        
    pass

    def moment_y_from_tip(self,y):
        """ Returns the moment at each section in the form of an array

        :param y: spanwise position
        :type y: float
        :return: moment in  Newton-meters
        :rtype: float
        """        
        torque_arr = np.zeros(len(y))

        for x_loc, y_loc in zip(self.x_rotor_loc, self.y_rotor_loc):
            index = np.argmin(np.absolute(y - y_loc))
            torque_arr[0:index + 1] =  (self.engine.mass*9.81)*((self.get_x_start_wb(0) + self.get_x_end_wb(0))/2 - x_loc) #Torque at from tip roto

        return -torque_arr


    def shear_z_from_tip(self, x):
        y = self.y
        return -self.weight_from_tip(x)*9.81 + self.lift_func(y)*self.flight_perf.n_ult

 
    def moment_x_from_tip(self, x):
        shear = self.shear_z_from_tip(x)
        moment = np.zeros(len(self.y))
        dy = (self.y[1]-self.y[0])
        for i in range(2,len(self.y)+1):
            moment[-i] = shear[-i+1]*dy + moment[-i+1]
        return moment
        # return self.shear_z_from_tip(x)*(self.span/2 - self.y)

#-----------Stress functions---------
    def bending_stress_x_from_tip(self, x):
        return self.moment_x_from_tip(x)/self.I_xx(x) * self.height(self.y)/2

    def shearflow_max_from_tip(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        y = self.y
        Vz = self.shear_z_from_tip(x)
        T = self.moment_y_from_tip(y)
        Ixx = self.I_xx(x)
        height = self.height(y)
        chord = self.chord(y)
        Nxy = np.zeros(len(y))
        max_shear_stress = np.zeros(len(y))
        l_sk = self.l_sk(y)

        for i in range(len(y)):
            # Base region 1
            def qb1(z):
                return Vz[i] * t_sk * (0.5 * height[i]) ** 2 * (np.cos(z) - 1) / Ixx[i]
            I1 = qb1(pi / 2)

            # Base region 2
            def qb2(z):
                return -Vz[i] * t_sp * z ** 2 / (2 * Ixx[i])
            I2 = qb2(height[i])
            s2 = np.arange(0, height[i]+ 0.1, 0.1)

            # Base region 3
            def qb3(z):
                return - Vz[i] * t_sk * (0.5 * height[i]) * z / Ixx[i] + I1 + I2
            I3 = qb3(0.6 * chord[i])
            s3 = np.arange(0, 0.6*chord[i]+ 0.1, 0.1)

            # Base region 4
            def qb4(z):
                return -Vz[i] * t_sp * z ** 2 / (2 * Ixx[i])
            I4 = qb4(height[i])
            s4=np.arange(0, height[i]+ 0.1, 0.1)

            # Base region 5
            def qb5(z):
                return -Vz[i] * t_sk / Ixx[i] * (0.5 * height[i] * z - 0.5 * 0.5 * height[i] * z ** 2 / l_sk[i]) + I3 + I4
            I5 = qb5(l_sk[i])

            # Base region 6
            def qb6(z):
                return Vz[i] * t_sk / Ixx[i] * 0.5 * 0.5 * height[i] / l_sk[i] * z ** 2 + I5
            I6 = qb6(l_sk[i])

            # Base region 7
            def qb7(z):
                return -Vz[i] * t_sp * 0.5 * z ** 2 / Ixx[i]
            I7 = qb7(-height[i])


            # Base region 8
            def qb8(z):
                return -Vz[i] * 0.5 * height[i] * t_sp * z / Ixx[i] + I6 - I7
            I8 = qb8(0.6 * chord[i])

            # Base region 9
            def qb9(z):
                return -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx[i]
            I9 = qb9(-height[i])

            # Base region 10
            def qb10(z):
                return -Vz[i] * t_sk * (0.5 * height[i]) ** 2 * (np.cos(z) - 1) / Ixx[i] + I8 - I9

            #Torsion
            A1 = float(np.pi*height[i]*chord[i]*0.15*0.5)
            A2 = float(height[i]*0.6*chord[i])
            A3 = float(height[i]*0.25*chord[i])

            T_A11 = 0.5 * A1 * self.perimiter_ellipse(height[i],0.15*chord[i]) * 0.5 * t_sk
            T_A12 = -A1 * height[i] * t_sp
            T_A13 = 0
            T_A14 = -1/(0.5*self.material.shear_modulus)

            T_A21 = -A2 * height[i] * t_sp
            T_A22 = A2 * height[i] * t_sp * 2 + chord[i]*0.6*2*A2*t_sk
            T_A23 = -height[i]*A2*t_sp
            T_A24 = -1/(0.5*self.material.shear_modulus)

            T_A31 = 0
            T_A32 = -A3 * height[i] *t_sp
            T_A33 = A3 * height[i] * t_sp + l_sk[i]*A3*t_sk*2
            T_A34 = -1/(0.5*self.material.shear_modulus)

            T_A41 = 2*A1
            T_A42 = 2*A2
            T_A43 = 2*A3
            T_A44 = 0

            T_A = np.array([[T_A11, T_A12, T_A13, T_A14], [T_A21, T_A22, T_A23, T_A24], [T_A31, T_A32, T_A33, T_A34],[T_A41,T_A42,T_A43,T_A44]])
            T_B = np.array([0,0,0,T[i]])
            T_X = np.linalg.solve(T_A, T_B)



            # Redundant shear flow
            A11 = pi * (0.5 * height[i]) / t_sk + height[i] / t_sp
            A12 = -height[i] / t_sp
            A21 = - height[i] / t_sp
            A22 = 1.2 * chord[i] / t_sk
            A23 = -height[i] / t_sp
            A32 = - height[i] / t_sp
            A33 = 2 * l_sk[i] / t_sk + height[i] / t_sp



            B1 = 0.5 * height[i] / t_sk * trapz([qb1(0),qb1(pi/2)], [0, pi / 2]) + trapz([qb2(0),qb2(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp - trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0])/ t_sp + trapz([qb10(-pi/2),qb10(0)], [-pi / 2, 0]) * 0.5 * height[i] / t_sk
            B2 = trapz([qb2(0),qb2(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp + trapz([qb3(0),qb3(0.6*chord[i])], [0, 0.6 * chord[i]]) / t_sk - trapz([qb7(-0.5*height[i]),qb7(0)], [-0.5 * height[i], 0]) / t_sp + \
                    trapz([qb4(0),qb4(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp + trapz([qb8(0),qb8(0.6*chord[i])], [0, 0.6 * chord[i]]) / t_sk - trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0]) / t_sp
            B3 = trapz([qb5(0),qb5(l_sk[i])], [0, l_sk[i]]) / t_sk + trapz([qb6(0),qb6(l_sk[i])], [0, l_sk[i]]) / t_sk + trapz([qb4(0),qb4(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp - \
                    trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0]) / t_sp

            A = np.array([[A11, A12, 0], [A21, A22, A23], [0, A32, A33]])
            B = -np.array([[B1], [B2], [B3]])
            X = np.linalg.solve(A, B)

            q01 = float(X[0])
            q02 = float(X[1])
            q03 = float(X[2])

            qT1 = float(T_X[0])
            qT2 = float(T_X[1])
            qT3 = float(T_X[1])

            # Compute final shear flow
            q2 = qb2(s2) - q01 - qT1 + q02 + qT2
            q3 = qb3(s3) + q02 + qT2
            q4 = qb4(s4) + q03 +qT3 - q02 - qT2

            max_region2 = max(q2)
            max_region3 = max(q3)
            max_region4 = max(q4)

            determine = max(max_region2, max_region3, max_region4)
            Nxy[i] = determine
            max_shear_stress[i] = max(max_region2/t_sp, max_region3/t_sk, max_region4/t_sp)
        return Nxy

    def distrcompr_max_from_tip(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return self.bending_stress_x_from_tip(x) * t_sk
    
    # def shear_flow_torque(self,x):
    #     T = self.torque_from_tip(X)
    #     y = self.y
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return T/(2*self.chord(y)*self.height(y))


#-------Georgiana Constraints-------
    



class Constraints(WingboxInternalForces):
    """ The following class offers the constraints 
    """    

    def local_buckling(self,t_sk):#TODO
        buck = 4* pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson ** 2)) * (t_sk / self.pitch_str) ** 2
        return buck


    def flange_buckling(self,t_st, w_st):#TODO
        buck = 2 * pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson ** 2)) * (t_st / w_st) ** 2
        return buck


    def web_buckling(self,t_st, h_st):#TODO
        buck = 4 * pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson ** 2)) * (t_st / h_st) ** 2
        return buck


    def global_buckling(self, h_st, t_st, t):#TODO
        # n = n_st(c_r, b_st)
        tsmr = (t * self.pitch_str + t_st * self.n_str * (h_st - t)) / self.pitch_str
        return 4 * pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson ** 2)) * (tsmr / self.pitch_str) ** 2


    def shear_buckling(self,t_sk):#TODO
        buck = 5.35 * pi ** 2 * self.material.young_modulus / (12 * (1 - self.material.poisson)) * (t_sk / self.pitch_str) ** 2
        return buck



    def buckling(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        Nxy = self.shearflow_max_from_tip(x)
        Nx = self.distrcompr_max_from_tip(x)
        # print("Nx",Nx)
        # print("Nxy",Nxy)
        Nx_crit = self.local_buckling(t_sk)*t_sk
        Nxy_crit = self.shear_buckling(t_sk)*t_sk
        buck = Nx*self.material.safety_factor / Nx_crit + (Nxy*self.material.safety_factor / Nxy_crit) ** 2
        return buck


    def column_st(self, h_st, w_st, t_st, t_sk):#
        #Lnew=new_L(b,L)
        Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
        i= pi ** 2 * self.material.young_modulus * Ist / (2*w_st* self.rib_pitch ** 2)#TODO IF HE FAILS REPLACE SPAN WITH RIB PITCH
        return i


    def f_ult(self, h_st,w_st,t_st,t_sk,y):#TODO
        A_st = self.get_area_str(h_st,w_st,t_st)
        # n=n_st(c_r,b_st)
        A=self.n_str*A_st+0.6*self.chord(y)*t_sk
        f_uts=self.sigma_uts*A
        return f_uts


    def buckling_constr(self, x):
        buck = self.buckling(x)
        return -1*(buck - 1)


    def global_local(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        glob = self.global_buckling(h_st, t_st, t_sk)
        loc = self.local_buckling(t_sk)
        diff = glob - loc
        return diff


    def local_column(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        col = self.column_st(h_st,w_st,t_st, t_sk)
        loc = self.local_buckling(t_sk)
        # print("col=",col/1e6)
        # print("loc=",loc/1e6)
        diff = col - loc
        return diff


    def flange_loc_loc(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        flange = self.flange_buckling(t_st,w_st)
        loc = self.local_buckling(t_sk)
        diff = flange - loc
        return diff


    def web_flange(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        web = self.web_buckling(t_st, h_st)
        loc = self.local_buckling(t_sk)
        diff =web-loc
        return diff


    def von_Mises(self, x):
        y = self.y
        t_sp, h_st, w_st, t_st, t_sk = x
        Nxy =self.shearflow_max_from_tip(x)
        bend_stress=self.bending_stress_x_from_tip(x)
        tau_shear_arr = Nxy/t_sk
        vm_lst =  np.sqrt(0.5 * (3 * tau_shear_arr ** 2+bend_stress**2))*self.material.safety_factor/self.material.sigma_yield
        return vm_lst


    def crippling(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        A = self.get_area_str(h_st,w_st,t_st)
        col = self.column_st( h_st,w_st,t_st,t_sk)
        crip = t_st * self.material.beta_crippling * self.material.sigma_yield* ((self.material.g_crippling * t_st ** 2 / A) * np.sqrt(self.material.young_modulus / self.material.sigma_yield)) ** self.m_crip
        return crip

    #----OWN CONSTRAINTS-----
    def str_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
        i= pi ** 2 * self.material.young_modulus * Ist / (self.rib_pitch ** 2)#TODO IF HE FAILS REPLACE SPAN WITH RIB PITCH
        i_sigma = (i/self.get_area_str(h_st,w_st,t_st))#convert to stress
        return -1*(self.material.safety_factor*self.bending_stress_x_from_tip(x)/(i_sigma) - 1)
    
    def f_ult_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_x_from_tip(x)/self.material.sigma_ultimate - 1)
    def flange_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_x_from_tip(x)/self.flange_buckling(t_st,w_st) - 1)
    
    def web_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_x_from_tip(x)/self.web_buckling(t_st,h_st) - 1)
    
    def global_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_x_from_tip(x)/self.global_buckling(h_st,t_st,t_sk) - 1)



def wingbox_optimization(aero, airfoil, engine, flight_perf, material, wing):
    """This functions optimizes the wingbox structures given various assumptions the details
    of which are given in :download:`Optimization documentation <eVTOL_Structural_Analysis.pdf>`.
    The coordinate system used for this analysis is as follows.

    .. image:: coordinate_system.png
        :width: 500 
        :alt:  Coordinate system used in computations

    :param aero: _description_
    :type aero: _type_
    :param engine: _description_
    :type engine: _type_
    :param material: _description_
    :type material: _type_
    :param flight_perf: _description_
    :type flight_perf: _type_
    :param wing: _description_
    :type wing: _type_
    :return: _description_
    :rtype: _type_
    """    
    WingGeom = WingboxGeometry(aero, airfoil, engine, flight_perf, material, wing)
    Constr =  Constraints(aero, airfoil, engine, flight_perf, material, wing)
    # NOTE Engine positions in the json are updated in the dump function so first it's dumped and then it's loaded again.
    # ------SET INITIAL VALUES------
    tsp= wing.spar_thickness
    hst= wing.stringer_height
    wst= wing.stringer_width
    tst= wing.stringer_thickness
    tsk= wing.skin_thickness


    X = [tsp, hst, wst, tst, tsk]
    y = WingGeom.y

    #------SET BOUNDS--------
    height_tip = WingGeom.height(WingGeom.wing.span/2) - 2e-2#NOTE Set upper value so the stringer is not bigger than the wing itself.
    xlower =  5e-3,1.5e-2, 1.5e-2, 2e-3, 8e-4
    xupper = height_tip/2, height_tip/2, 1e-1, 3.3e-2, 1e-1

    bounds = np.vstack((xlower, xupper)).T


    #NOTE GA optimizer to explore the design space
    objs = [WingGeom.total_weight]

    constr_ieq = [
        lambda x: -Constr.buckling_constr(x)[0],
        lambda x: -Constr.von_Mises(x)[0],
        lambda x: -Constr.str_buckling_constr(x)[0],
        lambda x: -Constr.f_ult_constr(x)[0],
        lambda x: -Constr.flange_buckling_constr(x)[0],
        lambda x: -Constr.web_buckling_constr(x)[0],
        lambda x: -Constr.global_buckling_constr(x)[0],
    ]

    problem = FunctionalProblem(len(X),
                                objs,
                                constr_ieq=constr_ieq,
                                xl=xlower,
                                xu=xupper,
                                )

    method = GA(pop_size=50, eliminate_duplicates=True)

    resGA = minimizeGA(problem, method, termination=('n_gen', 50   ), seed=1,
                    save_history=True, verbose=True)
    print('GA optimum variables', resGA.X)
    print('GA optimum weight', resGA.F)


        # NOTE final gradient descent to converget to a minimum point with SciPy minimize

    print()
    print('Final SciPy minimize optimization')
    options = dict(eps=1e-5, ftol=1e-3)
    constraints = [
        {'type': 'ineq', 'fun': lambda x: Constr.buckling_constr(x)[0]},
        {'type': 'ineq', 'fun': lambda x: Constr.von_Mises(x)[0]},
        {'type': 'ineq', 'fun': lambda x: Constr.str_buckling_constr(x)[0]},
        {'type': 'ineq', 'fun': lambda x: Constr.f_ult_constr(x)[0]},
        {'type': 'ineq', 'fun': lambda x: Constr.flange_buckling_constr(x)[0]},
        {'type': 'ineq', 'fun': lambda x: Constr.web_buckling_constr(x)[0]},
        {'type': 'ineq', 'fun': lambda x: Constr.global_buckling_constr(x)[0]},
    ]

    resMin = minimize(WingGeom.total_weight, resGA.X, method='SLSQP',
                constraints=constraints, bounds=bounds, jac='3-point',
                options=options)

    wing.spar_thickness, wing.stringer_height, wing.stringer_width, wing.stringer_thickness, wing.skin_thickness = resMin.x
    
    x_final = resMin.x
            
    return resMin.x


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

        

