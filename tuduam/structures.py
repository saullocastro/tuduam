from math import pi
import pdb
import numpy as np
from warnings import warn
from scipy.integrate import trapezoid, cumulative_trapezoid, trapz, dblquad
from scipy.constants import g
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

    Currently this class only supports z stringers as shown in the following figure.

    .. image:: z_stringer.png
        :width: 500 
        :alt:  z stringer
    
    The geometry of the wingbox is simplified as shown hereunder. There are several thickness and lengths that can be defined.
    The skin thickness defines the thickness of the leading edge and trailing edge skin. Then the flange and spar thickness
    can be specified individually. Further more the height, width and thickness of the stringer can also be specified.

    .. image:: wingbox_geometry.png
        :width: 500 
        :alt:  Coordinate system used in computations

    Further more the rib placement is evenly spaced throughout the wing


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

        # Aerodynamics
        self.lift_func = lift_distribution(aero, wing, flight_perf.cL_cruise/aero.cL_alpha + aero.alpha_zero_lift,ISA(flight_perf.h_cruise).density(),  flight_perf.v_cruise)

        #STRINGERS
        self.str_array = np.linspace(self.wing.wingbox_start, self.wing.wingbox_end,  self.wing.n_str)

        #GEOMETRY
        self.width_wingbox_root = (wing.wingbox_end - wing.wingbox_start)*wing.chord_root
        self.width_wingbox = (wing.wingbox_end - wing.wingbox_start)
        self.pitch_str = self.width_wingbox_root/(self.wing.n_str+1) #THE PROGRAM ASSUMES THERE ARE TWO STRINGERS AT EACH END AS WELL

        #OPT related
        if self.wing.rib_loc == None:
            self.rib_loc = np.linspace(0, 1, self.wing.n_ribs - self.engine.n_engines)**2*self.wing.span/2

            # Make sure ribs are placed at the mounting of the engines
            for eng_loc in self.engine.y_rotor_loc:
                if eng_loc < 0:
                    continue
                self.rib_loc = np.insert(self.rib_loc, 0, eng_loc - self.engine.diameter/2 )
                self.rib_loc = np.insert(self.rib_loc, 0, eng_loc + self.engine.diameter/2 )
            self.rib_loc.sort()
            self.wing.rib_loc = self.rib_loc
        else: 
            pass



    #---------------Geometry functions-----------------
    def _x_to_global(self, coordinate):
        """ Transform an x coordinate from the local frame attached to the leading edge of the root to the global 
        coordinate system

        :param coordinate: x coordinate to be transfomred
        :type coordinate: float
        :return: transformed x coordinate
        :rtype:  float
        """        
        return coordinate + self.wing.x_le_root_global

    def perimiter_ellipse(self,a,b):
        """
        Ramanujans first approximation formula

        :param a:  Semi-Major axis of the ellipse
        :type a: float
        :param b: Semi-Minor axis of the ellipse
        :type b: float
        :return: The perimeter of the ellipse
        :rtype: float
        """        

        return np.pi*(3*(a+b) - np.sqrt( (3*a + b)*(a + 3*b))) 

    def rad_ellipse_polar(self,theta,a,b):
        """Returns the distance the distance r as shown in the figure below where theta is defined against
        the major axis. Function is used  for the leading edge of wingbox.

        .. image:: ..\_static\Ellipse_Polar_center.png
            :width: 500 
            :alt:  Coordinate system used in computations

        :param theta: The angular coordinate in radians see figure
        :type theta: float 
        :param a: Semi-major axis (distance BC)
        :type a: float
        :param b: Semi-minor axis 
        :type b: float
        """        

        if b>a:
            raise Exception("Minor axis is defined larger than the major axis please  \
                            switch the axis in the input")

        e = np.sqrt(1 - (b/a)**2)
        return a*b/np.sqrt((b*np.cos(theta))**2 + (a*np.sin(theta))**2)


    def chord(self,y):
        """ Computes the chord at any given spanwise position

        :param y: Spanwise position
        :type y: float
        :return: local chord
        :rtype: float
        """        
        return self.wing.chord_root - self.wing.chord_root * (1 - self.wing.taper) * y * 2 / self.wing.span

    def height(self,y):
        """ Computes the height of the wingbox at any givn spanwise position

        :param y: spanwise position
        :type y: float
        :return: height of wingbox
        :rtype: float
        """        
        return self.airfoil.thickness_to_chord * self.chord(y)
    
    def l_sk_te(self,y):
        """ Computes the length of side 5 and 6 (i.e one edge not both simultaneoulsy) shown in the wingbox geometry. Used in 
        computing the shear flows through them

        :param y: Spanwise position
        :type y: float
        :return: Length of the trailing edge (i.e one edge not both simultaneoulsy) (region 5 and 6) shown
          in the simplified wingbox geometry
        :rtype: float
        """        
        return np.sqrt((self.height(y)/2)**2 + ((1 - self.wing.wingbox_end)*self.chord(y))**2)


    def get_area_str(self, h_st,w_st,t_st):
        """ Return the cross sectional area of a single z stringer

        :param h_st: Height of the web plate in m
        :type h_st: float
        :param w_st: width of the two flanges in m
        :type w_st: float
        :param t_st: thickness of the stringer considered constant in m 
        :type t_st: float
        :return: cross sectional area
        :rtype: float
        """        
        return t_st * (2 * w_st + h_st)

    def I_st_x(self, h_st,w_st,t_st):
        """ Return the second moment of area of a single z stringer
        arond the x axis (see reference frame)

        :param h_st: Height of the web plate in m
        :type h_st: float
        :param w_st: width of the two flanges in m
        :type w_st: float
        :param t_st: thickness of the stringer considered constant in m 
        :type t_st: float
        :return: cross sectional area
        :rtype: float
        """        
        return t_st*(h_st**3)/12 + w_st*(t_st**3)/6 + 2*t_st*w_st*(0.5*h_st + t_st/2)**2

    def I_st_z(self, h_st,w_st,t_st):
        """ Return the second moment of area of a single z stringer
        arond the z axis (see reference frame)

        :param h_st: Height of the web plate in m
        :type h_st: float
        :param w_st: width of the two flanges in m
        :type w_st: float
        :param t_st: thickness of the stringer considered constant in m 
        :type t_st: float
        :return: cross sectional area
        :rtype: float
        """        
        return (h_st*t_st ** 3)/12 + (t_st* w_st**3)/6 + 2*w_st*t_st*(w_st/2 - t_st/2)**2


    def l_sp(self,y):
        """ Returns the length of section 2 in the cross section view of the shear flow.
        Simple division by two.

        :param y: spanwise location
        :type y: float
        :return: Length section 2 in the cross sectional view of the shear flow diagram
        :rtype: float
        """        
        return self.height(y)
    
    def l_fl(self,y):
        """ Computes the length of the top and bottom flange of the wingbox

        :param y: spanwise position
        :type y: float
        :return: Length of top and bottom flange
        :rtype: float
        """        
        return self.width_wingbox*self.chord(y)


    def I_sp_fl_x(self,t_sp,t_fl, y):
        """ Return the moment of inertia of the spars and flanges around the x axis

        :param t_sp: thickness of the spar
        :type t_sp: float
        :param y: Spanwise locations
        :type y: float
        :return: Moment of inertia of the spar
        :rtype: float
        """        
        h = self.height(y)
        w_fl = self.l_fl(y)

        return w_fl*h**3/12 - (w_fl - 2*t_sp)*(h - 2*t_fl)**3/12

    def I_sp_fl_z(self,t_sp,t_fl,y):
        """ Return the moment of inertia of the spars and flanges around the z axis

        :param t_sp: thickness of the spar
        :type t_sp: float
        :param y: Spanwise locations
        :type y: float
        :return: Moment of inertia of the spar
        :rtype: float
        """        
        h = self.height(y)
        w_fl = self.l_fl(y)
        return w_fl**3*h/12 - (w_fl - 2*t_sp)**3*(h - 2*t_fl)/12

    def get_x_le(self,y):
        """ Compute the x coordinate of the leading edge at a given spanwise position.

        :param y: spanwise position
        :type y: float
        :return: The x coordinate of the leading edge
        :rtype: float
        """        
        return self.wing.x_le_root_global + np.tan(self.wing.sweep_le)*y

    def get_x_start_wb(self,y):
        """Compute the start of the wingbox in the global reference
         frame  at a given spanwise location

        :param y: spanwise location
        :type y: float
        :return: x coordinate of the start of wingbox
        :rtype: float
        """        
        return self.get_x_le(y) + self.wing.wingbox_start*self.chord(y)

    def get_x_end_wb(self,y):
        """Compute the end of the wingbox in the global reference
         frame  at a given spanwise location

        :param y: spanwise location
        :type y: float
        :return: x coordinate of the start of wingbox
        :rtype: float
        """        
        return self.get_x_le(y) + self.wing.wingbox_end*self.chord(y)

    def I_xx(self, x):
        """ Computes the moment of inertia around the x axis  of the wingbox at a mutlitude of points and returns it as vectors.
        Note that it only takes into account the spar webs, flanges and stringer. The skin of the leading and trailing 
        edge are neglected since they do not carry much bending load

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the moment of inertia at various locations
        :rtype: list
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x
        h = self.height(self.rib_loc)
        Ist = self.I_st_x(h_st,w_st,t_st)
        I_box = self.I_sp_fl_x(t_sp, t_fl, self.rib_loc)
        A_str = self.get_area_str(h_st,w_st,t_st)

        return (Ist + A_str*(h/2 - h_st/2)**2)*self.wing.n_str*2 + I_box

    def I_zz(self, x):
        """ Computes the moment of inertia around the z axis of the wingbox at a mutlitude of points and returns it as vectors.
        Note that it only takes into account the spar webs, flanges and stringer. The skin of the leading and trailing 
        edge are neglected since they do not carry much bending load

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the moment of inertia at various locations
        :type x: numpy.ndarray
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x

        h = self.height(self.rib_loc)
        vec_chord = self.chord(self.rib_loc)
        Ist = self.I_st_z(h_st,w_st,t_st)
        I_box = self.I_sp_fl_z(t_sp, t_fl, self.rib_loc)
        Ast = self.get_area_str(h_st,w_st,t_st)
        centre_line = (self.wing.wingbox_start + self.wing.wingbox_end)/2*vec_chord

        pos_str =  self.str_array*vec_chord.reshape(-1,1) # Utilize NumPy broadcasting (1,y) X (x,1) = (x,y) w
        moment_arms = pos_str -  centre_line.reshape(-1,1) # Utilize broadcasting property again
        moi_str = np.sum(Ast*moment_arms**2, axis=1)*2

        return moi_str + I_box + Ist*self.wing.n_str*2

    
    def str_weight_from_tip(self, x):
        """ Computes the weight of the stringers in the planform. Assumes stringers are straight along the planform,
        in reality they follow the local sweep of the wing.

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the stringer weight at each section
        :rtype: numpy.ndarray
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x
        warn("stringer weight assumes they are straight while in reality they follow the local sweep of the wing")
        return self.material.density*self.get_area_str(h_st,w_st,t_st)*(np.flip(self.rib_loc))*self.wing.n_str*2*g

    def le_wingbox_weight_from_tip(self,x):
        """ Uses trapezium integration approximation to compute the leading edge weight of the skin. This introduces a slight error as the perimeter
        is a non linear equation

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the leading edge skin  of the wingbox
        :rtype: numpy.ndarray
        """        
        t_sp,t_fl, h_st, w_st, t_st, t_sk = x
        return (trapezoid(self.perimiter_ellipse(self.wing.wingbox_start*self.chord(self.rib_loc),self.height(self.rib_loc))*t_sk, self.rib_loc) \
                - np.insert(cumulative_trapezoid(self.perimiter_ellipse(self.wing.wingbox_start*self.chord(self.rib_loc),self.height(self.rib_loc))*t_sk, self.rib_loc),0,0))*self.material.density*g

    def te_wingbox_weight_from_tip(self,x):
        """ Computes the weight of the skin in the trailing edge of the wingbox. Weight is approximated using the trapezium integration method
        in reality this does not hold due to the non-linearity of the length of the trailing w.r.t to the span.

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the leading edge skin  of the wingbox
        :rtype: numpy.ndarray
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x
        return (trapezoid(self.l_sk_te(self.rib_loc)*t_sk*2, self.rib_loc) - np.insert(cumulative_trapezoid(self.l_sk_te(self.rib_loc)*t_sk*2, self.rib_loc),0,0))*self.material.density*g
    
    def fl_weight_from_tip(self, x):
        """ Computes the weight of the flanges in the planform as a vector

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the flange weight at each section to the tip
        :rtype: numpy.ndarray
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x
        return (trapezoid(self.l_fl(self.rib_loc)*t_fl*2, self.rib_loc) - np.insert(cumulative_trapezoid(self.l_fl(self.rib_loc)*t_fl*2, self.rib_loc),0,0))*self.material.density*g

    def spar_weight_from_tip(self,x):
        """ Computes the weight of the spars in the planform in the form of a vector showing the amount of weight to the tip

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: Vector containing the spar weight at each section
        :rtype: numpy.ndarray
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x
        return (trapezoid(self.height(self.rib_loc)*t_sp*2, self.rib_loc) - np.insert(cumulative_trapezoid(self.height(self.rib_loc)*t_sp*2, self.rib_loc),0,0))*self.material.density*g
    
    def rib_weight_from_tip(self):
        """ Computes a vector of the weight of the ribs to the tip

        :return: A vector of the weight of the ribs to the tip
        :rtype: list
        """        
        weight_rib =  self.chord(self.rib_loc) * self.height(self.rib_loc) * self.wing.t_rib * self.material.density
        return np.cumsum(np.ones(len(self.rib_loc))*weight_rib)[::-1]*g

    def engine_weight_from_tip(self):
        """ Computes a vector of the weight of the engines. Each element in this vector presents
        the resultant magnitude as if a cut was made at that section. No sign is attached to these values

        :return: Vector of weight at each rib location (see function description)
        :rtype: numpy.ndarray
        """        
        weight_arr = np.zeros(len(self.rib_loc))
        filter = np.array(self.engine.y_rotor_loc) > 0

        if self.engine.ignore_loc != None:
            try:
                filter[self.engine.ignore_locations] = False
            except:
                raise Exception("Unable to assign engine locations that had to be ignored")
                
        y_rotor_loc = np.array(self.engine.y_rotor_loc)[filter]

        for y_loc in y_rotor_loc:
            index = np.argmin(np.absolute(self.rib_loc - y_loc))
            weight_arr[0:index + 1] += self.engine.mass

        return weight_arr*g


       

    def weight_from_tip(self, x):
        """ Computes the weight as a cumulutative vector, where each element represents the weight from the tip

        :param x: Design vector t_sp, t_fl h_st, w_st, t_st, t_sk
        :type x: list
        :return: vector where each element represent the weight from the tip
        :rtype: np.ndarray
        """        
        t_sp, t_fl, h_st, w_st, t_st, t_sk = x

        y = self.rib_loc
        weight_str =  self.str_weight_from_tip(x)
        weight_le = self.le_wingbox_weight_from_tip(x)
        weight_te =  self.te_wingbox_weight_from_tip(x)
        weight_fl = self.fl_weight_from_tip(x)
        weight_spar_web = self.spar_weight_from_tip(x)
        rib_weight = self.rib_weight_from_tip()

        #weight ribs

        return  weight_str +  weight_le + weight_te + weight_fl + weight_spar_web + rib_weight

    def total_weight(self, x):
        """ Returns the total weight of the wingbox

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: The total weight of the wingbox
        :rtype: numpy.ndarray
        """        
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
        :alt: Wingbox Geometery

    """        

    def shear_z_from_tip(self, x):
        """ Internal shear force in the z direction using the sign convention shown in the class description

        :param x: Design vector t_sp, h_st, w_st, t_st, t_sk
        :type x: list
        :return: A vector of the internal shear force from root to tip
        :rtype: numpy.ndarray
        """        
        return (-self.weight_from_tip(x) - self.engine_weight_from_tip() + self.lift_func(self.rib_loc)*self.flight_perf.n_ult)*-1

 
    def moment_x_from_tip(self, x):
        """ Internal moment in the x direction using the sign convention show in the class description.
        To compute the internal moment $\Delta M = - \int V $ is used. This is used from the tip onwards
        to avoid computation of the reaction forces

        :param x: Design vector t_sp, h_st, w_st, t_st, t_sk
        :type x: list
        :return: A vector of the internal shear force from root to tip
        :rtype: numpy.ndarray
        """        
        shear = np.flip(self.shear_z_from_tip(x))
        moment = -cumulative_trapezoid(shear, dx= np.absolute(np.diff(np.flip(self.rib_loc))))
        moment = np.insert(moment, 0, 0)

        return np.flip(moment)

    def moment_y_from_tip(self):
        """ Returns the moment at each section in the form of an array

        :param y: spanwise position
        :type y: float
        :return: moment in  Newton-meters
        :rtype: float
        """        
        y = self.rib_loc
        torque_arr = np.zeros(len(y))
        filter = np.array(self.engine.y_rotor_loc) > 0

        # Assign any engines that have to be ignored in the analysis
        if self.engine.ignore_loc != None:
            try:
                filter[self.engine.ignore_locations] = False
            except:
                raise Exception("Unable to assign engine locations that had to be ignored")

        if self.engine.x_rotor_loc == None:
            y_rotor_loc = np.array(self.engine.y_rotor_loc)[filter]
            x_rotor_loc = np.array(self.engine.x_rotor_rel_loc)[filter] + self.get_x_le(y_rotor_loc)
        elif self.engine.x_rotor_rel_loc == None:   
            y_rotor_loc = np.array(self.engine.y_rotor_loc)[filter]
            x_rotor_loc = np.array(self.engine.x_rotor_loc)[filter]
        else:
            raise Exception("Both relative and absolute x rotor locations were defined")

        for x_loc, y_loc in zip(x_rotor_loc, y_rotor_loc):
            index = np.argmin(np.absolute(y - y_loc))
            torque_arr[0:index + 1] += (self.engine.mass*9.81)*((self.get_x_start_wb(self.rib_loc[0:index + 1]) + self.get_x_end_wb(self.rib_loc[0:index + 1]))/2 - x_loc) #Torque at from tip roto

        return -torque_arr

#-----------Stress functions---------
    def bending_stress_y_from_tip(self, x):
        """ Returns the maximum bending stress as a vector at the rib locations

        :param x: Design vector t_sp,t_fl h_st, w_st, t_st, t_sk
        :type x: list
        :return: A vector of the bending stress at the rib locations from root to tip
        :rtype: numpy.ndarray
        """        
        return self.moment_x_from_tip(x)/self.I_xx(x) * self.height(self.rib_loc)/2

    def shearflow_max_from_tip(self, x):
        """ The following function performs a shear analysis at all rib locations according to the following
        figure. Each region is computed and the maximum shear stress is then returned

        .. image:: wingbox_geometry.png
            :width: 500 

        :param x: Design vector t_sp, t_fl, h_st, w_st, t_st, t_sk
        :type x: list
        :return: _description_
        :rtype: _type_
        """        
        t_sp,t_fl, h_st, w_st, t_st, t_sk = x
        
        #Initialize the necessary arrays for the shear computation
        y = self.rib_loc
        Vz = self.shear_z_from_tip(x)
        T = self.moment_y_from_tip()
        Ixx = self.I_xx(x)
        height = self.height(y)
        chord = self.chord(y)
        Nxy = np.zeros(len(y))
        max_shear_stress = np.zeros(len(y))
        l_sk = self.l_sk_te(y)

        for i in range(len(y)):
            # Base region 1
            def qb1(z):
                warn("Assumes it is a circle but most of the time it is not")
                return -Vz[i]*t_sk*height[i]**2/4*(1- np.cos(z)) / Ixx[i]
            I1 = qb1(pi / 2)

            # Base region 2
            def qb2(z):
                return -Vz[i] * t_sp * z ** 2 / (2 * Ixx[i])
            I2 = qb2(height[i]/2)
            s2 = np.linspace(0, height[i]/2, 100)

            # Base region 3
            def qb3(z):
                return - Vz[i] * t_fl * (0.5 * height[i]) * z / Ixx[i] + I1 + I2
            I3 = qb3(self.width_wingbox * chord[i])
            s3 = np.linspace(0, self.width_wingbox*chord[i],100)

            # Base region 4
            def qb4(z):
                return -Vz[i] * t_sp * z ** 2 / (2 * Ixx[i])
            I4 = qb4(height[i]/2)
            s4=np.linspace(0, height[i]/2, 100)

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
                return -Vz[i]*t_sp/Ixx[i]*(-height[i]/2*z + z**2/2) + I6
            I7 = qb7(-height[i]/2)


            # Base region 8
            def qb8(z):
                return Vz[i] * 0.5 * height[i] * t_fl * z / Ixx[i] + I6 - I7
            I8 = qb8(self.width_wingbox * chord[i])

            # Base region 9
            def qb9(z):
                return -Vz[i] * 0.5 * t_sp * z ** 2 / Ixx[i]
            I9 = qb9(-height[i]/2)

            # Base region 10
            def qb10(z):
                return  -Vz[i]*t_sk*height[i]**2/4*(1- np.cos(z))/Ixx[i] + I8 - I9

            #Torsion
            A1 = float(np.pi*height[i]*chord[i]*self.wing.wingbox_start*0.5)
            A2 = float(height[i]*self.width_wingbox*chord[i])
            A3 = float(height[i]*(1 - self.wing.wingbox_end)*chord[i])

            T_A11 = 0.5 * A1 * self.perimiter_ellipse(height[i],self.wing.wingbox_start*chord[i]) * 0.5 * t_sk
            T_A12 = -A1 * height[i] * t_sp
            T_A13 = 0
            T_A14 = -1/(0.5*self.material.shear_modulus)

            T_A21 = -A2 * height[i] * t_sp
            T_A22 = A2 * height[i] * t_sp * 2 + chord[i]*self.width_wingbox*2*A2*t_fl
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

            B1 =    (0.5*height[i]/t_sk*trapz([qb1(0),qb1(pi/2)], [0, pi / 2]) + 
                    trapz([qb2(0),qb2(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp - 
                    trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0])/ t_sp + 
                    trapz([qb10(-pi/2),qb10(0)], [-pi / 2, 0]) * 0.5 * height[i] / t_sk)

            B2 =    (trapz([qb2(0),qb2(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp +  
                    trapz([qb3(0),qb3(self.width_wingbox*chord[i])], [0, self.width_wingbox * chord[i]]) / t_sk -  
                    trapz([qb7(-0.5*height[i]),qb7(0)], [-0.5 * height[i], 0]) / t_sp + 
                    trapz([qb4(0),qb4(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp +  
                    trapz([qb8(0),qb8(self.width_wingbox*chord[i])], [0, self.width_wingbox * chord[i]]) / t_sk -  
                    trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0]) / t_sp)

            B3 =    (trapz([qb5(0),qb5(l_sk[i])], [0, l_sk[i]]) / t_sk + 
                    trapz([qb6(0),qb6(l_sk[i])], [0, l_sk[i]]) / t_sk + 
                    trapz([qb4(0),qb4(0.5*height[i])], [0, 0.5 * height[i]]) / t_sp - 
                    trapz([qb9(-0.5*height[i]),qb9(0)], [-0.5 * height[i], 0]) / t_sp)

            A = np.array([[A11, A12, 0], [A21, A22, A23], [0, A32, A33]])
            B = -np.array([[B1], [B2], [B3]])
            X = np.linalg.solve(A, B)

            q01 = float(X[0])
            q02 = float(X[1])
            q03 = float(X[2])

            qT1 = float(T_X[0])
            qT2 = float(T_X[1])
            qT3 = float(T_X[2])

            # Compute final shear flow
            q2 = qb2(s2) - q01 - qT1 + q02 + qT2
            q3 = qb3(s3) + q02 + qT2
            q4 = qb4(s4) + q03 +qT3 - q02 - qT2

            max_region2 = max(q2)
            max_region3 = max(q3)
            max_region4 = max(q4)
            
            shear_lst = [max_region2, max_region3, max_region4]
            stress_lst = [max_region2/t_sp, max_region3/t_fl, max_region4/t_sp]

            shear_max_idx = np.argmax(np.abs(shear_lst))
            stress_max_idx = np.argmax(np.abs(stress_lst))

            Nxy[i] = shear_lst[shear_max_idx]
            max_shear_stress[i] = stress_lst[stress_max_idx]
            base_shear_func_lst = [qb1, qb2, qb3,qb4,qb5,qb6,qb7,qb8,qb9,qb10] # Can be accessed by the user but is mostly here for test capability
            cut_shear_func_lst = [q01, q02, q03, qT1, qT2, qT3] # Can be accessed by the user but is mostly here for test capability



        return Nxy, base_shear_func_lst, cut_shear_func_lst

    def distrcompr_max_from_tip(self, x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return self.bending_stress_y_from_tip(x) * t_sk
    


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
        tsmr = (t * self.pitch_str + t_st * self.wing.n_str * (h_st - t)) / self.pitch_str
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
        A=self.wing.n_str*A_st+self.width_wingbox*self.chord(y)*t_sk
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
        y = self.rib_loc
        t_sp, h_st, w_st, t_st, t_sk = x
        Nxy =self.shearflow_max_from_tip(x)
        bend_stress=self.bending_stress_y_from_tip(x)
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
        return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/(i_sigma) - 1)
    
    def f_ult_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.material.sigma_ultimate - 1)
    def flange_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.flange_buckling(t_st,w_st) - 1)
    
    def web_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.web_buckling(t_st,h_st) - 1)
    
    def global_buckling_constr(self,x):
        t_sp, h_st, w_st, t_st, t_sk = x
        return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.global_buckling(h_st,t_st,t_sk) - 1)



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
    y = WingGeom.wing.rib_loc

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

def create_bounds(wing, xlower= [5e-3,1.5e-2, 1.5e-2, 2e-3, 8e-4], xupper= [0.01, 0.01, 1e-1, 3.3e-2, 1e-1]):
    """_summary_

    :param wing: _description_
    :type wing: _type_
    :param xlower: _description_
    :type xlower: _type_
    :param xupper: _description_
    :type xupper: _type_
    """    

    #------SET BOUNDS--------
    height_tip = wing.chord_root - wing.chord_root*(1 - wing.taper)*2 
    if height_tip/2<xupper[0]: 
        warn(f"The given spar thickness limit {xupper[0]} was limited by the height of the wingbox, values was changed to {height_tip/2}")
        xupper[0] = height_tip/2
    if height_tip/2<xupper[1]: 
        warn(f"The given stringer height limit {xupper[0]} was limited by the height of the wingbox, values was changed to {height_tip/2}")
        xupper[1] = height_tip/2


    return np.vstack((xlower, xupper)).T


# def GA_optimizer(aero, airfoil, engine, flight_perf, material, wing, bounds):

#     WingGeom = WingboxGeometry(aero, airfoil, engine, flight_perf, material, wing)
#     Constr =  Constraints(aero, airfoil, engine, flight_perf, material, wing)

#     # ------SET INITIAL VALUES------
#     tsp= wing.spar_thickness
#     hst= wing.stringer_height
#     wst= wing.stringer_width
#     tst= wing.stringer_thickness
#     tsk= wing.skin_thickness


#     X = [tsp, hst, wst, tst, tsk]
#     y = WingGeom.y


#     objs = [WingGeom.total_weight]

#     constr_ieq = [
#         lambda x: -Constr.buckling_constr(x)[0],
#         lambda x: -Constr.von_Mises(x)[0],
#         lambda x: -Constr.str_buckling_constr(x)[0],
#         lambda x: -Constr.f_ult_constr(x)[0],
#         lambda x: -Constr.flange_buckling_constr(x)[0],
#         lambda x: -Constr.web_buckling_constr(x)[0],
#         lambda x: -Constr.global_buckling_constr(x)[0],
#     ]

#     problem = FunctionalProblem(len(X),
#                                 objs,
#                                 constr_ieq=constr_ieq,
#                                 xl=xlower,
#                                 xu=xupper,
#                                 )

#     method = GA(pop_size=50, eliminate_duplicates=True)

#     resGA = minimizeGA(problem, method, termination=('n_gen', 50   ), seed=1,
#                     save_history=True, verbose=True)
#     print('GA optimum variables', resGA.X)
#     print('GA optimum weight', resGA.F)
#     pass

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

        

