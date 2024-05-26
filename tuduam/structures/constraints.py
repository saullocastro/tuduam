import numpy as np

from scipy.interpolate import CubicSpline
from .wingbox import IdealWingbox
from ..data_structures import *



class IsotropicWingboxConstraints:
    """
          Some description.
    
        Parameters
        ----------
        features_to_drop : str or list, default=None
            Variable(s) to be dropped from the dataframe
    
        Methods
        -------
        fit:
           some description
        transform:
          some description
        fit_transform:
          some description
        """
    def __init__(self, wingbox:IdealWingbox, material_struct: Material, len_to_rib: float) -> None:
        self.wingbox = wingbox
        self.material_struct = material_struct
        self.len_to_rib = len_to_rib
        self.pnl_lst =  [i for i in self.wingbox.panel_dict.values()] 
        self.tens_pnl_idx =  [idx for idx, i in enumerate(self.wingbox.panel_dict.values()) if (i.b1.sigma > 0) and (i.b2.sigma > 0)] # Panel which are in tension since for some of the constraints it is not relevant here

        # Value used for the interpolation to get Kb
        x = [1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ]
        y = [9.5, 7.2, 6.4, 6, 5.8, 5.9, 5.8, 5.6, 5.4]

        # Interpolator for critical shear function
        self.kb_spline = CubicSpline(x,y, extrapolate=False) 
    
    def _kb_interp(self, ar_lst: list) -> np.ndarray:
        """ Computes the coefficients for the critical shear stability of a sheet.

        :param ar_lst: List containing the aspect ratio of each sheet
        :type ar_lst: list
        :return: Returns an array with the criticial shear coefficients
        :rtype: np.ndarray
        """        
        res = self.kb_spline(ar_lst)
        res = np.nan_to_num(res, nan= 5.) # Set values that were outside of the interpolation range to 5.
        return res

    def crit_instability_compr(self) -> list:
        r""" 
        Compute the elastic instability of a flat sheet in compression for each panel in the idealized wingbox using 
        the equation shown below.


        .. math::
            \sigma_{cr} = k_c  \frac{pi^2 E}{12(1 - \nu)} \left(\frac{t_{sk}}{b}\right)^2

        Where b is the short dimension of plate or loaded edge. For :math:`K_c` a value of 4 was chosen. Please see the figure below for the reasoning.
        Since all edges are considerded simpy supported from either the stringer or the ribs it is conservative to go for a value of 4.
        For any other information please see source 1.

        .. image:: ../_static/buckle_coef.png
            :width: 300
        
        **Future improvements**
        1.  compute the proper buckling coefficient in real time using the sheet aspect ratio (or just check for aspect ratio's smaller than 1. These seem to be the most relevant to catch)
        2. A plasticity factor could be implemented (see source 1, equation C5.2)

        **Bibliography**

        1. Chapter C5.2, Bruhn, Analysis & Design of Flight Vehicle Structures

        :param wingbox: Ideal wingbox class which is utilized to create constraints per panel 
        :type wingbox: IdealWingbox
        :param material_struct: Data structure containing all material propertie
        :type material_struct: Material
        :param len_to_rib: The  distance to the next rib
        :type len_to_rib: float
        :return: The critical buckling stress due to compression
        :rtype: float
        """    
        kc = 4 # buckling coefficient (currently very conservative but should be computed in real time using)
        t_arr = np.array([i.t_pnl for i in self.pnl_lst])  # thickness array of all the panels
        b = np.array([min(i.length(), self.len_to_rib) for i in self.pnl_lst])
        res = kc*np.pi**2*self.material_struct.young_modulus/(12*(1 - self.material_struct.poisson ** 2)) * (t_arr/b) ** 2
        # res[self.tens_pnl_idx] = 1
        return res


    def crit_instability_shear(self) -> list:#TODO
        r""" 
        Compute the elastic instability of a flat sheet in shear for each panel in the idealized wingbox using  
        the equation shown below. Very similar for the case in compression (see :func:`crit_instability_compr`) except for the shear buckling coefficient.


        .. math::
            \sigma_{cr} = k_b  \frac{pi^2 E}{12(1 - \nu)} \left(\frac{t_{sk}}{b}\right)^2

        Where b is the short dimension of plate or loaded edge. For :math:`K_b`, the shear buckling coefficient, 
        the figure below was used to make a polynomial fit of the 3rd degree. The dataset used was as follows:

        x = [1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ] 
        y = [9.5, 7.2, 6.4, 6, 5.8, 5.9, 5.8, 5.6, 5.4]


        The interpolator was created in the initilisation of the class. For any other information please see source 1.

        .. image:: ../_static/shear_buck_coef.png
            :width: 300
        
        **Future improvements**
        1. A plasticity factor could be implemented (see source 1, equation C5.2)

        **Bibliography**

        1. Chapter C5.7, Bruhn, Analysis & Design of Flight Vehicle Structures

        :param wingbox: Ideal wingbox class which is utilized to create constraints per panel 
        :type wingbox: IdealWingbox
        :param material_struct: Data structure containing all material propertie
        :type material_struct: Material
        :param len_to_rib: The length to the next rib, thus the length to the next simply supported edge in the spanwise direction.
        :type len_to_rib: float
        :return: The critical buckling stress due to compression
        :rtype: float
        """    
        t_arr = np.array([i.t_pnl for i in self.pnl_lst])  # thickness array of all the panels

        # For the length of each panel we must look at the stringers 

        asp_lst = [] # Aspect ratio list
        b_lst = []
        for pnl in self.pnl_lst:
            len_pnl: float = pnl.length()
            # Make sure to divide by the shortest side
            if self.len_to_rib < len_pnl:
                asp_lst.append(len_pnl/self.len_to_rib)
                b_lst.append(self.len_to_rib)
            else: 
                asp_lst.append(self.len_to_rib/len_pnl)
                b_lst.append(len_pnl)

        asp_lst = np.array(asp_lst)
        b_lst = np.array(b_lst)

        # Compute the shear buckling coefficient using poly fit described in the docstring
        kb_vec: np.ndarray =  self._kb_interp(asp_lst)

        res = kb_vec* np.pi ** 2 * self.material_struct.young_modulus/(12*(1 - self.material_struct.poisson**2)) * (t_arr/ b_lst)**2
        return res


    def interaction_curve(self):
        r""" The following function ensures the panel remains below the interaction curve of a composite panel
        under combined compression and shear forces. This function is designed to be used with the :func:`SectionOptimization._get_constraint_vector` however it
        can also be used to check this specific constraints for any given design. The following equation is used for the 
        interaction curve which has been rewritten from equation 6.38, page 144 in source [1]:

        .. math::
            -\frac{N_x}{N_{x,crit}} - \left(\frac{N_{xy}}{N_{xy,crit}}\right)^2 + 1 > 0



        **Bibliography**
        [1] Kassapoglou, C. (2010). page 137, equation 6.38, Design and analysis of composite structures: With applications to aerospace structures. John Wiley & Sons.

        :param wingbox: _description_
        :type wingbox: IdealWingbox
        :param material_struct: _description_
        :type material_struct: Material
        :param len_to_rib: _description_
        :type len_to_rib: float
        :return: _description_
        :rtype: _type_
        """    

        
        area_pnl: list = [pnl.t_pnl*pnl.length() for pnl in self.pnl_lst]
        Nx_crit = self.crit_instability_compr()*area_pnl
        Nxy_crit = self.crit_instability_shear()*area_pnl

        Nx: list = np.abs([min(pnl.b1.sigma*pnl.b1.A, pnl.b2.sigma*pnl.b2.A) for pnl in self.pnl_lst]) # Take the maximum of the two booms
        Nxy: list = np.abs([pnl.q_tot*pnl.length() for pnl in self.pnl_lst])

        interaction_constr = -Nx/ Nx_crit - (Nxy/Nxy_crit)**2 + 1
        # interaction_constr[self.tens_pnl_idx] = 1


        return interaction_constr


    def column_str_buckling(self) -> list:#
        r"""
        The following contraints ensure that local skin buckling occurs before colum stringer buckling, which makes failure of the structure
        more predictable. In order to be able to compare it to the skin buckling load, the critical distributed compressive load
        acting on all stringers is computed in Equation 50, this approach being also conservative, given that in reality
        the skin beneath the stringer takes part of the compression load. The constraint is expressed below as well.

        .. math::
            N_{col} = \frac{\pi^2 E I}{L^2 2 w_{st}}

        .. math::
             N_{col} - \sigma_{cr}*t_{sk} \geq 0
        

        :return: list
        :rtype:list
        """        
        t_st = self.wingbox.wingbox_struct.t_st
        h_st = self.wingbox.wingbox_struct.h_st
        w_st = self.wingbox.wingbox_struct.w_st

        I_arr = np.array([t_st*h_st**3/12 + 2*(w_st - t_st)*t_st**3/12 +  i.t_pnl*w_st*(0.5*h_st)**2 for i in self.pnl_lst])
        n_col = (np.pi**2*self.material_struct.young_modulus*I_arr)/(self.len_to_rib**2*2*w_st)

        t_sk_arr = np.array([i.t_pnl for i in self.pnl_lst])
        crit_stress_arr = self.crit_instability_compr()

        return n_col - t_sk_arr*crit_stress_arr



    def stringer_flange_buckling(self):
        r"""
        The individual flanges of the stringer can also buckle, (see :meth:`crit_instability_compr` for the equation and its parameters). The buckling coefficient changes as one edge is free and one is simply supported, thus k is conser-
        vatively chosen to be 2. The constraint is expressed in Equation 52.

        .. math::
            \sigma_{cr,fl} - \sigma_{cr,loc} \geq 0
        """        

        sigma_loc = self.crit_instability_compr()

        kc = 2 # buckling coefficient 
        t_st =  self.wingbox.wingbox_struct.t_st
        b =  self.wingbox.wingbox_struct.w_st
        sigma_fl= kc*np.pi**2*self.material_struct.young_modulus/(12*(1 - self.material_struct.poisson ** 2)) * (t_st/b) ** 2
        return sigma_fl - sigma_loc

    def stringer_web_buckling(self):
        r"""
        The individual webs of the stringers can also buckle, (see :meth:`crit_instability_compr` for the equation and its parameters). 
        he edges can be conservatively considered simply supported, the buckling coefficient k being 4. The constraints is expressed below:

        .. math::
            \sigma_{cr,web} - \sigma_{cr,loc} \geq 0
        """        

        sigma_loc = self.crit_instability_compr()

        kc = 4 # buckling coefficient 
        t_st =  self.wingbox.wingbox_struct.t_st
        b =  self.wingbox.wingbox_struct.h_st
        sigma_web= kc*np.pi**2*self.material_struct.young_modulus/(12*(1 - self.material_struct.poisson ** 2)) * (t_st/b) ** 2
        return sigma_web - sigma_loc


    # def global_local(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     glob = self.global_buckling(h_st, t_st, t_sk)
    #     loc = self.local_buckling(t_sk)
    #     diff = glob - loc
    #     return diff


    # def local_column(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     col = self.column_st(h_st,w_st,t_st, t_sk)
    #     loc = self.local_buckling(t_sk)
    #     # print("col=",col/1e6)
    #     # print("loc=",loc/1e6)
    #     diff = col - loc
    #     return diff


    # def flange_loc_loc(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     flange = self.flange_buckling(t_st,w_st)
    #     loc = self.local_buckling(t_sk)
    #     diff = flange - loc
    #     return diff


    # def web_flange(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     web = self.web_buckling(t_st, h_st)
    #     loc = self.local_buckling(t_sk)
    #     diff =web-loc
    #     return diff


    def von_Mises(self):
        r""" THe following constraint implemetns the von mises failure criterion which is defined as follows for the case where only a direct stress
        in the y axis occurs and one shear stress is present.

        .. math::
             \sigma_y & \geq  \sigma_v \\
              \sigma_y  - \sqrt{\sigma_{11}^2 + 3\tau^2} & \geq  0 \\


        :return: _description_
        :rtype: _type_
        """        

        shear_arr = np.array([i.tau for i in  self.pnl_lst])
        direct_stress_arr = np.array([(i.b1.sigma + i.b2.sigma)/2 for i in  self.pnl_lst])
        return self.material_struct.sigma_yield - np.sqrt(direct_stress_arr**2 + 3*shear_arr**2)


    # def crippling(self, x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     A = self.get_area_str(h_st,w_st,t_st)
    #     col = self.column_st( h_st,w_st,t_st,t_sk)
    #     crip = t_st * self.material.beta_crippling * self.material.sigma_yield* ((self.material.g_crippling * t_st ** 2 / A) * np.sqrt(self.material.young_modulus / self.material.sigma_yield)) ** self.m_crip
    #     return crip

    # #----OWN CONSTRAINTS-----
    # def str_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     Ist = t_st * h_st ** 3 / 12 + (w_st - t_st) * t_st ** 3 / 12 + t_sk**3*w_st/12+t_sk*w_st*(0.5*h_st)**2
    #     i= pi ** 2 * self.material.young_modulus * Ist / (self.rib_pitch ** 2)#TODO IF HE FAILS REPLACE SPAN WITH RIB PITCH
    #     i_sigma = (i/self.get_area_str(h_st,w_st,t_st))#convert to stress
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/(i_sigma) - 1)

    # def f_ult_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.material.sigma_ultimate - 1)
    # def flange_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.flange_buckling(t_st,w_st) - 1)

    # def web_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.web_buckling(t_st,h_st) - 1)

    # def global_buckling_constr(self,x):
    #     t_sp, h_st, w_st, t_st, t_sk = x
    #     return -1*(self.material.safety_factor*self.bending_stress_y_from_tip(x)/self.global_buckling(h_st,t_st,t_sk) - 1)

 
