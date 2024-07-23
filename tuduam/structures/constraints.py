import numpy as np

from scipy.interpolate import CubicSpline
from .wingbox import IdealWingbox
from ..data_structures import *


class IsotropicWingboxConstraints:
    """
    Class to handle constraints on an isotropic wingbox structure.

    Parameters
    ----------
    wingbox : IdealWingbox
        The wingbox structure for which constraints are evaluated.
    material_struct : Material
        the material properties of the wingbox structure.
    len_to_rib : float
        the length between ribs in the wingbox structure.

    attributes
    ----------
    pnl_lst : list
        list of panels in the wingbox.
    tens_pnl_idx : list
        indices of panels which are in tension.
    t_st : float
        thickness of the stringer.
    w_st : float
        width of the stringer.
    h_st : float
        height of the stringer.
    area_str : float
        area of the stringer.
    kb_spline : cubicspline
        interpolator for critical shear function.
    """


    def __init__(
        self, wingbox: IdealWingbox, material_struct: Material, len_to_rib: float
        ) -> None:

        """
        initialize the isotropicwingboxconstraints class.

        parameters
        ----------
        wingbox : idealwingbox
            the wingbox structure for which constraints are evaluated.
        material_struct : material
            the material properties of the wingbox structure.
        len_to_rib : float
            the length between ribs in the wingbox structure.

        raises
        ------
        runtimeerror
            raised if both stringer area and stringer geometry are specified.
        """

 

        self.wingbox = wingbox
        self.material_struct = material_struct
        self.len_to_rib = len_to_rib
        self.pnl_lst = [i for i in self.wingbox.panel_dict.values()]
        self.tens_pnl_idx = [
            idx
            for idx, i in enumerate(self.wingbox.panel_dict.values())
            if (i.b1.sigma > 0) and (i.b2.sigma > 0)
        ]  # panel which are in tension since for some of the constraints it is not relevant here
        self.t_st = self.wingbox.wingbox_struct.t_st
        self.w_st = self.wingbox.wingbox_struct.w_st
        self.h_st = self.wingbox.wingbox_struct.h_st

        if self.t_st is None or self.w_st is None or self.h_st is None:
            self.area_str = self.wingbox.wingbox_struct.area_str
        elif self.wingbox.wingbox_struct.area_str is None:
            self.area_str = (
                2 * self.w_st * self.t_st + (self.h_st - 2 * self.t_st) * self.t_st
            )
        else:
            raise RuntimeError(
                "both stringer area and stringer geometry were specified"
            )

        # value used for the interpolation to get kb
        x = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        y = [9.5, 7.2, 6.4, 6, 5.8, 5.9, 5.8, 5.6, 5.4]

        # interpolator for critical shear function
        self.kb_spline = CubicSpline(x, y, extrapolate=False)

    def _kb_interp(self, ar_lst: list) -> np.ndarray:
        """
        computes the coefficients for the critical shear stability of a sheet.

        parameters
        ----------
        ar_lst : list
            list containing the aspect ratio of each sheet.

        returns
        -------
        np.ndarray
            an array with the critical shear coefficients.
        """
   
        res = self.kb_spline(ar_lst)
        res = np.nan_to_num(
            res, nan=5.0
        )  # set values that were outside of the interpolation range to 5.
        return res

    def crit_instability_compr(self) -> list:
        r"""
        compute the elastic instability of a flat sheet in compression for each panel in the idealized wingbox using 
        the equation shown below.

        .. math::
            \sigma_{cr} = k_c  \frac{\pi^2 e}{12(1 - \nu^2)} \left(\frac{t_{sk}}{b}\right)^2

        where `b` is the short dimension of the plate or loaded edge. for :math:`k_c`, a value of 4 was chosen. please see the figure below for the reasoning.
        since all edges are considered simply supported from either the stringer or the ribs, it is conservative to go for a value of 4.
        for any other information please see source 1.

        .. image:: ../_static/buckle_coef.png
            :width: 300
        
        **future improvements**
        1. compute the proper buckling coefficient in real time using the sheet aspect ratio (or just check for aspect ratios smaller than 1. these seem to be the most relevant to catch).
        2. a plasticity factor could be implemented (see source 1, equation c5.2).

        **bibliography**

        1. chapter c5.2, bruhn, analysis & design of flight vehicle structures

        parameters
        ----------
        wingbox : idealwingbox
            ideal wingbox class which is utilized to create constraints per panel.
        material_struct : material
            data structure containing all material properties.
        len_to_rib : float
            the distance to the next rib.

        returns
        -------
        float
            the critical buckling stress due to compression.
        """

      
        kc = 4  # buckling coefficient (currently very conservative but should be computed in real time using)
        t_arr = np.array(
            [i.t_pnl for i in self.pnl_lst]
        )  # thickness array of all the panels
        b = np.array([min(i.length(), self.len_to_rib) for i in self.pnl_lst])
        res = (
            kc
            * np.pi ** 2
            * self.material_struct.young_modulus
            / (12 * (1 - self.material_struct.poisson ** 2))
            * (t_arr / b) ** 2
        )
        # res[self.tens_pnl_idx] = 1
        return res

    def crit_instability_shear(self) -> list:  # todo
        r"""
        compute the elastic instability of a flat sheet in shear for each panel in the idealized wingbox using  
        the equation shown below. this is very similar to the case in compression (see :func:`crit_instability_compr`) except for the shear buckling coefficient.

        .. math::
            \sigma_{cr} = k_b  \frac{\pi^2 e}{12(1 - \nu^2)} \left(\frac{t_{sk}}{b}\right)^2

        where `b` is the short dimension of the plate or loaded edge. for :math:`k_b`, the shear buckling coefficient, 
        the figure below was used to make a polynomial fit of the 3rd degree. the dataset used was as follows:

        x = [1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. ] 
        y = [9.5, 7.2, 6.4, 6, 5.8, 5.9, 5.8, 5.6, 5.4]

        the interpolator was created in the initialization of the class. for any other information please see source 1.

        .. image:: ../_static/shear_buck_coef.png
            :width: 300
        
        **future improvements**
        1. a plasticity factor could be implemented (see source 1, equation c5.2).

        **bibliography**

        1. chapter c5.7, bruhn, analysis & design of flight vehicle structures

        parameters
        ----------
        wingbox : idealwingbox
            ideal wingbox class which is utilized to create constraints per panel.
        material_struct : material
            data structure containing all material properties.
        len_to_rib : float
            the length to the next rib, thus the length to the next simply supported edge in the spanwise direction.

        returns
        -------
        float
            the critical buckling stress due to shear.
        """

 
        t_arr = np.array(
            [i.t_pnl for i in self.pnl_lst]
        )  # thickness array of all the panels

        # For the length of each panel we must look at the stringers

        asp_lst = []  # Aspect ratio list
        b_lst = []
        for pnl in self.pnl_lst:
            len_pnl: float = pnl.length()
            # Make sure to divide by the shortest side
            if self.len_to_rib < len_pnl:
                asp_lst.append(len_pnl / self.len_to_rib)
                b_lst.append(self.len_to_rib)
            else:
                asp_lst.append(self.len_to_rib / len_pnl)
                b_lst.append(len_pnl)

        asp_lst = np.array(asp_lst)
        b_lst = np.array(b_lst)

        # Compute the shear buckling coefficient using poly fit described in the docstring
        kb_vec: np.ndarray = self._kb_interp(asp_lst)

        res = (
            kb_vec
            * np.pi ** 2
            * self.material_struct.young_modulus
            / (12 * (1 - self.material_struct.poisson ** 2))
            * (t_arr / b_lst) ** 2
        )
        return res

    def interaction_curve(self):
        r"""
        The following function ensures the panel remains below the interaction curve of a composite panel
        under combined compression and shear forces. This function is designed to be used with the :func:`SectionOptimization._get_constraint_vector` however it
        can also be used to check this specific constraint for any given design. The following equation is used for the 
        interaction curve which has been rewritten from equation 6.38, page 144 in source [1]:

        .. math::
            -\frac{N_x}{N_{x,crit}} - \left(\frac{N_{xy}}{N_{xy,crit}}\right)^2 + 1 > 0

        **Bibliography**
        [1] Kassapoglou, C. (2010). Design and analysis of composite structures: With applications to aerospace structures. John Wiley & Sons, page 137, equation 6.38.

        Parameters
        ----------
        wingbox : IdealWingbox
            Ideal wingbox class which is utilized to create constraints per panel.
        material_struct : Material
            Data structure containing all material properties.
        len_to_rib : float
            The distance to the next rib.

        Returns
        -------
        bool
            True if the panel remains below the interaction curve, False otherwise.
        """


        area_pnl: list = [pnl.t_pnl * pnl.length() for pnl in self.pnl_lst]
        Nx_crit = self.crit_instability_compr() * area_pnl
        Nxy_crit = self.crit_instability_shear() * area_pnl

        Nx: list = np.abs(
            [
                min(pnl.b1.sigma * pnl.b1.A, pnl.b2.sigma * pnl.b2.A)
                for pnl in self.pnl_lst
            ]
        )  # Take the maximum of the two booms
        Nxy: list = np.abs([pnl.q_tot * pnl.length() for pnl in self.pnl_lst])

        interaction_constr = -Nx / Nx_crit - (Nxy / Nxy_crit) ** 2 + 1
        # interaction_constr[self.tens_pnl_idx] = 1

        return interaction_constr

    def _n_col(self) -> np.ndarray:
        r"""
        The critical distributed compressive load acting on all stringers is computed in Equation 50, 
        this approach being also conservative, given that in reality the skin beneath the stringer takes part of the compression load. 

        .. math::
            N_{col} = \frac{\pi^2 E I}{L^2 2 w_{st}}

        Returns
        -------
        np.ndarray
            The critical distributed compressive load acting on all stringers in N/m.
        """


        t_st = self.t_st
        h_st = self.h_st
        w_st = self.w_st

        I_arr = np.array(
            [
                t_st * h_st ** 3 / 12
                + 2 * (w_st - t_st) * t_st ** 3 / 12
                + i.t_pnl * w_st * (0.5 * h_st) ** 2
                for i in self.pnl_lst
            ]
        )
        n_col = (np.pi ** 2 * self.material_struct.young_modulus * I_arr) / (
            self.len_to_rib ** 2 * 2 * w_st
        )
        return n_col

    def column_str_buckling(self) -> list:  #
        r"""
        The following constraints ensure that local skin buckling occurs before column stringer buckling, which makes failure of the structure
        more predictable. In order to be able to compare it to the skin buckling load, the critical distributed compressive load
        acting on all stringers is computed in Equation 50. This approach is also conservative, given that in reality
        the skin beneath the stringer takes part of the compression load. The constraint is expressed below as well.

        .. math::
            N_{col} = \frac{\pi^2 E I}{L^2 2 w_{st}}

        .. math::
            N_{col} - \sigma_{cr} t_{sk} \geq 0

        Returns
        -------
        list
            A list of constraints ensuring local skin buckling occurs before column stringer buckling.
        """

   

        n_col = self._n_col()
        t_sk_arr = np.array([i.t_pnl for i in self.pnl_lst])
        crit_stress_arr = self.crit_instability_compr()

        return n_col - t_sk_arr * crit_stress_arr

    def stringer_flange_buckling(self):
        r"""
        The individual flanges of the stringer can also buckle (see :meth:`crit_instability_compr` for the equation and its parameters). 
        The buckling coefficient changes as one edge is free and one is simply supported, thus `k` is conservatively chosen to be 2. 
        The constraint is expressed in Equation 52.

        .. math::
            \sigma_{cr,fl} - \sigma_{cr,loc} \geq 0

        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise.
        """

 

        sigma_loc = self.crit_instability_compr()

        kc = 2  # buckling coefficient
        t_st = self.wingbox.wingbox_struct.t_st
        b = self.wingbox.wingbox_struct.w_st
        sigma_fl = (
            kc
            * np.pi ** 2
            * self.material_struct.young_modulus
            / (12 * (1 - self.material_struct.poisson ** 2))
            * (t_st / b) ** 2
        )
        return sigma_fl - sigma_loc

    def stringer_web_buckling(self):
        r"""
        The individual webs of the stringers can also buckle (see :meth:`crit_instability_compr` for the equation and its parameters).
        The edges can be conservatively considered simply supported, the buckling coefficient `k` being 4. The constraint is expressed below:

        .. math::
            \sigma_{cr,web} - \sigma_{cr,loc} \geq 0

        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise.
        """

        sigma_loc = self.crit_instability_compr()

        kc = 4  # buckling coefficient
        t_st = self.wingbox.wingbox_struct.t_st
        b = self.wingbox.wingbox_struct.h_st
        sigma_web = (
            kc
            * np.pi ** 2
            * self.material_struct.young_modulus
            / (12 * (1 - self.material_struct.poisson ** 2))
            * (t_st / b) ** 2
        )
        return sigma_web - sigma_loc

    def crippling(self) -> np.ndarray:
        r"""
        Crippling is a form of local buckling that occurs in columns, leading to the failure of the structure. It is
        related to plastic deformation of the stringer, and it is desired to have the load higher than the column buckling
        of the stringers (and subsequently higher than the local skin buckling), as crippling leads to the entire failure
        of the structure. The crippling load is expressed in Equation 56 (in N/m), where for aluminium alloys the
        constants are β=1.42, m=0.85, and for Z stringers g=5 [4]. The constraint is stated in Equation 57.

        .. math::
            N_f = t_{st} \beta \sigma_y \left[\frac{g t_{st}^2}{A_{st}} \sqrt{\frac{E}{\sigma_y}} \right]^m

        Returns
        -------
        np.ndarray
            An array containing the inequality constraints described above, where each element represents a panel. 
            The elements should be greater than zero to satisfy the constraint.
        """

        n_col = self._n_col()
        beta = 1.42
        m = 0.85
        g = 5
        E = self.material_struct.young_modulus
        sigma_y = self.material_struct.sigma_yield
        n_f = (
            self.t_st
            * beta
            * sigma_y
            * (g * self.t_st ** 2 / self.area_str * np.sqrt(E / sigma_y)) ** m
        )
        return n_f - n_col

    def global_skin_buckling(self):
        r"""
        A stiffened panel can also buckle as a whole. 
        In this case, the width of the panel is utilized instead of the stringer pitch, and simply supported conditions 
        can be assumed. The contribution of the stringers that still provide a stiffening effect can be considered by 
        smearing their thickness to the skin thickness, as in Equation 47.

        .. math::
            t_{smeared} = \frac{t_{sk} \cdot b + N_{str} \cdot A_{st}}{b}

        The smeared thickness is substituted in the equation for critical sheet compression (see :meth:`crit_instability_compr`).
        The constraint is expressed below.

        .. math::
            \sigma_{cr,glob} - \sigma_{cr,loc} \geq 0

        Returns
        -------
        bool
            True if the constraint is satisfied, False otherwise.
        """


        kc = 4  # Chosen conservatily
        b_arr = np.array([min(i.length(), self.len_to_rib) for i in self.pnl_lst])

        t_smr_arr = []  # Smeared thickness list

        for pnl, b in zip(self.pnl_lst, b_arr):
            res = (pnl.t_pnl * b + self.area_str) / b
            t_smr_arr.append(res)

        sigma_glob = (
            kc
            * np.pi ** 2
            * self.material_struct.young_modulus
            / (12 * (1 - self.material_struct.poisson ** 2))
            * (t_smr_arr / b_arr) ** 2
        )
        sigma_loc = self.crit_instability_compr()

        return sigma_glob - sigma_loc

    # def f_ult(b,c_r,L,b_st,h_st,t_st,w_st,t):
    # A_st = area_st(h_st,t_st,w_st)
    # n=n_st(c_r,b_st)
    # tarr=t_arr(b,L,t)
    # c=chord(b,c_r)
    # h=height(b,c_r)
    # stations=rib_coordinates(b,L)
    # f_uts=np.zeros(len(tarr))
    # for i in range(len(tarr)):
    #     A=n*A_st+0.6*c(stations[i])*tarr[i]
    #     f_uts[i]=sigma_uts*A
    # return f_uts

    # def post_buckling(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st, t):
    #     f=f_ult(b,c_r,L,b_st,h_st,t_st,w_st,t)
    #     ratio=2/(2+1.3*(1-1/pb))
    #     px= n_max*shear_force(b, c_r, t_sp, t_rib, L, b_st, h_st,t_st,w_st,t)
    #     diff=np.subtract(ratio*f,px)
    #     return diff[0]

    def von_Mises(self):
        r"""
        The following constraint implements the von Mises failure criterion, which is defined as follows for the case where only a direct stress
        in the y-axis occurs and one shear stress is present.

        .. math::
            \sigma_y & \geq  \sigma_v \\
            \sigma_y  - \sqrt{\sigma_{11}^2 + 3\tau^2} & \geq  0 \\

        Returns
        -------
        np.ndarray 
            An array of where each element represents a the von Mises condition in a section. If it is met the value should be greater than zero. 
        """



        shear_arr = np.array([i.tau for i in self.pnl_lst])
        direct_stress_arr = np.array(
            [(i.b1.sigma + i.b2.sigma) / 2 for i in self.pnl_lst]
        )
        return self.material_struct.sigma_yield - np.sqrt(
            direct_stress_arr ** 2 + 3 * shear_arr ** 2
        )
