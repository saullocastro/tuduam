from pydantic import BaseModel, ConfigDict, FilePath, conlist, Field
from typing import Optional, List 
import json
import numpy as np

#TODO Create documentation usig sphinx-pydantic plugin

class Parent(BaseModel):
    model_config = ConfigDict(extra='forbid',
                              validate_assignment=True)

class SingleWing(Parent):
    aspect_ratio: float  = Field(gt=0, le=20)
    taper: float  = Field(gt=0, le=1)
    quarterchord_sweep: float  
    washout: float  
    """"The geometric twist between the tip and root chord 
    in radians. Negative value indicates washin i.e the tip is at a
    higher AoA."""
    x_le_root_global: Optional[float]  =  None               # y position leading edge mean aerodynamic chord
    """"The coordinate of the leading edge of the root chord. Defines the 
    positioning of the wing"""
    surface: Optional[float]  =  None
    span: Optional[float]  =  None
    chord_root: Optional[float]  =  None
    chord_tip: Optional[float]  =  None
    chord_mac: Optional[float]  =  None             # Mean Aerodynamic chord
    sweep_le: Optional[float]  =  None
    x_lemac_local: Optional[float]  =  None               # x position (longitudinal) leading edge mean aerodynamic chord
    y_mac: Optional[float]  =  None               # y position leading edge mean aerodynamic chord
    mass: Optional[float]  =  None               # Mass of both wings

    #--------------wingbox structure --------------
    wingbox_start: Optional[float]  =  None
    wingbox_end: Optional[float]  =  None
    n_ribs: Optional[int]  =  None
    n_str: Optional[int]  =  None
    """"The amount of stringers on one flange"""
    t_rib: Optional[float]  =  None
    spar_thickness: Optional[float]  =  None
    fl_thickness: Optional[float]  =  None
    stringer_height: Optional[float]  =  None
    stringer_width: Optional[float]  =  None
    stringer_thickness: Optional[float]  =  None
    skin_thickness: Optional[float]  =  None
    rib_loc: Optional[List]  =  None
    """"A list of the spanwise position of the ribs. Can be left to None
    as it will be automatically created. Howeve if pre-defined it will keep this list"""


    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class Airfoil(Parent):
    cl_alpha: float  = Field(gt=2)
    thickness_to_chord: Optional[float] = None

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class Fuselage(Parent):
    length_fuselage: Optional[float] = None
    width_fuselage: Optional[float] = None
    height_fuselage: Optional[float] = None
    mass: Optional[float] = None

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")


class HybridPowertrain(Parent):
    fuel_cell_mass: Optional[float] = None
    battery_mass : Optional[float] = None

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class Engine(Parent):
    n_engines: int  = Field(gt=0)
    diameter: Optional[float] = None
    thrust: Optional[float] = None
    mass: Optional[float] = None
    power_grav_density: Optional[float] = None
    power_vol_density: Optional[float] = None
    x_rotor_loc: Optional[List] = None
    """"A list of the x coordinates of your engines for both sides. Thus if you have 4
    engines in total, the list must be 4 elements long. eg [2,4,4,2]. See section about 
    coordinate systems if you have questions regardign this."""
    x_rotor_rel_loc: Optional[List] = None
    """"Same as x_rotor_locations but relative to the wing, that is the leading edge of the
    spanwise location of the engine. Negative values are infront the wing"""
    y_rotor_loc: Optional[List] = None
    """"A list of the y coordinates of your engines for both sides. Thus if you have 4
    engines in total, the list must be 4 elements long and correspond to
    the elements x_rotor_locations. eg [-6,-3,3,6]. See section about 
    coordinate systems if you have questions regardign this."""
    ignore_loc: Optional[List] = None
    """"Index of which engne positions to ignore in the wingbox analysis. For example when 
    the engine is placed on the fuselage"""


    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class VTOL(Parent):
    mtom: Optional[float] = None
    oem: Optional[float] = None

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class FlightPerformance(Parent):
    wingloading: Optional[float] = None
    cL_cruise: Optional[float] = None
    mission_energy: Optional[float] = None
    n_ult: Optional[float] = None
    v_cruise: Optional[float] = None
    h_cruise: Optional[float] = None
    """"cruise altitude"""

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class Aerodynamics(Parent):
    cL_alpha: Optional[float] = None
    alpha_zero_lift: Optional[float] = None
    cd0: Optional[float] = None
    """"Zero lift drag coefficient"""
    spanwise_points: Optional[int] = None
    """"The spanwise points used in the Weissinger-L/Lifiting line method.
    Odd integer"""

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")


class Material(Parent):
    young_modulus: Optional[float] = None
    safety_factor: Optional[float] = None
    shear_modulus: Optional[float] = None
    poisson: Optional[float] = None
    density: Optional[float] = None
    sigma_yield: Optional[float] = None
    sigma_ultimate: Optional[float] = None
    shear_strength: Optional[float] = None
    beta_crippling: Optional[float] = None 
    """Material constant specific for crippling"""
    m_crippling: Optional[float] = None
    """Material constant specific for crippling"""
    g_crippling: Optional[float] = None
    """Material constant specific for crippling"""

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")


class Propeller(Parent):
    n_blades:float
    """The number of blades on the propellor"""    
    r_prop:float
    """"Propeller radius"""
    rpm_cruise:float 
    """"The rotations per minute during cruise flight"""
    xi_0:float
    """"Non-dimensional hub radius (r_hub/R) [-]"""
    chord_arr: Optional[list] = None
    """"Array with the chords at each station"""
    rad_arr: Optional[list] = None
    """"Radial coordinates for each station"""
    pitch_arr: Optional[list] = None
    """"Array with the pitch at each station"""
    tc_ratio:Optional[float] = None
    """Thickness over chord ratio of the airfoil"""    


    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class Wingbox(Parent):
    rib_loc:list
    """A list of the rib locations in meters"""    
    n_cell: int
    """"The amount of cells in the wingbox structure"""
    spar_loc_dimless:list
    """"The location of the spar over the chord so dimensionless. Length should be n_cell - 1 """
    t_sk_cell:list
    """The thickness of the skin in each cell, length should be equal to n_cell"""
    t_sp: float
    """The thickness of the spars"""
    area_str:float
    """"Area of the stringer"""
    str_cell: list
    """"List of stringers (both top and bottom) per cell. Length should be n_cell"""
    booms_sk: int
    """"Amount of booms on the skin """
    booms_spar: int
    """"Amount of booms on each spar"""

    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")
