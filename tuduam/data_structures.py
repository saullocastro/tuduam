from pydantic import BaseModel, ConfigDict, FilePath, conlist, Field
from typing import Optional, List 
import numpy as np
import json

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
    stringer_height: Optional[float]  =  None
    stringer_width: Optional[float]  =  None
    stringer_thickness: Optional[float]  =  None
    skin_thickness: Optional[float]  =  None


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
    thrust: Optional[float] = None
    mass: Optional[float] = None
    power_grav_density: Optional[float] = None
    power_vol_density: Optional[float] = None
    x_rotor_locations: Optional[List] = None
    y_rotor_locations: Optional[List] = None


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
