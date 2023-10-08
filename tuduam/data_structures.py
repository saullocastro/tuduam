from pydantic import BaseModel, ConfigDict, FilePath
from typing import Optional, List
import numpy as np
import json

#TODO Create documentation usig sphinx-pydantic plugin

class Parent(BaseModel):
    model_config = ConfigDict(extra='forbid',
                              validate_assignment=True)

class SingleWing(Parent):
    aspect_ratio: float  
    quarterchord_sweep: float  
    surface: Optional[float]  =  None
    span: Optional[float]  =  None
    chord_root: Optional[float]  =  None
    chord_tip: Optional[float]  =  None
    chord_mac: Optional[float]  =  None             # Mean Aerodynamic chord
    x_lemac: Optional[float]  =  None               # x position (longitudinal) leading edge mean aerodynamic chord
    y_mac: Optional[float]  =  None               # y position leading edge mean aerodynamic chord
    mass_wing: Optional[float]  =  None               # Mass of both wings

    #--------------wingbox--------------
    wingbox_front: Optional[float]  =  None
    wingbox_rear: Optional[float]  =  None


    @classmethod
    def load_from_json(cls, file_path:FilePath):
        with open(file_path) as jsonFile:
            data = json.load(jsonFile)
        try:
            return cls(**data)
        except:
            raise Exception(f"There was an error when loading in {cls}")

class Airfoil(Parent):
    thick_to_chord: Optional[float] = None

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
    mass_fuselage: Optional[float] = None

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
    n_engines: int 
    power_grav_density: Optional[float] = None
    power_vol_density: Optional[float] = None
    x_rotor_locatons: Optional[List] = None
    y_rotor_locatons: Optional[List] = None


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
    mission_energy: Optional[float] = None
    n_ult: Optional[float] = None
    v_cruise: Optional[float] = None

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
    poisson: Optional[float] = None
    density: Optional[float] = None
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
