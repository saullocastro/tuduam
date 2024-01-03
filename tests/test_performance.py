import sys
import os
import pytest
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam as tud
import numpy as np

def test_isa():
    h1 = 2000
    h2  = 4200

    isa1 = tud.ISA(h1)
    isa2 = tud.ISA(h2)

    assert np.isclose(isa1.temperature(), 275.15) 
    assert np.isclose(isa1.pressure(), 79495.215511, rtol=0.001) 
    assert np.isclose(isa1.density(), 1.0064902545, rtol=0.001) 
    assert np.isclose(isa1.soundspeed(), 332.52922598, rtol=0.001) 
    assert np.isclose(isa1.viscosity_dyn(), 0.000017464491280, atol=1e-7) 

    assert np.isclose(isa2.temperature(), 260.85, rtol=0.001) 
    assert np.isclose(isa2.pressure(), 60050.515656, rtol=0.001) 
    assert np.isclose(isa2.density(), 0.80198085377, rtol=0.001) 
    assert np.isclose(isa2.soundspeed(), 323.77289119, rtol=0.001) 
    assert np.isclose(isa2.viscosity_dyn(), 0.000016726148639, atol=1e-7) 

    with pytest.raises(ValueError):
        res = tud.ISA(11000.0000001)
