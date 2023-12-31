import sys
import os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..")))

import tuduam as tud

def test_load_json():
    path = os.path.realpath(os.path.join(os.path.dirname(__file__), "setup", "test_load_json.json"))
    res = tud.Material.load_from_json(path)
    assert isinstance(res, tud.Parent)
