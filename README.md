TU Delft, Urban Air Mobility
---

Github Actions status:
[![Actions Status](https://github.com/saullocastro/tuduam/actions/workflows/pytest.yml/badge.svg)](https://github.com/saullocastro/tuduam/actions/workflows/pytest.yml)

Coverage status:
[![codecov](https://codecov.io/gh/saullocastro/tuduam/graph/badge.svg?token=QG08Lm2vwL)](https://codecov.io/gh/saullocastro/tuduam)


TU Delft

Urban Air Mobility


This Python module can be used by the students to assist in the course assignments.

Documentation
-------------

The documentation is available on: https://saullocastro.github.io/tuduam/


License
-------
Distrubuted under the 3-Clause BSD license
(https://raw.github.com/saullocastro/tuduam/master/LICENSE).

Contact: S.G.P.Castro@tudelft.nl




# For the developers


### Building documentation locally 

The documentation is build using Sphinx , so as always you can use the `make` file in the doc directory to build it. There are some points to look out for though:

- Make sure you have all packages listed in `requirements_doc.txt` before attempting to build. ( use `pip install -r requirements_doc.text`)

-  [Myst-nb](https://myst-nb.readthedocs.io/en/latest/quickstart.html) is used to embed the notebooks into the documentation. As a consequence, all cells in the  notebooks that are in the `doc/source` directory get run. **Thus only put notebooks there that are in working order**, otherwise it will crash.

- Another consequence of the previous bullet point is that the directory `doc/source/notebooks/tuduam` gets created everytime you build. This is because most notebooks used the `!git clone` command to access some data from this repo. **This needs to be deleted before making another build.** Otherwise, the entire repository is scanned again since it is now in the `source` directory.






