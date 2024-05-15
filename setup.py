import os
import inspect
import subprocess
from setuptools import setup, find_packages


is_released = True
version = '2024.13'


def git_version():
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_revision = out.strip().decode('ascii')
    except OSError:
        git_revision = "Unknown"

    return git_revision


def get_version_info(version, is_released):
    fullversion = version
    if not is_released:
        git_revision = git_version()
        fullversion += '.dev0+' + git_revision[:7]
    return fullversion


def write_version_py(version, is_released, filename='tuduam/version.py'):
    fullversion = get_version_info(version, is_released)
    with open("./tuduam/version.py", "wb") as f:
        f.write(b'__version__ = "%s"\n' % fullversion.encode())
    return fullversion


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    setupdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return open(os.path.join(setupdir, fname)).read()


#_____________________________________________________________________________

install_requires = [
        "numpy",
        "scipy",
        ]

#Trove classifiers
CLASSIFIERS = """\

Development Status :: 1 - Planning
Intended Audience :: Education
Intended Audience :: Science/Research
Intended Audience :: Developers
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Mathematics
Topic :: Education
Topic :: Software Development
Topic :: Software Development :: Libraries :: Python Modules
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: POSIX :: BSD
Programming Language :: Python :: 3.8
Programming Language :: Python :: 3.9
Programming Language :: Python :: 3.10
Programming Language :: Python :: 3.11
License :: OSI Approved :: BSD License

"""

fullversion = write_version_py(version, is_released)

data_files = [('', [
        'README.md',
        'LICENSE',
        ])]

s = setup(
    name = "tuduam",
    version = fullversion,
    author = "Saullo G. P. Castro, Damien Keijzer",
    author_email = "S.G.P.Castro@tudelft.nl",
    description = ("Python module related to the course on Urban Air Mobility at the faculty of Aeropsace Engineering, TU Delft"),
    license = "3-Clause BSD",
    keywords = "eVTOL design",
    url = "https://github.com/saullocastro/tuduam",
    packages=find_packages(),
    data_files=data_files,
    long_description=read('README.md'),
    long_description_content_type = 'text/markdown',
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    install_requires=install_requires,
    include_package_data=True,
)

