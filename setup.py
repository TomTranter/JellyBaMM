import os
import sys
from distutils.util import convert_path

sys.path.append(os.getcwd())

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

main_ = {}
ver_path = convert_path('ecm/__init__.py')
with open(ver_path) as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line, main_)

setup(
    name='ecm',
    description='Equivalent Circuit Model for battery modelling using PyBaMM',
    version=main_['__version__'],
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Physics'],
    packages=['ecm'],
    install_requires=['pybamm',
                      'openpnm',],
    author='Tom Tranter',
    author_email='t.g.tranter@gmail.com',
    url='https://ecm.readthedocs.io/en/latest/',
    project_urls={
        'Documentation': 'https://ecm.readthedocs.io/en/latest/',
        'Source': 'https://github.com/TomTranter/pybamm_pnm',
        'Tracker': 'https://github.com/TomTranter/pybamm_pnm/issues',
    },
)
