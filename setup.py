#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='pytrackers',
    version='0.1.dev0',
    packages=['tracking','tracking.feature_based','tracking.segmentation','utils'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    install_requires=['matplotlib>=1.5.3',
    ],
    long_description=open('README.txt').read(),
)