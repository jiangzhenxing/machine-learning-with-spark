from setuptools import find_packages
from distutils.core import setup
import setuptools

setup(  name='spark-test',
        version = '0.1',
        packages = find_packages('src'),
        package_dir = {'':'src'},
        py_modules = ['__main__','topk','wordcount'],
        data_files = [('config', ['etc/config.txt']),]
)
