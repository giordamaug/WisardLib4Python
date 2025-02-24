from setuptools import setup, Extension
import pybind11

# Define the extension module
ext_modules = [
    Extension(
        'wisard',  # Name of the module
        #['wisard_v.cpp'],  # Source files
        ['wisard.cpp', 'ram.cpp'],  # Source files
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True)],  # Include pybind11 headers
        language='c++',  # Set the language to C++
        extra_compile_args=['-std=c++14'],  # Additional compilation flags
    ),
]

# Setup configuration
setup(
    name='WiSARD',
    version='0.1',
    description='WiSARD Classifier C++/Python binding',
    ext_modules=ext_modules,
    zip_safe=False,
)
