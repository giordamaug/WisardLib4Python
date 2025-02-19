# WisardLib4Python
 WiSARD implementation in C++ for Python binding

## Requirementa

+ pybind11 is a required package for compiling and binding C++ modules into python.
+ a C++ compiler with support for C++14 (which is no the standard for Clang, and GCC compiler)

```
$ pip install pybind11
```

## Installation

```
$ python setup.py build_ext --inplace
```

This will create the library implemmenting the `wisard` python module to bind in yor applications.
-- library name example is `wisard.cpyton-<XY>-<arch>.so` (or `.dll`) --

