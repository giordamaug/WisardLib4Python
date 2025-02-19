# WisardLib4Python
 WiSARD implementation in C++ for Python binding

## Requirementa

+ Python version 3.x
+ pybind11 is a required package for compiling and binding C++ modules into python.
+ numpy package
+ a C++ compiler with support for C++14 (which is no the standard for Clang, and GCC compiler)

```
$ pip install pybind11
$ pip install numpy
```

## Installation

```
$ python setup.py build_ext --inplace
```

This will create the library implemmenting the `wisard` python module to bind in yor applications.
-- library name example is `wisard.cpyton-<XY>-<arch>.so` (or `.dll`) --

**Please be sure that the compiler has the same version of the one used to build Python.**

## Test

```
$ python test.py
```
