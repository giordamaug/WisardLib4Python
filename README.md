# WisardLib4Python
 WiSARD implementation in C++ for Python binding

# Description

WiSARD was originally conceived as a pattern recognition device mainly focusing on image processing domain.
With ad hoc data transformation, WiSARD can also be used successfully as multiclass classifier in machine learning domain.

The WiSARD is a RAM-based neuron network working as an <i>n</i>-tuple classifier.
A WiSARD is formed by as many discriminators as the number of classes it has to discriminate between. 
Each discriminator consists of a set of <i>N</i> RAMs that, during the training phase, l
earn the occurrences of <i>n</i>-tuples extracted from the input binary vector (the <i>retina</i>).

In the WiSARD model, <i>n</i>-tuples selected from the input binary vector are regarded as the “features” of the input pattern to be recognised. It has been demonstrated in literature [14] that the randomness of feature extraction makes WiSARD more sensitive to detect global features than an ordered map which makes a single layer system sensitive to detect local features.

More information and details about the WiSARD neural network model can be found in Aleksander and Morton's book [Introduction to neural computing](https://books.google.co.uk/books/about/An_introduction_to_neural_computing.html?id=H4dQAAAAMAAJ&redir_esc=y&hl=it).

The WiSARD4WEKA package implements a multi-class classification method based on the WiSARD weightless neural model
for the Weka machine learning toolkit. A data-preprocessing filter allows to exploit WiSARD neural model 
training/classification capabilities on multi-attribute numeric data making WiSARD overcome the restriction to
binary pattern recognition.

For more information on the WiSARD classifier implemented in the WiSARD4WEKA package, see:

> Massimo De Gregorio and Maurizio Giordano (2018). 
> <i>An experimental evaluation of weightless neural networks for 
> multi-class classification</i>.
> Journal of Applied Soft Computing. Vol.72. pp. 338-354<br>

If you use this software, please cite it as:

<pre>
&#64;article{DEGREGORIO2018338,
 title = "An experimental evaluation of weightless neural networks for multi-class classification",
 journal = "Applied Soft Computing",
 volume = "72",
 pages = "338 - 354",
 year = "2018",
 issn = "1568-4946",
 doi = "https://doi.org/10.1016/j.asoc.2018.07.052",
 url = "http://www.sciencedirect.com/science/article/pii/S156849461830440X",
 author = "Massimo De Gregorio and Maurizio Giordano",
 keywords = "Weightless neural network, WiSARD, Machine learning"
}
</pre>

## Requirements

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

## Library testing

```
$ python test.py
```


## Notebooks

### pattern recognition
To run the ``test_digits.py`` example notebook you need the following python packages:

+ sklearn
+ PIL

### image recognition for object following
To run the ``test_follower.py`` example notebook you need the following python packages:

+ opencv
+ opencv_jupyter_ui

### machine learning clssifier and regressor
To run the ``test_classifier.py`` or ``test_regressor.py`` example notebooks you need the following python packages:

+ pandas
+ sklearn
+ matplotlib

All notebooks run in Google Colab by launching them with the button on top of the file.
