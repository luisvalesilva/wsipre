wsipre
======

**wsipre** (**W**\ hole-**S**\ lide **I**\ mage **PRE**\ processing) is a small
Python package to handle whole-slide images (WSI; also known as
`virtual slides`_) with region-level annotations.

**wsipre** is a wrapper of the `OpenSlide Python`_ package, an interface to the
excellent `OpenSlide`_ C library which allows reading WSIs. **wsipre**
conserves OpenSlide Python's API and extends it to handle WSI annotations and
to perform processing tasks. The underlying objective is the preparation of
WSIs for Machine Learning (particularly Deep Learning).

You may also want to check out `py-wsi`_, a different Python package providing
overlapping functionality.

.. _virtual slides: https://en.wikipedia.org/wiki/Virtual_slide
.. _py-wsi: https://github.com/ysbecca/py-wsi/ 

User guide
==========

.. toctree::
    :maxdepth: 1

    quick_intro.rst
    Jupyter notebook demo <https://github.com/luisvalesilva/wsipre/blob/master/demo.ipynb>
    beyondwsipre.rst


Installation
============

Main dependencies
~~~~~~~~~~~~~~~~~

**wsipre** was developed in Python version 3.7.1. It has not been tested with
earlier versions, but it should generally work with Python version 3.*.
The main requirements are the following packages (the versions used for
development are listed):

    * `OpenSlide`_ C library (v3.4.1)
    * `OpenSlide Python`_ (v1.1.1)
    * `Numpy`_ (v1.15.4)
    * `OpenCV-Python`_ (v3.4.4.19)


.. note:: Some linux package managers currently distribute an outdated version
   of the `OpenSlide`_ C library: version 3.4.0. This version lacks support for
   some recent WSI formats, displaying wrong tiled downsampled views of the
   slides and crashing upon reading some regions. To avoid these problems
   please make sure you have **OpenSlide version 3.4.1**.
   
   You can check the installed version by running the following code in a
   Python interpreter: ::
   
       >>> import openslide
       >>> openslide.__library_version__
       '3.4.1'


.. _OpenSlide: https://openslide.org/
.. _OpenSlide Python: https://openslide.org/api/python/
.. _Numpy: http://www.numpy.org/
.. _OpenCV-Python: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html


User installation
~~~~~~~~~~~~~~~~~

**wsipre** can be installed from PyPI: ::
   
   pip install wsipre 

The source code is hosted on `GitHub`_.

.. _GitHub: https://github.com/luisvalesilva/wsipre/


License
=======

This project is licensed under the terms of the MIT license. See `LICENSE`_
file for details.

.. _LICENSE: https://github.com/luisvalesilva/wsipre/LICENSE


Package reference
=================

.. toctree::
   :maxdepth: 1

   slide
   show 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
