"""
pyarbus: eyetracking data analysis

The module has several sub-modules:


- ``algorithms``: Contains various algorithms used in contemporary eyetracking
  data analsysis.

- ``utils``: Utility functions.

- ``viz``: Vizualization

All of the sub-modules will be imported as part of ``__init__``, so that users
have all of these things at their fingertips.
"""

__docformat__ = 'restructuredtext'

from version import  __version__

import algorithms

from data import *

import utils

from utils import velocity

#import viz

from testlib import test
