"""pyarbus version/release information"""

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
#_version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

description = "pyarbus: eyetracking data analysis"

# Note: this long_description is actually a copy/paste from the top-level
# README.txt, so that it shows up nicely on PyPI.  So please remember to edit
# it only in one place and sync it correctly.
long_description = """
===================================================
 pyarbus: eyetracking data analysis 
===================================================

pyarbus is a library of tools and algorithms for the analysis of time-series data
from eyetracking experiments. 

Website and mailing list
========================

Current information can always be found at the pyarbus `website`_. Questions and
comments can be directed to `Paul Ivanov`_. 

.. _website: http://pyarbus.pirsquared.org/
.. _Paul Ivanov: http://pirsquared.org

Code
====

You can find our sources and single-click downloads:

* `Main repository`_ on Github.
* Documentation_ for all releases and current development tree.
* Download as a tar/zip file the `current trunk`_.
* Downloads of all `available releases`_.

.. _main repository: http://github.com/ivanov/pyarbus
.. _Documentation: http://pyarbus.pirsquared.org
.. _current trunk: http://github.com/ivanov/pyarbus/archives/master
.. _available releases: http://github.com/ivanov/pyarbus/downloads


License information
===================

pyarbus is licensed under the terms of the new BSD license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2009-2011, Paul Ivanov
All rights reserved.
"""

NAME = "pyarbus"
MAINTAINER = "Paul Ivanov"
MAINTAINER_EMAIL = "pi@berkeley.edu"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://pyarbus.pirsquared.org"
DOWNLOAD_URL = "http://github.com/ivanov/pyarbus/downloads"
LICENSE = "Simplified BSD"
AUTHOR = "Paul Ivanov"
AUTHOR_EMAIL = "pi@berkeley.edu"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGES = ['pyarbus',
            'pyarbus.tests',
            ]
PACKAGE_DATA = {"pyarbus": ["LICENSE","tests/*.txt", "tests/*.npy",
                                  "data/*.txt", "data/*.csv"]}
REQUIRES = ["numpy", "matplotlib", "nitime"]
