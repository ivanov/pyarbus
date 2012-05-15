from os import path
import numpy.testing as npt

# If you are running nosetests right now, you might want to use 'agg' as a
# backend:
import sys
if "nose" in sys.modules:
        import matplotlib
        matplotlib.use('agg')

import matplotlib.pyplot as plt
import pyarbus
import pyarbus.viz as viz
from pyarbus.tests import gz_test_file as test_file

def test_saccade_scatter():
    "Plotting of saccade scatter plot (polar coordinates)"
    el = pyarbus.read_eyelink_cached(test_file)
    viz.plot_saccade_scatter(el.r.saccades)
    plt.show()

def test_saccade_hist():
    "Plotting of saccade histogram (polar coordinates)"
    el = pyarbus.read_eyelink_cached(test_file)
    viz.plot_saccade_hist(el.r.saccades)
    plt.show()

def test_eyelink_to_plot_xyp():
    "Give useful feedback on common error for plot_xyp(el)"
    el = pyarbus.read_eyelink_cached(test_file)
    npt.assert_raises(ValueError, viz.plot_xyp, el)
