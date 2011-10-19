from os import path
import matplotlib.pyplot as plt
import pyarbus
import pyarbus.viz as viz
from .test_data import data_path

def test_saccade_scatter():
    test_file = path.join(data_path,'pi.asc.gz')
    el = pyarbus.read_eyelink(test_file)
    viz.plot_saccade_scatter(el.r.saccades)
    plt.show()

def test_saccade_hist():
    test_file = path.join(data_path,'pi.asc.gz')
    el = pyarbus.read_eyelink(test_file)
    viz.plot_saccade_hist(el.r.saccades)
    plt.show()
