import numpy as np
import pyarbus

def test_velocity():
    "Velocity calculations from x,y data"
    a = np.arange(0.,20,2.)
    vel = pyarbus.velocity(a,a)
    # velocity calculation should mask out invalid data, but keep the length
    # the same
    assert len(a) == len(vel)
    vel_c = pyarbus.velocity(a,a, use_central=True)
    assert len(a) == len(vel_c)
    # first and last value must be masked out when using central difference
    assert vel.mask.sum() == 2
    vel_diff = pyarbus.velocity(a,a, use_central=False)
    assert len(a) == len(vel_diff)
    # only first value must be masked out when using np.diff
    assert vel_diff.mask.sum() == 1
    # since signal is linear - the resulting velocity (which is in
    # length units/sample) should be the same
    assert vel_diff.mean() == vel_c.mean()
    
    # check sampling rate works
    vel_10 = pyarbus.velocity(a,a, use_central=False, sampling_rate=10)
    vel_10c = pyarbus.velocity(a,a, use_central=True, sampling_rate=10)
    assert vel_10.mean() == vel_10c.mean()
    assert vel_10.mean() == vel_diff.mean()*10
