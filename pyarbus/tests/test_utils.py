import numpy as np
import numpy.testing as npt
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
    vel_10
    vel_10c = pyarbus.velocity(a,a, use_central=True, sampling_rate=10)
    assert vel_10.mean() == vel_10c.mean()
    assert vel_10.mean() == vel_diff.mean()*10

    # check that xres and yres are respected
    xr = yr = np.ones_like(a) * 10
    vel_10r = pyarbus.velocity(a,a, use_central=False, sampling_rate=10,
            xres=xr, yres=yr)
    vel_10cr = pyarbus.velocity(a,a, use_central=True, sampling_rate=10,
            xres=xr, yres=yr)
    # since sampling rate and xres/yres are the same, we should get
    npt.assert_almost_equal(vel_diff, vel_10r)
    npt.assert_equal(vel_diff.mask, vel_10r.mask)
    npt.assert_almost_equal(vel_c, vel_10cr)
    npt.assert_equal(vel_c.mask, vel_10cr.mask)
def test_accel():
    "Acceleration calculations from velocity"
    v = np.arange(10.)
    ac = pyarbus.acceleration(v, use_central=True)
    npt.assert_equal(ac, np.ones_like(v))
    mask = np.zeros(10, dtype=np.bool)
    mask[0] = mask[-1] = True
    npt.assert_equal(ac.mask, mask)

    anc = pyarbus.acceleration(v, use_central=False)
    mask[-1] = False
    npt.assert_equal(anc,np.ones_like(v))
    npt.assert_equal(anc.mask, mask)
