from os import path
import numpy as np
import numpy.testing as npt
import nitime
import pyarbus

# these can be used in test
gz_test_file = pyarbus.data.get_sample_data_filename()
short_test_file = pyarbus.data.get_sample_data_filename(short=True)

def test_read_eyelink():
    "Read a short eyelink file with discontinuous time points"
    return pyarbus.read_eyelink(short_test_file)

def test_get_sample_data():
    el = pyarbus.data.get_sample_data()
    npt.assert_equal(el.__class__, pyarbus.data.Eyelink, "sample data broken")


def test_read_eyelink_large():
    "Read a gunzipped .asc file (~10MB)"
    el = pyarbus.read_eyelink(gz_test_file)
    # cache this for viz tests which use this file, to save time
    pyarbus.data._cache[gz_test_file] = el
    return el


def test_cached_read():
    "Read a file (and cache it)"
    import time
    #del(pyarbus.data._cache[short_test_file])
    tic = time.time()
    el = pyarbus.read_eyelink_cached(short_test_file)
    toc = time.time()
    total_time = toc - tic
    # Here, we add a dummy attribute to the object - it should still be there
    # after we re-read short_test_file from cache, since we're not modifying
    # the object that's stored in the cache.
    el.__extra_dummy = 10
    tic = time.time()
    el = pyarbus.read_eyelink_cached(short_test_file)
    # windows time.time is only ms resolution:
    # http://mail.python.org/pipermail/python-list/2007-January/1121263.html
    toc = time.time() + 0.001
    cached_time = toc-tic
    speedup = total_time / cached_time
    el.__extra_dummy # should *not* raise an attribute error
    print "speedup = %.3fx" % speedup
    # the speedup in general depends on the file size, disk speed, and how fast
    # we read it, etc, but for the file on this test, I see speedups of ~ 500x
    # on my machine, so I'm just putting a conservative speedup assertion of at
    # least 20x test in here for good measure. NOTE: this test might actually
    # fail if we write a much faster file reader, such that caching it is not
    # longer as much of a performance boost as it was before, but that' a nice
    # bug to have. Also, for test this small, pyarbus.data.log.level
    # actually makes a big difference in the speedup, since printing to stderr
    # and capturing that output has overhead for nose.
    assert(speedup > 20.)
    # remove it from the cache
    del(pyarbus.data._cache[short_test_file])

def test_missing_data():
    "Missing data should be filled with nans"
    el = pyarbus.read_eyelink(short_test_file)
    time = el.eye_used.time
    diffs = np.unique(np.diff(time))
    assert diffs == nitime.TimeArray([1], time_unit='ms')
    assert nitime.TimeArray(len(time)-1, time_unit='ms') == (time[-1]-time[0])
