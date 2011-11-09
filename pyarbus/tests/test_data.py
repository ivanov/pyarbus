from os import path
import pyarbus

pyarbus_path = path.dirname(pyarbus.__file__)
data_path = path.join(pyarbus_path,'data')
def test_read_eyelink():
    "Read a short eyelink file with discontinuous time points"
    test_file = path.join(data_path,'pi_short.asc')
    return pyarbus.read_eyelink(test_file)

def test_read_eyelink_large():
    "Read a gunzipped .asc file (~10MB)"
    test_file = path.join(data_path,'pi.asc.gz')
    return pyarbus.read_eyelink(test_file)


def test_cached_read():
    "Read a file (and cache it)"
    test_file = path.join(data_path,'pi_short.asc')
    import time
    tic = time.time()
    el = pyarbus.read_eyelink_cached(test_file)
    toc = time.time()
    total_time = toc - tic
    # Here, we add a dummy attribute to the object - it should still be there
    # after we re-read test_file from cache, since we're not modifying the
    # object that's stored in the cache.
    el.__extra_dummy = 10
    tic = time.time()
    el = pyarbus.read_eyelink_cached(test_file)
    toc = time.time()
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
    del(pyarbus.data._cache[test_file])
